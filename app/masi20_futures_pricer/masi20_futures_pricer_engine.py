# -*- coding: utf-8 -*-

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import datetime, timedelta
import io
import os
import re
import unicodedata
from typing import Mapping, Optional

import numpy as np
import pandas as pd


# ============================================================================
# 1. CHARGEMENT DES POIDS
# ============================================================================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
FACTEUR_FILE = os.path.join(APP_DIR, "Flottant_plafonnements_masi20.xlsx")


def load_index_weights(year_label: str = "2025-2026") -> pd.DataFrame:
    """
    Charge les poids reels des titres dans l'indice MASI 20.
    """
    df = pd.read_excel(FACTEUR_FILE)
    year_col = [c for c in df.columns if "Ann" in str(c) or "ann" in str(c)][0]
    df = df.rename(columns={year_col: "Annee"})

    df_year = df[df["Annee"] == year_label].copy()
    if df_year.empty:
        raise ValueError(f"Annee '{year_label}' non trouvee.")

    df_year = df_year.reset_index(drop=True)
    df_year["Ticker_Short"] = df_year["Valeur"].str.replace(" MC Equity", "", regex=False)
    return df_year[["Valeur", "Ticker_Short", "Flottant", "Plafonnement"]].copy()


def get_available_years() -> list:
    """Retourne les annees disponibles dans le fichier de facteurs."""
    df = pd.read_excel(FACTEUR_FILE)
    year_col = [c for c in df.columns if "Ann" in str(c) or "ann" in str(c)][0]
    return sorted(df[year_col].dropna().unique().tolist())


# ============================================================================
# 2. COURBE DES TAUX SANS RISQUE
# ============================================================================

MONEY_MARKET_BASIS = "money_market"
ACTUARIAL_BASIS = "actuarial"
SHORT_TERM_CUTOFF_DAYS = 365
MONEY_MARKET_DAY_COUNT = 360.0
ACTUARIAL_DAY_COUNT = 365.0


@dataclass(frozen=True)
class RatePillar:
    maturity_days: int
    market_rate: float
    basis: str
    value_date: Optional[pd.Timestamp] = None
    maturity_date: Optional[pd.Timestamp] = None


@dataclass(frozen=True)
class RiskFreeCurve:
    pillars: tuple[RatePillar, ...]
    source_name: str = "Courbe des taux"
    interpolation_method: str = "linear_rate"

    def __post_init__(self) -> None:
        if not self.pillars:
            raise ValueError("La courbe des taux est vide.")

        sorted_pillars = tuple(sorted(self.pillars, key=lambda pillar: pillar.maturity_days))
        if any(pillar.maturity_days <= 0 for pillar in sorted_pillars):
            raise ValueError("Toutes les maturites de la courbe doivent etre strictement positives.")

        maturity_days = [pillar.maturity_days for pillar in sorted_pillars]
        if len(set(maturity_days)) != len(maturity_days):
            raise ValueError("La courbe contient des maturites dupliquees.")

        object.__setattr__(self, "pillars", sorted_pillars)

    @property
    def reference_date(self) -> Optional[pd.Timestamp]:
        dates = {pillar.value_date for pillar in self.pillars if pillar.value_date is not None}
        if len(dates) == 1:
            return next(iter(dates))
        return None

    @staticmethod
    def _normalize_timestamp(date_value: Optional[object]) -> Optional[pd.Timestamp]:
        if date_value is None:
            return None

        timestamp = pd.Timestamp(date_value)
        if pd.isna(timestamp):
            raise ValueError("La date de valorisation est invalide.")
        return timestamp.normalize()

    def projected_pillars(self, valuation_date: Optional[object] = None) -> list[tuple[RatePillar, int]]:
        if valuation_date is not None:
            self._normalize_timestamp(valuation_date)
        projected: list[tuple[RatePillar, int]] = []

        for pillar in self.pillars:
            effective_days = int(pillar.maturity_days)
            projected.append((pillar, effective_days))

        if not projected:
            raise ValueError("Aucun pilier de la courbe n'est disponible a la date de valorisation.")

        projected.sort(key=lambda item: item[1])
        return projected

    def _homogenized_pillar_rates(
        self,
        target_basis: str,
        valuation_date: Optional[object] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        projected = self.projected_pillars(valuation_date=valuation_date)
        maturities = np.array([effective_days for _, effective_days in projected], dtype=float)
        rates = np.array(
            [
                convert_rate_between_bases(
                    pillar.market_rate,
                    pillar.maturity_days,
                    source_basis=pillar.basis,
                    target_basis=target_basis,
                )
                for pillar, effective_days in projected
            ],
            dtype=float,
        )
        return maturities, rates

    @staticmethod
    def _linear_interpolate_or_extrapolate(days: float, maturities: np.ndarray, rates: np.ndarray) -> float:
        if len(maturities) == 1:
            return float(rates[0])

        if days <= maturities[0]:
            t1, t2 = maturities[0], maturities[1]
            r1, r2 = rates[0], rates[1]
        elif days >= maturities[-1]:
            t1, t2 = maturities[-2], maturities[-1]
            r1, r2 = rates[-2], rates[-1]
        else:
            upper_idx = int(np.searchsorted(maturities, days, side="right"))
            lower_idx = upper_idx - 1
            t1, t2 = maturities[lower_idx], maturities[upper_idx]
            r1, r2 = rates[lower_idx], rates[upper_idx]

        if t2 == t1:
            return float(r1)
        return float(r1 + (r2 - r1) * ((days - t1) / (t2 - t1)))

    def rate_for_days(
        self,
        days_to_maturity: float,
        target_maturity: Optional[float] = None,
        valuation_date: Optional[object] = None,
    ) -> float:
        days = _validate_days(days_to_maturity, allow_zero=True)
        if days == 0:
            return 0.0

        target_days = days if target_maturity is None else _validate_days(target_maturity, allow_zero=True)
        target_basis = rate_basis_for_days(target_days)
        maturities, rates = self._homogenized_pillar_rates(target_basis, valuation_date=valuation_date)
        return self._linear_interpolate_or_extrapolate(days, maturities, rates)

    def pillar_dict_pct(
        self,
        target_maturity: Optional[float] = None,
        valuation_date: Optional[object] = None,
    ) -> dict[int, float]:
        if valuation_date is not None:
            self._normalize_timestamp(valuation_date)
        return {
            effective_days: convert_rate_between_bases(
                pillar.market_rate,
                pillar.maturity_days,
                source_basis=pillar.basis,
                target_basis=rate_basis_for_days(
                    effective_days if target_maturity is None else target_maturity
                ),
            )
            * 100.0
            for pillar, effective_days in self.projected_pillars(valuation_date=valuation_date)
        }


def _validate_days(days_to_maturity: float, allow_zero: bool = False) -> float:
    if pd.isna(days_to_maturity):
        raise ValueError("La maturite doit etre renseignee.")

    days = float(days_to_maturity)
    if days < 0:
        raise ValueError("La maturite ne peut pas etre negative.")
    if days == 0 and not allow_zero:
        raise ValueError("La maturite doit etre strictement positive.")
    return days


def rate_basis_for_days(days_to_maturity: float) -> str:
    days = _validate_days(days_to_maturity, allow_zero=True)
    if days <= SHORT_TERM_CUTOFF_DAYS:
        return MONEY_MARKET_BASIS
    return ACTUARIAL_BASIS


def basis_label(basis: str) -> str:
    if basis == MONEY_MARKET_BASIS:
        return "Monetaire ACT/360"
    if basis == ACTUARIAL_BASIS:
        return "Actuariel ACT/365"
    raise ValueError(f"Convention inconnue: {basis}")


def basis_label_for_days(days_to_maturity: float) -> str:
    return basis_label(rate_basis_for_days(days_to_maturity))


def year_fraction(days_to_maturity: float, basis: Optional[str] = None) -> float:
    days = _validate_days(days_to_maturity, allow_zero=True)
    if days == 0:
        return 0.0

    basis_to_use = basis or rate_basis_for_days(days)
    if basis_to_use == MONEY_MARKET_BASIS:
        return days / MONEY_MARKET_DAY_COUNT
    if basis_to_use == ACTUARIAL_BASIS:
        return days / ACTUARIAL_DAY_COUNT
    raise ValueError(f"Convention inconnue: {basis_to_use}")


def capitalization_factor(rate_decimal: float, days_to_maturity: float, basis: Optional[str] = None) -> float:
    """
    Facteur de capitalisation compatible avec la convention du marche.
    """
    days = _validate_days(days_to_maturity, allow_zero=True)
    if days == 0:
        return 1.0

    basis_to_use = basis or rate_basis_for_days(days)
    tau = year_fraction(days, basis_to_use)
    rate = float(rate_decimal)

    if basis_to_use == MONEY_MARKET_BASIS:
        factor = 1.0 + rate * tau
        if factor <= 0:
            raise ValueError(
                "Le taux monetaire est trop negatif pour cette maturite : le facteur de capitalisation devient non positif."
            )
        return factor

    if basis_to_use == ACTUARIAL_BASIS:
        if 1.0 + rate <= 0:
            raise ValueError("Un taux actuariel inferieur ou egal a -100% est invalide.")
        return (1.0 + rate) ** tau

    raise ValueError(f"Convention inconnue: {basis_to_use}")


def pricing_time_fraction(days_to_maturity: float) -> float:
    """Fraction de temps reglementaire utilisee dans la formule exp((r-d)t)."""
    days = _validate_days(days_to_maturity, allow_zero=True)
    if days == 0:
        return 0.0
    return days / MONEY_MARKET_DAY_COUNT


def convert_rate_to_pricing_formula_rate(
    rate_decimal: float,
    days_to_maturity: float,
    basis: Optional[str] = None,
) -> float:
    """
    Convertit un taux quote dans sa convention de marche vers le taux a
    injecter dans la formule reglementaire :

        F = S * exp((r - d) * t), avec t = jours / 360

    L'idee est de conserver le facteur de capitalisation implicite du taux
    de marche sur la maturite consideree, puis d'en deduire le taux continu
    equivalent sur la base reglementaire ACT/360.
    """
    days = _validate_days(days_to_maturity, allow_zero=True)
    if days == 0:
        return 0.0

    source_basis = basis or rate_basis_for_days(days)
    factor = capitalization_factor(rate_decimal, days, basis=source_basis)
    tau_formula = pricing_time_fraction(days)
    return float(np.log(factor) / tau_formula)


def convert_rate_between_bases(
    rate_decimal: float,
    days_to_maturity: float,
    source_basis: Optional[str] = None,
    target_basis: Optional[str] = None,
) -> float:
    """
    Convertit un taux d'une convention vers une autre a maturite identique.

    Formules equivalentes :
        ta = (1 + tm * n / 360) ** (365 / n) - 1
        tm = ((1 + ta) ** (n / 365) - 1) * (360 / n)
    """
    days = _validate_days(days_to_maturity, allow_zero=True)
    if days == 0:
        return 0.0

    source_basis_to_use = source_basis or rate_basis_for_days(days)
    target_basis_to_use = target_basis or rate_basis_for_days(days)
    rate = float(rate_decimal)

    if source_basis_to_use == target_basis_to_use:
        return rate

    capitalization = capitalization_factor(rate, days, basis=source_basis_to_use)
    if target_basis_to_use == MONEY_MARKET_BASIS:
        return (capitalization - 1.0) * (MONEY_MARKET_DAY_COUNT / days)
    if target_basis_to_use == ACTUARIAL_BASIS:
        return capitalization ** (ACTUARIAL_DAY_COUNT / days) - 1.0
    raise ValueError(f"Convention inconnue: {target_basis_to_use}")


def _normalize_text(value: object) -> str:
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", ascii_text.lower()).strip("_")


def _maybe_repair_mojibake(text: str) -> str:
    if "Ã" not in text and "Â" not in text:
        return text
    try:
        repaired = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text
    return repaired if repaired.count("Ã") < text.count("Ã") else text


def _parse_localized_number(value: object, percent: bool = False) -> float:
    if pd.isna(value):
        raise ValueError("valeur manquante")

    text = str(value).strip()
    if not text or text.lower() in {"nan", "na", "none", "-"}:
        raise ValueError("valeur manquante")

    text = text.replace("\xa0", " ").replace(" ", "")
    has_percent = "%" in text or percent
    cleaned = text.replace("%", "").replace(",", ".")
    number = float(cleaned)
    return number / 100.0 if has_percent else number


def _find_column(columns: list[str], *tokens: str) -> Optional[str]:
    normalized_map = {column: _normalize_text(column) for column in columns}
    for column, normalized in normalized_map.items():
        if all(token in normalized for token in tokens):
            return column
    return None


def read_market_rate_csv(file_content: str) -> pd.DataFrame:
    """
    Lit le CSV des taux de reference du marche secondaire.

    Colonnes attendues :
        - Date d'echeance
        - Taux moyen pondere
        - Date de la valeur
    """
    if not file_content or not file_content.strip():
        raise ValueError("Le fichier CSV de taux est vide.")

    repaired_content = _maybe_repair_mojibake(file_content)

    try:
        raw_df = pd.read_csv(
            io.StringIO(repaired_content),
            sep=";",
            skiprows=2,
            dtype=str,
            keep_default_na=False,
        )
    except Exception as exc:
        raise ValueError(f"Impossible de lire le fichier CSV de taux: {exc}") from exc

    if raw_df.empty:
        raise ValueError("Le fichier CSV de taux ne contient aucune ligne exploitable.")

    maturity_col = _find_column(raw_df.columns.tolist(), "date", "echeance")
    rate_col = _find_column(raw_df.columns.tolist(), "taux", "moyen")
    value_col = _find_column(raw_df.columns.tolist(), "date", "valeur")
    transaction_col = _find_column(raw_df.columns.tolist(), "transaction")

    missing_columns = []
    if maturity_col is None:
        missing_columns.append("Date d'echeance")
    if rate_col is None:
        missing_columns.append("Taux moyen pondere")
    if value_col is None:
        missing_columns.append("Date de la valeur")
    if missing_columns:
        raise ValueError(
            "Colonnes obligatoires introuvables dans le CSV: " + ", ".join(missing_columns) + "."
        )

    issues: list[str] = []
    records: list[dict] = []

    for row_idx, row in raw_df.iterrows():
        csv_line_number = row_idx + 4
        raw_maturity_date = str(row[maturity_col]).strip()
        if not raw_maturity_date:
            issues.append(f"Ligne {csv_line_number}: date d'echeance manquante.")
            continue
        if _normalize_text(raw_maturity_date) == "total":
            continue

        raw_value_date = str(row[value_col]).strip()
        raw_rate = row[rate_col]

        try:
            maturity_date = pd.to_datetime(raw_maturity_date, dayfirst=True, errors="raise").normalize()
        except Exception:
            issues.append(
                f"Ligne {csv_line_number}: format de date d'echeance invalide ({raw_maturity_date})."
            )
            continue

        try:
            value_date = pd.to_datetime(raw_value_date, dayfirst=True, errors="raise").normalize()
        except Exception:
            issues.append(
                f"Ligne {csv_line_number}: format de date de valeur invalide ({raw_value_date})."
            )
            continue

        try:
            market_rate = _parse_localized_number(raw_rate, percent=True)
        except Exception:
            issues.append(f"Ligne {csv_line_number}: taux invalide ({raw_rate}).")
            continue

        maturity_days = int((maturity_date - value_date).days)
        if maturity_days <= 0:
            issues.append(
                f"Ligne {csv_line_number}: maturite non positive ({maturity_days} jours)."
            )
            continue

        source_basis = rate_basis_for_days(maturity_days)
        try:
            capitalization_factor(
                rate_decimal=market_rate,
                days_to_maturity=maturity_days,
                basis=source_basis,
            )
        except ValueError as exc:
            issues.append(f"Ligne {csv_line_number}: {exc}")
            continue

        transaction_amount = np.nan
        if transaction_col is not None:
            try:
                transaction_amount = _parse_localized_number(row[transaction_col], percent=False)
            except Exception:
                transaction_amount = np.nan

        records.append(
            {
                "Maturity Date": maturity_date,
                "Value Date": value_date,
                "Maturity Days": maturity_days,
                "Market Rate": market_rate,
                "Market Rate (%)": market_rate * 100.0,
                "Source Basis": basis_label(source_basis),
                "Source Basis Code": source_basis,
                "Transaction": transaction_amount,
            }
        )

    if not records:
        detail = " ".join(issues[:3]) if issues else "Aucune ligne valide detectee."
        raise ValueError(f"Aucune donnee de taux valide n'a ete extraite. {detail}")

    market_df = (
        pd.DataFrame(records)
        .sort_values(by=["Maturity Days", "Maturity Date"])
        .drop_duplicates(subset=["Maturity Days"], keep="last")
        .reset_index(drop=True)
    )
    market_df.attrs["warnings"] = issues
    return market_df


def build_market_curve_from_csv(
    file_content: str,
    source_name: str = "CSV marche",
) -> tuple[RiskFreeCurve, pd.DataFrame]:
    market_df = read_market_rate_csv(file_content)
    pillars = tuple(
        RatePillar(
            maturity_days=int(row["Maturity Days"]),
            market_rate=float(row["Market Rate"]),
            basis=str(row["Source Basis Code"]),
            value_date=row["Value Date"],
            maturity_date=row["Maturity Date"],
        )
        for _, row in market_df.iterrows()
    )
    return RiskFreeCurve(pillars=pillars, source_name=source_name), market_df


def build_risk_free_curve_from_dict(
    yield_curve: Mapping[int, float],
    source_name: str = "Courbe importee",
) -> RiskFreeCurve:
    curve_dict = dict(yield_curve)
    if not curve_dict:
        raise ValueError("La courbe des taux est vide.")

    pillars: dict[int, RatePillar] = {}
    for raw_days, raw_rate_pct in curve_dict.items():
        try:
            maturity_days = int(float(raw_days))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Maturite invalide dans la courbe: {raw_days}") from exc

        if maturity_days <= 0:
            raise ValueError(f"La maturite {maturity_days} doit etre strictement positive.")

        try:
            market_rate = float(raw_rate_pct) / 100.0
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Taux invalide pour {maturity_days} jours: {raw_rate_pct}") from exc

        basis = rate_basis_for_days(maturity_days)
        pillars[maturity_days] = RatePillar(
            maturity_days=maturity_days,
            market_rate=market_rate,
            basis=basis,
        )

    return RiskFreeCurve(
        pillars=tuple(sorted(pillars.values(), key=lambda pillar: pillar.maturity_days)),
        source_name=source_name,
    )


def build_flat_curve_from_manual_rate(
    rate_pct: float,
    anchor_days: int,
    source_name: str = "Taux manuel",
) -> RiskFreeCurve:
    anchor = max(1, int(anchor_days))
    return build_risk_free_curve_from_dict({anchor: float(rate_pct)}, source_name=source_name)


def ensure_risk_free_curve(
    yield_curve: Optional[object] = None,
    source_name: str = "Courbe des taux",
) -> RiskFreeCurve:
    if yield_curve is None:
        raise ValueError("Aucune courbe des taux n'a ete fournie.")
    if isinstance(yield_curve, RiskFreeCurve):
        return yield_curve
    if isinstance(yield_curve, Mapping):
        return build_risk_free_curve_from_dict(dict(yield_curve), source_name=source_name)
    raise TypeError("Le format de courbe des taux n'est pas supporte.")


def format_maturity_label(days_to_maturity: int) -> str:
    days = int(days_to_maturity)
    if days >= 365 and days % 365 == 0:
        years = days // 365
        return f"{years} an" if years == 1 else f"{years} ans"
    if days >= 30 and days % 30 == 0:
        months = days // 30
        return f"{months} mois"
    return f"{days} j"


def build_yield_curve_df(
    yield_curve: Optional[object] = None,
    target_maturity: Optional[float] = None,
    valuation_date: Optional[object] = None,
) -> pd.DataFrame:
    curve = ensure_risk_free_curve(yield_curve)
    if valuation_date is not None:
        curve._normalize_timestamp(valuation_date)
    records = []
    for pillar, effective_days in curve.projected_pillars(valuation_date=valuation_date):
        target_days = effective_days if target_maturity is None else target_maturity
        homogenized_rate = convert_rate_between_bases(
            pillar.market_rate,
            pillar.maturity_days,
            source_basis=pillar.basis,
            target_basis=rate_basis_for_days(target_days),
        )
        records.append(
            {
                "Date de valeur": pillar.value_date,
                "Date d'echeance": pillar.maturity_date,
                "Maturite courbe (jours)": effective_days,
                "Label": format_maturity_label(effective_days),
                "Convention cible": basis_label_for_days(
                    effective_days if target_maturity is None else target_maturity
                ),
                "Convention source": basis_label(pillar.basis),
                "Taux source (%)": pillar.market_rate * 100.0,
                "Taux converti (%)": homogenized_rate * 100.0,
            }
        )
    return pd.DataFrame(records)


def adapt_yield_curve(
    yield_curve: Optional[object],
    target_maturity: float,
    valuation_date: Optional[object] = None,
) -> dict:
    curve = ensure_risk_free_curve(yield_curve)
    return curve.pillar_dict_pct(target_maturity=target_maturity, valuation_date=valuation_date)


def interpolate_rate(
    days_to_maturity: float,
    yield_curve: Optional[object] = None,
    target_maturity: Optional[float] = None,
    valuation_date: Optional[object] = None,
) -> float:
    """
    Retourne le taux sans risque en decimal.

    Le taux est toujours re-exprime dans la convention cible liee a la
    maturite du future (ou, a defaut, a la maturite demandee), puis
    obtenu par interpolation lineaire ou extrapolation lineaire sur les
    deux piliers voisins homogenises. Les points BAM restent ancres sur
    leur maturite d'origine (date d'echeance - date de valeur), meme si
    une date de valorisation est fournie.
    """
    curve = ensure_risk_free_curve(yield_curve)
    return curve.rate_for_days(
        days_to_maturity,
        target_maturity=target_maturity,
        valuation_date=valuation_date,
    )


def parse_bam_csv(file_content: str) -> dict:
    """
    Compatibilite avec l'ancienne API: retourne {maturite_jours: taux_%}.
    """
    market_df = read_market_rate_csv(file_content)
    return {
        int(row["Maturity Days"]): float(row["Market Rate (%)"])
        for _, row in market_df.iterrows()
    }


# ============================================================================
# 3. TAUX DE DIVIDENDE
# ============================================================================

def compute_dividend_yield(
    stock_prices: dict,
    dividends: dict,
    weights: dict = None,
) -> tuple:
    """
    Calcule le taux de dividende pondere de l'indice MASI 20.
    Div Yield = Somme((Di / Ci) x pi)
    """
    if weights is None:
        tickers = sorted(set(stock_prices.keys()) | set(dividends.keys()))
    else:
        tickers = sorted(weights.keys())

    if not tickers:
        return 0.0, pd.DataFrame()

    if weights is None:
        weights = {ticker: 1.0 / len(tickers) for ticker in tickers}

    total_weight = sum(weights.get(ticker, 0) for ticker in tickers)
    if total_weight == 0:
        return 0.0, pd.DataFrame()

    records = []
    div_yield_total = 0.0

    for ticker in tickers:
        current_price = stock_prices.get(ticker, None)
        dividend = dividends.get(ticker, None)
        normalized_weight = weights.get(ticker, 0) / total_weight

        if current_price is not None and current_price > 0 and dividend is not None:
            div_yield_i = (dividend / current_price) * normalized_weight
        else:
            div_yield_i = 0.0

        div_yield_total += div_yield_i
        records.append(
            {
                "Ticker": ticker,
                "Cours (Ci)": current_price,
                "Dividende (Di)": dividend,
                "Di/Ci (%)": (dividend / current_price * 100.0)
                if current_price is not None and current_price > 0 and dividend is not None
                else 0.0,
                "Poids (pi)": normalized_weight,
                "Contribution (%)": div_yield_i * 100.0,
                "Donnees completes": current_price is not None and current_price > 0 and dividend is not None,
            }
        )

    return div_yield_total, pd.DataFrame(records)


# ============================================================================
# 4. PRICING DU FUTURE
# ============================================================================


def price_future(
    spot: float,
    risk_free_rate: float,
    dividend_yield: float,
    days_to_maturity: int,
    risk_free_basis: Optional[str] = None,
) -> dict:
    """
    Calcule le cours theorique du contrat a terme sur l'indice MASI 20.
    """
    if spot < 0:
        raise ValueError("Le spot ne peut pas etre negatif.")

    validated_days = _validate_days(days_to_maturity, allow_zero=True)
    risk_free_basis_to_use = risk_free_basis or rate_basis_for_days(validated_days)
    risk_free_rate_curve = float(risk_free_rate)
    # Le pricer utilise directement le taux issu de la courbe, sans
    # reconversion vers un taux continu equivalent.
    risk_free_rate_pricing = risk_free_rate_curve
    t = pricing_time_fraction(validated_days)

    cost_of_carry = risk_free_rate_pricing - float(dividend_yield)
    carry_factor = float(np.exp(cost_of_carry * t))
    future_price = float(spot) * carry_factor
    basis = future_price - float(spot)
    basis_pct = (basis / float(spot)) * 100.0 if spot > 0 else 0.0
    fair_value = float(spot) * (carry_factor - 1.0)

    return {
        "spot": float(spot),
        "future_price": future_price,
        "risk_free_rate_market": risk_free_rate_curve,
        "risk_free_rate_curve": risk_free_rate_curve,
        "risk_free_rate_pricing": risk_free_rate_pricing,
        "risk_free_basis": basis_label(risk_free_basis_to_use),
        "dividend_yield": float(dividend_yield),
        "cost_of_carry": cost_of_carry,
        "days_to_maturity": int(validated_days),
        "t_fraction": t,
        "basis": basis,
        "basis_pct": basis_pct,
        "fair_value_spread": fair_value,
        "carry_factor": carry_factor,
        "rate_basis": basis_label_for_days(validated_days),
    }


def generate_term_structure(
    spot: float,
    risk_free_rate_or_curve,
    dividend_yield: float,
    max_days: int = 360,
    step: int = 1,
    target_maturity: float = None,
    valuation_date: Optional[object] = None,
) -> pd.DataFrame:
    """
    Genere la structure par terme des futures.
    """
    records = []
    use_curve = isinstance(risk_free_rate_or_curve, (RiskFreeCurve, Mapping))

    for days in range(1, max_days + 1, step):
        if use_curve:
            rate = interpolate_rate(
                days,
                risk_free_rate_or_curve,
                target_maturity=target_maturity,
                valuation_date=valuation_date,
            )
        else:
            rate = float(risk_free_rate_or_curve)

        rate_basis = rate_basis_for_days(target_maturity if target_maturity is not None else days)
        result = price_future(
            spot,
            rate,
            dividend_yield,
            days,
            risk_free_basis=rate_basis,
        )
        records.append(
            {
                "Jours": days,
                "Taux courbe (%)": rate * 100.0,
                "Taux utilise (%)": result["risk_free_rate_pricing"] * 100.0,
                "Future": result["future_price"],
                "Basis": result["basis"],
                "Basis (%)": result["basis_pct"],
            }
        )

    return pd.DataFrame(records)


def _last_friday_of_month(year: int, month: int) -> datetime:
    friday_dates = [
        week[calendar.FRIDAY]
        for week in calendar.monthcalendar(year, month)
        if week[calendar.FRIDAY] != 0
    ]
    return datetime(year, month, friday_dates[-1])


def generate_maturity_schedule(
    reference_date: datetime = None,
    year: int = None,
    contract_count: int = 4,
) -> list:
    """
    Genere les echeances trimestrielles des contrats futures MASI 20.
    """
    if reference_date is None:
        reference_date = datetime.now()
    if contract_count <= 0:
        return []

    maturity_months = [3, 6, 9, 12]
    month_codes = {3: "MAR", 6: "JUI", 9: "SEP", 12: "DEC"}

    maturities = []
    current_year = year if year is not None else reference_date.year

    while True:
        for month in maturity_months:
            maturity_date = _last_friday_of_month(current_year, month)
            days_to = (maturity_date - reference_date).days
            if days_to > 0:
                ticker = f"FMASI20{month_codes[month]}{str(current_year)[-2:]}"
                maturities.append({"label": ticker, "date": maturity_date, "days": days_to})
                if year is None and len(maturities) >= contract_count:
                    return maturities

        if year is not None:
            return maturities
        current_year += 1

    return maturities


def compute_index_weights_from_caps(
    df_weights: pd.DataFrame,
    market_caps: dict,
) -> tuple[dict, pd.DataFrame]:
    """
    Calcule les poids de l'indice a partir des capitalisations flottantes plafonnees.

    Formule :
        poids_i = (Cap_i x flottant_i x plafonnement_i) / Somme_j(...)

    Aucune approximation n'est utilisee : sans capitalisation boursiere,
    les poids ne sont pas calcules.
    """
    def _normalize_factor(value: object) -> float:
        if pd.isna(value):
            return 0.0
        factor = float(value)
        if factor < 0:
            return 0.0
        return factor / 100.0 if factor > 1.0 else factor

    caps = {}
    records = []
    missing_market_caps = []

    for _, row in df_weights.iterrows():
        ticker = row.get("Ticker_Short", row.get("Ticker", row.get("Valeur", "")))
        ticker = str(ticker).replace(" MC Equity", "").strip()
        free_float = _normalize_factor(row.get("Flottant", 0))
        cap_factor = _normalize_factor(row.get("Plafonnement", 0))
        market_cap = market_caps.get(ticker, 0) or 0
        if market_cap <= 0:
            missing_market_caps.append(ticker)
        adjusted_cap = market_cap * free_float * cap_factor if market_cap > 0 else 0.0
        caps[ticker] = adjusted_cap
        records.append(
            {
                "Ticker": ticker,
                "Capitalisation brute": market_cap,
                "Flottant": free_float,
                "Plafonnement": cap_factor,
                "Capitalisation ajustee": adjusted_cap,
                "Capitalisation disponible": market_cap > 0,
            }
        )

    total_cap = sum(caps.values())
    details_df = pd.DataFrame(records)

    if missing_market_caps:
        if not details_df.empty:
            details_df["Poids"] = np.nan
        return {}, details_df

    if total_cap == 0:
        if not details_df.empty:
            details_df["Poids"] = np.nan
        return {}, details_df

    weights = {ticker: cap / total_cap for ticker, cap in caps.items()}
    if not details_df.empty:
        details_df["Poids"] = details_df["Ticker"].map(weights).fillna(0.0)
    return weights, details_df


def sensitivity_analysis(
    spot: float,
    risk_free_rate: float,
    dividend_yield: float,
    days_to_maturity: int,
    spot_range_pct: float = 5.0,
    rate_range_bps: float = 50.0,
    n_steps: int = 11,
) -> pd.DataFrame:
    """
    Analyse de sensibilite du prix du future aux variations de spot et de taux.
    """
    spot_shifts = np.linspace(-spot_range_pct, spot_range_pct, n_steps)
    rate_shifts = np.linspace(-rate_range_bps, rate_range_bps, n_steps)

    results = {}
    for rate_shift in rate_shifts:
        column_name = f"{rate_shift:+.0f} bps"
        column_values = []
        for spot_shift in spot_shifts:
            shifted_spot = spot * (1 + spot_shift / 100.0)
            shifted_rate = risk_free_rate + rate_shift / 10000.0
            try:
                result = price_future(shifted_spot, shifted_rate, dividend_yield, days_to_maturity)
                column_values.append(result["future_price"])
            except ValueError:
                column_values.append(np.nan)
        results[column_name] = column_values

    index_labels = [f"{spot_shift:+.1f}%" for spot_shift in spot_shifts]
    sensitivity_df = pd.DataFrame(results, index=index_labels)
    sensitivity_df.index.name = "Spot Shift"
    return sensitivity_df
