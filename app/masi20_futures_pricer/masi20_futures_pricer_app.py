# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import datetime
import json
import os
import re
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except Exception:  # pragma: no cover - fallback for older Streamlit/bare mode
    def get_script_run_ctx():
        return None

from masi20_futures_pricer_engine import (
    basis_label_for_days,
    build_flat_curve_from_manual_rate,
    build_market_curve_from_csv,
    build_yield_curve_df,
    compute_dividend_yield,
    compute_index_weights_from_caps,
    generate_maturity_schedule,
    interpolate_rate,
    price_future,
)

APP_TITLE = "MASI20 Futures Pricer"

NAVIGATION_STATE_KEY = 'selected_app'
DISPLAY_YEAR_DAY_COUNT = 360.0
SPOT_SOURCE_MANUAL = "Manuel"
SPOT_SOURCE_AUTO = "Automatique"
MASI20_SPOT_API_URL = "https://api.casablanca-bourse.com/fr/api/bourse/dashboard/index_cotation/512343"


def run():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="https://www.google.com/s2/favicons?domain=attijariwafabank.com&sz=128",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if st.button("\u2b05\ufe0f Retour a l'accueil", key='back_to_home_futures_pricer'):
        st.session_state[NAVIGATION_STATE_KEY] = None
        st.rerun()


    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as handle:
            st.markdown(f"<style>{handle.read()}</style>", unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] nav[aria-label="Page navigation"],
        section[data-testid="stSidebar"] div[data-testid="stSidebarNav"],
        section[data-testid="stSidebar"] ul[data-testid="stSidebarNavItems"],
        section[data-testid="stSidebar"] [data-testid="stSidebarNavSeparator"],
        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


    CHART_LAYOUT = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(family="Inter, sans-serif", color="#94a3b8", size=12),
        title_font=dict(size=16, color="#e2e8f0", family="Inter, sans-serif"),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", size=11),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.06)"),
        margin=dict(l=50, r=30, t=60, b=50),
    )

    COLORS = {
        "future": "#f59e0b",
        "spot": "#6c63ff",
        "basis": "#00d4aa",
        "rate": "#f87171",
        "bar1": "#fbbf24",
    }

    RATE_SOURCE_MARKET = "CSV marche"
    RATE_SOURCE_MANUAL = "Taux manuel"
    DEFAULT_MANUAL_RATE = 0.00
    EXPECTED_MASI20_TITLES = 20


    def contract_price_card(
        contract_label: str,
        future_price: float,
        maturity_date,
        maturity_years: float,
        rate_used_pct: float,
    ) -> str:
        maturity_text = maturity_date.strftime("%d/%m/%Y") if hasattr(maturity_date, "strftime") else str(maturity_date)
        return (
            '<div class="contract-price-card">'
            '<div class="contract-price-accent"></div>'
            '<div class="contract-price-header">'
            f'<div class="contract-price-code">{contract_label}</div>'
            '</div>'
            f'<div class="contract-price-value">{future_price:.2f}</div>'
            '<div class="contract-price-subtitle">Cours theorique</div>'
            '<div class="contract-price-meta">'
            '<div class="contract-price-meta-item">'
            '<span class="contract-price-meta-label">Date d\'echeance</span>'
            f'<span class="contract-price-meta-value">{maturity_text}</span>'
            '</div>'
            '<div class="contract-price-meta-item">'
            '<span class="contract-price-meta-label">Horizon</span>'
            f'<span class="contract-price-meta-value">{maturity_years:.3f} an</span>'
            '</div>'
            '<div class="contract-price-meta-item">'
            '<span class="contract-price-meta-label">Taux sans risque</span>'
            f'<span class="contract-price-meta-value">{rate_used_pct:.4f}%</span>'
            '</div>'
            '</div>'
            '</div>'
        )


    def decode_text_file(raw_bytes: bytes) -> str:
        for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                return raw_bytes.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw_bytes.decode("utf-8", errors="replace")


    def normalize_key(value: object) -> str:
        normalized = unicodedata.normalize("NFKD", str(value))
        ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
        return re.sub(r"[^a-z0-9]+", "_", ascii_text.lower()).strip("_")


    def parse_localized_float(value: object) -> float:
        if pd.isna(value):
            raise ValueError("valeur manquante")
        text = str(value).strip()
        if not text or text.lower() in {"nan", "na", "none", "-"}:
            raise ValueError("valeur manquante")
        text = text.replace("\xa0", " ").replace(" ", "").replace("%", "").replace(",", ".")
        return float(text)


    def parse_optional_localized_float(value: object) -> tuple[float, str | None]:
        text = str(value).strip() if value is not None else ""
        if not text:
            return float("nan"), None
        try:
            parsed_value = parse_localized_float(text)
        except ValueError:
            return float("nan"), "Le spot doit etre numerique."
        if parsed_value < 0:
            return float("nan"), "Le spot ne peut pas etre negatif."
        return parsed_value, None


    def fetch_masi20_spot(timeout: float = 10.0) -> tuple[float, str]:
        request = Request(
            MASI20_SPOT_API_URL,
            headers={
                "Accept": "application/json",
                "User-Agent": "MASI20-Futures-Pricer/1.0",
            },
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
        except HTTPError as exc:
            raise RuntimeError(f"Erreur API Casablanca Bourse ({exc.code}).") from exc
        except URLError as exc:
            raise RuntimeError("Connexion a l'API Casablanca Bourse impossible.") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("Reponse API invalide pour le spot MASI20.") from exc

        raw_spot = (payload.get("data") or {}).get("field_index_value")
        try:
            spot_value = float(raw_spot)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Le spot MASI20 est absent de la reponse API.") from exc

        if spot_value <= 0:
            raise RuntimeError("Le spot MASI20 retourne par l'API est invalide.")

        fetched_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
        return spot_value, fetched_at


    def format_fetch_timestamp(value: object) -> str:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return "-"
        return parsed.strftime("%Y-%m-%d %H:%M:%S")


    def clean_title(value: object) -> str:
        if pd.isna(value):
            return ""
        text = str(value).strip().replace(" MC Equity", "")
        if not text or text.lower() in {"nan", "na", "none", "-"}:
            return ""
        return re.sub(r"\s+", " ", text)


    def title_key(value: object) -> str:
        return normalize_key(clean_title(value))


    def read_uploaded_table(uploaded_file) -> pd.DataFrame:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        return pd.read_excel(uploaded_file)


    def find_matching_column(columns: list[str], candidates: list[tuple[str, ...]]) -> str | None:
        normalized = {column: normalize_key(column) for column in columns}
        for tokens in candidates:
            for column, normalized_name in normalized.items():
                if all(token in normalized_name for token in tokens):
                    return column
        return None


    def require_title_first_column(source_df: pd.DataFrame, source_label: str) -> str:
        if source_df.empty or not source_df.columns.tolist():
            raise ValueError(f"Le fichier {source_label} est vide.")
        first_column = source_df.columns.tolist()[0]
        if normalize_key(first_column) != "titre":
            raise ValueError(
                f"Le fichier {source_label} doit avoir 'Titre' comme premiere colonne."
            )
        return first_column


    def validate_masi20_title_lists(
        market_titles: list[str],
        factor_titles: list[str],
    ) -> list[str]:
        def _build_title_map(titles: list[str], source_label: str) -> tuple[list[str], dict[str, str]]:
            cleaned_titles = [clean_title(title) for title in titles if clean_title(title)]
            if len(cleaned_titles) != EXPECTED_MASI20_TITLES:
                raise ValueError(
                    f"Le fichier {source_label} doit contenir exactement {EXPECTED_MASI20_TITLES} titres. "
                    f"{len(cleaned_titles)} titres detectes."
                )

            title_map: dict[str, str] = {}
            duplicates: list[str] = []
            for title in cleaned_titles:
                key = title_key(title)
                if key in title_map:
                    duplicates.append(title)
                else:
                    title_map[key] = title

            if duplicates:
                duplicate_list = ", ".join(sorted(set(duplicates))[:5])
                suffix = " ..." if len(set(duplicates)) > 5 else ""
                raise ValueError(
                    f"Le fichier {source_label} contient des titres en double : {duplicate_list}{suffix}"
                )
            return cleaned_titles, title_map

        market_ordered_titles, market_map = _build_title_map(market_titles, "marche actions")
        _, factor_map = _build_title_map(factor_titles, "facteurs MASI20")

        market_keys = set(market_map.keys())
        factor_keys = set(factor_map.keys())
        if market_keys != factor_keys:
            missing_in_factors = [market_map[key] for key in market_keys - factor_keys]
            missing_in_market = [factor_map[key] for key in factor_keys - market_keys]
            details = []
            if missing_in_factors:
                details.append("absents des facteurs: " + ", ".join(sorted(missing_in_factors)[:5]))
            if missing_in_market:
                details.append("absents du marche actions: " + ", ".join(sorted(missing_in_market)[:5]))
            raise ValueError("Les deux bases ne contiennent pas exactement les memes titres (" + " | ".join(details) + ").")

        return market_ordered_titles


    def parse_bvc_market_file(source_df: pd.DataFrame) -> tuple[list[str], dict, dict, dict]:
        ticker_col = require_title_first_column(source_df, "marche actions")
        price_col = find_matching_column(source_df.columns.tolist(), [("cours",), ("prix",), ("price",)])
        div_col = find_matching_column(source_df.columns.tolist(), [("divid",), ("div",), ("yield",)])
        cap_col = find_matching_column(
            source_df.columns.tolist(),
            [("capitalisation",), ("capitalization",), ("market", "cap"), ("cap", "bours")],
        )

        if price_col is None and div_col is None and cap_col is None:
            raise ValueError("Le fichier BVC doit contenir au moins une colonne parmi cours, dividende ou capitalisation.")

        titles: list[str] = []
        prices: dict[str, float] = {}
        dividends: dict[str, float] = {}
        market_caps: dict[str, float] = {}

        for _, row in source_df.iterrows():
            ticker = clean_title(row[ticker_col])
            if not ticker or ticker.lower() == "nan":
                continue
            titles.append(ticker)
            if price_col is not None and pd.notna(row[price_col]):
                try:
                    prices[ticker] = parse_localized_float(row[price_col])
                except ValueError:
                    pass
            if div_col is not None and pd.notna(row[div_col]):
                try:
                    dividends[ticker] = parse_localized_float(row[div_col])
                except ValueError:
                    pass
            if cap_col is not None and pd.notna(row[cap_col]):
                try:
                    market_caps[ticker] = parse_localized_float(row[cap_col])
                except ValueError:
                    pass

        if not titles:
            raise ValueError("Aucun titre valide n'a ete trouve dans la base marche actions.")
        return titles, prices, dividends, market_caps


    def parse_masi_factors_file(source_df: pd.DataFrame) -> pd.DataFrame:
        ticker_col = require_title_first_column(source_df, "facteurs MASI20")
        flottant_col = find_matching_column(source_df.columns.tolist(), [("flottant",), ("free", "float")])
        plafonnement_col = find_matching_column(source_df.columns.tolist(), [("plafonnement",), ("plafond",), ("capping",)])

        if ticker_col is None or flottant_col is None or plafonnement_col is None:
            raise ValueError("Le fichier facteurs doit contenir Titre, Flottant et Plafonnement.")

        records = []
        for _, row in source_df.iterrows():
            ticker = clean_title(row[ticker_col])
            if not ticker or ticker.lower() == "nan":
                continue
            try:
                flottant = parse_localized_float(row[flottant_col])
                plafonnement = parse_localized_float(row[plafonnement_col])
            except ValueError:
                continue

            if flottant > 1.0:
                flottant /= 100.0
            if plafonnement > 1.0:
                plafonnement /= 100.0

            records.append(
                {
                    "Titre": ticker,
                    "Ticker_Short": ticker,
                    "Flottant": flottant,
                    "Plafonnement": plafonnement,
                }
            )

        if not records:
            raise ValueError("Aucune ligne facteur valide n'a ete trouvee.")
        return pd.DataFrame(records).drop_duplicates(subset=["Ticker_Short"], keep="last").reset_index(drop=True)


    def format_market_table(
        market_curve_table: pd.DataFrame,
        valuation_date: object | None = None,
    ) -> pd.DataFrame:
        if market_curve_table.empty:
            return market_curve_table

        formatted = market_curve_table.copy()
        for column in ("Value Date", "Maturity Date"):
            if column in formatted.columns:
                formatted[column] = pd.to_datetime(
                    formatted[column],
                    dayfirst=True,
                    errors="coerce",
                ).dt.strftime("%d/%m/%Y")
        formatted = formatted.rename(
            columns={
                "Value Date": "Date de valeur",
                "Maturity Date": "Date d'echeance",
                "Maturity Days": "Maturite courbe (jours)",
                "Market Rate (%)": "Taux marche (%)",
                "Source Basis": "Convention source",
            }
        )
        return formatted


    def build_curve_table(
        curve_display_df: pd.DataFrame,
        market_curve_table: pd.DataFrame,
    ) -> pd.DataFrame:
        if curve_display_df.empty:
            return curve_display_df

        combined = curve_display_df.copy()
        for column in ("Date de valeur", "Date d'echeance"):
            if column in combined.columns:
                combined[column] = pd.to_datetime(combined[column], errors="coerce").dt.strftime("%d/%m/%Y")

        combined = combined.rename(columns={"Taux source (%)": "Taux marche (%)"})

        if market_curve_table.empty:
            return combined

        source_table = format_market_table(market_curve_table)
        merge_keys = ["Date de valeur", "Date d'echeance", "Maturite courbe (jours)"]
        source_columns = merge_keys + ["Taux marche (%)", "Convention source"]
        if any(column not in source_table.columns for column in source_columns):
            return combined

        return (
            combined.drop(
                columns=["Taux marche (%)", "Convention source"],
                errors="ignore",
            )
            .merge(
                source_table[source_columns].drop_duplicates(subset=merge_keys),
                on=merge_keys,
                how="left",
            )
        )


    def days_to_display_years(days_value: object) -> float:
        numeric_value = pd.to_numeric(days_value, errors="coerce")
        if pd.isna(numeric_value):
            return float("nan")
        return round(float(numeric_value) / DISPLAY_YEAR_DAY_COUNT, 3)


    def format_display_years(days_value: object) -> str:
        years_value = days_to_display_years(days_value)
        if pd.isna(years_value):
            return "-"
        years_text = f"{years_value:.3f}".rstrip("0").rstrip(".")
        unit = "an" if years_text == "1" else "ans"
        return f"{years_text} {unit}"


    def build_curve_table_ui(curve_table: pd.DataFrame) -> pd.DataFrame:
        if curve_table.empty:
            return curve_table

        curve_table_ui = curve_table.copy()
        if "Maturite courbe (jours)" in curve_table_ui.columns:
            curve_table_ui["Maturite courbe (ans)"] = pd.to_numeric(
                curve_table_ui["Maturite courbe (jours)"],
                errors="coerce",
            ).apply(days_to_display_years)
        return curve_table_ui


    def format_future_contract_code(maturity_date) -> str:
        month_codes = {
            1: "JAN",
            2: "FEV",
            3: "MAR",
            4: "AVR",
            5: "MAI",
            6: "JUI",
            7: "JUL",
            8: "AOU",
            9: "SEP",
            10: "OCT",
            11: "NOV",
            12: "DEC",
        }
        return f"MASI20 FUTURE {month_codes[maturity_date.month]}{str(maturity_date.year)[-2:]}"


    def build_upcoming_contracts(reference_date, contract_count: int) -> list[dict]:
        raw_maturities = generate_maturity_schedule(
            datetime.combine(reference_date, datetime.min.time()),
            contract_count=contract_count,
        )
        contracts = []
        for maturity in raw_maturities[:contract_count]:
            maturity_date = maturity["date"].date() if hasattr(maturity["date"], "date") else maturity["date"]
            contracts.append(
                {
                    "label": format_future_contract_code(maturity_date),
                    "date": maturity_date,
                    "days": max(0, int((maturity_date - reference_date).days)),
                }
            )
        return contracts


    with st.sidebar:
        default_eval_date = datetime.now().date()
        default_contracts = build_upcoming_contracts(default_eval_date, 1)

        if "pricer_spot_mode" not in st.session_state:
            st.session_state["pricer_spot_mode"] = SPOT_SOURCE_MANUAL
        if "pricer_spot_manual_input" not in st.session_state:
            st.session_state["pricer_spot_manual_input"] = ""
        if "pricer_spot_auto_value" not in st.session_state:
            st.session_state["pricer_spot_auto_value"] = None
        if "pricer_spot_auto_fetched_at" not in st.session_state:
            st.session_state["pricer_spot_auto_fetched_at"] = None

        st.markdown(
            f"""
            <div class="sidebar-logo" style="display: flex; align-items: center; margin-bottom: 20px;">
                <img src="https://www.google.com/s2/favicons?domain=attijariwafabank.com&sz=128" style="width: 40px; margin-right: 12px; border-radius: 4px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
                <div class="logo-text" style="font-size: 1.1rem; font-weight: 600;">{APP_TITLE}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown("##### Parametres")

        spot_mode = st.radio(
            "Source du spot MASI20",
            options=[SPOT_SOURCE_MANUAL, SPOT_SOURCE_AUTO],
            key="pricer_spot_mode",
            horizontal=True,
        )

        if spot_mode == SPOT_SOURCE_MANUAL:
            spot_input = st.text_input(
                "Spot MASI20 (S)",
                key="pricer_spot_manual_input",
                placeholder="Ex: 1423.18",
                help="Saisissez manuellement le spot MASI20.",
            )
            spot_price, spot_error = parse_optional_localized_float(spot_input)
            if spot_error:
                st.error(spot_error)
            spot_source_caption = "Saisi manuellement dans la barre laterale"
        else:
            st.caption(f"API : {MASI20_SPOT_API_URL}")
            if st.button("Fetch spot MASI20", key="fetch_spot_api", width="stretch"):
                try:
                    auto_spot_value, auto_spot_fetched_at = fetch_masi20_spot()
                except RuntimeError as exc:
                    st.error(str(exc))
                else:
                    st.session_state["pricer_spot_auto_value"] = auto_spot_value
                    st.session_state["pricer_spot_auto_fetched_at"] = auto_spot_fetched_at
                    st.rerun()

            auto_spot_value = st.session_state.get("pricer_spot_auto_value")
            auto_spot_fetched_at = st.session_state.get("pricer_spot_auto_fetched_at")
            st.text_input(
                "Dernier spot auto",
                value="-" if auto_spot_value is None else f"{float(auto_spot_value):.4f}",
                disabled=True,
            )
            st.text_input(
                "Dernier fetch",
                value=format_fetch_timestamp(auto_spot_fetched_at),
                disabled=True,
            )
            spot_price = float(auto_spot_value) if auto_spot_value is not None else float("nan")
            spot_error = None
            spot_source_caption = (
                "Recupere automatiquement via l'API Casablanca Bourse"
                if auto_spot_value is not None
                else "Cliquez sur le bouton de fetch pour recuperer le spot"
            )

        st.markdown("---")

        eval_date = st.date_input(
            "Date de calcul",
            value=default_eval_date,
            help="Date de depart du calcul (t=0).",
        )
        multi_contract_count = int(st.number_input(
            "Nombre de contrats",
            min_value=1,
            value=3,
            step=1,
            format="%d",
            help="Selectionne autant de contrats trimestriels que necessaire apres la date de calcul.",
        ))
        selected_contracts = build_upcoming_contracts(eval_date, multi_contract_count)
        maturity_date = selected_contracts[0]["date"] if selected_contracts else eval_date

        days_to_maturity = selected_contracts[0]["days"] if selected_contracts else 0
        future_contract_code = selected_contracts[0]["label"] if selected_contracts else "-"
        st.caption(f"Premier contrat : {future_contract_code}")
        if selected_contracts:
            st.caption(
                "Contrats selectionnes : " + ", ".join(contract["label"] for contract in selected_contracts)
            )
        else:
            st.warning("Aucune echeance trimestrielle disponible apres la date de calcul.")

        st.markdown("---")
        st.markdown("##### Resume")
        st.markdown(
            f"""
            <div class="sidebar-config-grid">
                <div class="sidebar-config-item">
                    <div class="cfg-val">{days_to_maturity}j</div>
                    <div class="cfg-label">1re echeance</div>
                </div>
                <div class="sidebar-config-item">
                    <div class="cfg-val">{days_to_maturity / 360:.6f}</div>
                    <div class="cfg-label">Horizon (an)</div>
                </div>
                <div class="sidebar-config-item">
                    <div class="cfg-val" style="font-size: 0.95rem;">{future_contract_code}</div>
                    <div class="cfg-label">Premier contrat</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


    st.markdown(
        f"""
        <div class="hero-wrap">
            <h1 class="hero-title">{APP_TITLE}</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


    st.markdown(
        """
        <div class="section-card">
            <div class="section-label"><span class="num">1</span> IMPORT</div>
            <div class="section-heading">Donnees de marche</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_import_rate, col_import_bvc = st.columns(2)

    yc_ready = False
    yc_used = None
    r_final = None
    r_pricing_final = None
    curve_display_df = pd.DataFrame()
    market_curve_table = pd.DataFrame()
    rate_source_label = ""
    rate_warnings: list[str] = []
    curve_axis_label = "Horizon (ans)"
    effective_eval_date = eval_date
    effective_days_to_maturity = days_to_maturity

    with col_import_rate:
        st.markdown("#### Taux sans risque")
        rate_source = st.radio(
            "Source",
            [RATE_SOURCE_MARKET, RATE_SOURCE_MANUAL],
            help="Choisissez un CSV marche ou un taux manuel.",
        )

        if rate_source == RATE_SOURCE_MARKET:
            st.markdown(
                """
                <div class="alert-info" style="font-size: 0.82rem; padding: 0.6rem;">
                    Importez le CSV de taux. Les maturites sont detectees depuis les dates,
                    puis converties vers la convention cible du future.
                </div>
                """,
                unsafe_allow_html=True,
            )
            uploaded_market = st.file_uploader("CSV bons du Tresor", type=["csv"], key="market_csv")
            if uploaded_market is not None:
                try:
                    market_curve, market_curve_table = build_market_curve_from_csv(
                        decode_text_file(uploaded_market.read()),
                        source_name=uploaded_market.name,
                    )
                    st.session_state["rf_market_curve"] = market_curve
                    st.session_state["rf_market_table"] = market_curve_table
                except Exception as exc:
                    st.session_state["rf_market_curve"] = None
                    st.session_state["rf_market_table"] = pd.DataFrame()
                    st.error(f"Lecture du CSV impossible : {exc}")

            stored_curve = st.session_state.get("rf_market_curve")
            stored_table = st.session_state.get("rf_market_table", pd.DataFrame())
            if stored_curve is not None:
                yc_used = stored_curve
                yc_ready = True
                rate_source_label = "CSV marche"
                if isinstance(stored_table, pd.DataFrame):
                    market_curve_table = stored_table.copy()
                    rate_warnings = market_curve_table.attrs.get("warnings", [])

                st.markdown(
                    f"""
                    <div class="alert-success" style="font-size: 0.85rem; padding: 0.5rem;">
                        Courbe prete : <strong>{len(yc_used.pillars)} points</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info("Importez un CSV pour construire la courbe.")
        else:
            manual_rate_pct = st.number_input(
                "Taux manuel (%)",
                min_value=-10.0,
                max_value=25.0,
                value=DEFAULT_MANUAL_RATE,
                step=0.01,
                format="%.4f",
                help="Taux fixe applique a tous les calculs.",
            )
            try:
                yc_used = build_flat_curve_from_manual_rate(
                    manual_rate_pct,
                    max(1, int(effective_days_to_maturity)),
                    source_name="Taux manuel",
                )
                yc_ready = True
                rate_source_label = "Taux manuel"
                market_curve_table = pd.DataFrame(
                    [
                        {
                            "Type": "Taux manuel",
                            "Maturite d'ancrage (jours)": max(1, int(effective_days_to_maturity)),
                            "Convention cible": basis_label_for_days(effective_days_to_maturity),
                            "Taux saisi (%)": manual_rate_pct,
                        }
                    ]
                )
                st.markdown(
                    f"""
                    <div class="alert-info" style="font-size: 0.82rem; padding: 0.6rem;">
                        Convention cible : <strong>{basis_label_for_days(effective_days_to_maturity)}</strong>.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception as exc:
                st.error(f"Taux manuel invalide : {exc}")

    with col_import_bvc:
        stock_prices = {}
        st.markdown("#### Donnees actions")
        st.markdown(
            """
            <div class="alert-info" style="font-size: 0.85rem; padding: 0.75rem;">
                Importez deux bases avec <code>Titre</code> en premiere colonne.
                Base actions : <code>Titre</code>, <code>Cours</code>, <code>Dividende</code>, <code>Capitalisation</code>.
                Base facteurs : <code>Titre</code>, <code>Flottant</code>, <code>Plafonnement</code>.
                Les deux bases doivent contenir les 20 memes titres.
            </div>
            """,
            unsafe_allow_html=True,
        )
        price_file = st.file_uploader(
            "Base actions (CSV/XLSX)",
            type=["csv", "xlsx", "xls"],
            key="price_file_upload",
        )
        factor_file = st.file_uploader(
            "Base facteurs MASI20 (CSV/XLSX)",
            type=["csv", "xlsx", "xls"],
            key="factor_file_upload",
        )
        bvc_ready = False
        st.session_state["bvc_prices_imported"] = {}
        st.session_state["bvc_dividends_imported"] = {}
        st.session_state["bvc_caps_imported"] = {}
        st.session_state["bvc_titles_imported"] = []
        st.session_state["masi_factor_df_imported"] = pd.DataFrame()

        if price_file is not None and factor_file is not None:
            try:
                price_df = read_uploaded_table(price_file)
                factor_source_df = read_uploaded_table(factor_file)
                market_titles, parsed_prices, parsed_dividends, parsed_caps = parse_bvc_market_file(price_df)
                factor_df = parse_masi_factors_file(factor_source_df)
                validated_titles = validate_masi20_title_lists(market_titles, factor_df["Titre"].tolist())
                title_order = {title_key(title): idx for idx, title in enumerate(validated_titles)}
                factor_df = (
                    factor_df.assign(_sort_key=factor_df["Titre"].map(lambda value: title_order[title_key(value)]))
                    .sort_values(by="_sort_key")
                    .drop(columns=["_sort_key"])
                    .reset_index(drop=True)
                )
                bvc_ready = True
                st.session_state["bvc_prices_imported"] = parsed_prices
                st.session_state["bvc_dividends_imported"] = parsed_dividends
                st.session_state["bvc_caps_imported"] = parsed_caps
                st.session_state["bvc_titles_imported"] = validated_titles
                st.session_state["masi_factor_df_imported"] = factor_df
                st.markdown(
                    '<div class="alert-success" style="font-size: 0.85rem; padding: 0.5rem;">Validation OK : 20 titres identiques.</div>',
                    unsafe_allow_html=True,
                )
            except Exception as exc:
                st.error(f"Validation impossible : {exc}")
        elif price_file is not None or factor_file is not None:
            st.info("Importez les deux bases pour lancer la validation MASI20.")
        else:
            st.info("Le controle MASI20 s'activera apres import des deux bases.")

    if yc_ready:
        try:
            curve_axis_label = "Horizon (ans)"
            r_final = interpolate_rate(
                effective_days_to_maturity,
                yc_used,
                target_maturity=effective_days_to_maturity,
                valuation_date=effective_eval_date,
            )
            r_pricing_final = r_final
            curve_display_df = build_yield_curve_df(
                yc_used,
                target_maturity=effective_days_to_maturity,
                valuation_date=effective_eval_date,
            )
        except Exception as exc:
            yc_ready = False
            r_final = None
            r_pricing_final = None
            curve_display_df = pd.DataFrame()
            st.error(
                "Courbe de taux inutilisable a la date de valorisation "
                f"{effective_eval_date.strftime('%d/%m/%Y')} : {exc}"
            )


    st.markdown(
        """
        <div class="section-card" style="margin-top:2rem;">
            <div class="section-label"><span class="num">2</span> COURBE TAUX</div>
            <div class="section-heading">Courbe des taux</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs_yield = st.tabs(["Graphique", "Tables"])

    with tabs_yield[0]:
        if not yc_ready:
            st.empty()
        else:
            yc_df = curve_display_df.sort_values(by="Maturite courbe (jours)").reset_index(drop=True)
            yc_years = yc_df["Maturite courbe (jours)"].apply(days_to_display_years)
            horizon_max = max(
                360,
                int(effective_days_to_maturity) + 360,
                int(yc_df["Maturite courbe (jours)"].max()),
            )
            fine_days = list(range(1, min(10800, horizon_max) + 1, 5))
            fine_years = [day / DISPLAY_YEAR_DAY_COUNT for day in fine_days]
            future_years_to_maturity = effective_days_to_maturity / DISPLAY_YEAR_DAY_COUNT
            future_years_label = format_display_years(effective_days_to_maturity)
            fine_rates = [
                interpolate_rate(
                    day,
                    yc_used,
                    target_maturity=effective_days_to_maturity,
                    valuation_date=effective_eval_date,
                )
                * 100.0
                for day in fine_days
            ]

            fig_yc = go.Figure()
            fig_yc.add_trace(
                go.Scatter(
                    x=fine_years,
                    y=fine_rates,
                    mode="lines",
                    name="Courbe taux",
                    line=dict(color=COLORS["rate"], width=2.5),
                    fill="tozeroy",
                    fillcolor="rgba(248, 113, 113, 0.08)",
                    hovertemplate="Horizon: %{x:.3f} ans<br>Taux: %{y:.3f}%<extra></extra>",
                )
            )
            fig_yc.add_trace(
                go.Scatter(
                    x=yc_years.tolist(),
                    y=yc_df["Taux converti (%)"].tolist(),
                    mode="markers+text",
                    name="Piliers",
                    marker=dict(color="#fcd34d", size=10, line=dict(width=2, color="#f59e0b")),
                    text=yc_df["Maturite courbe (jours)"].apply(format_display_years).tolist(),
                    textposition="top center",
                    textfont=dict(color="#fcd34d", size=10),
                    hovertemplate="Pilier: %{text}<br>Horizon: %{x:.3f} ans<br>Taux: %{y:.3f}%<extra></extra>",
                )
            )
            fig_yc.add_trace(
                go.Scatter(
                    x=[future_years_to_maturity],
                    y=[r_final * 100.0],
                    mode="markers+text",
                    name=f"Taux future @ {future_years_label}",
                    marker=dict(color="#00d4aa", size=14, symbol="star", line=dict(width=2, color="white")),
                    text=[f"{r_final * 100.0:.3f}%"],
                    textposition="top center",
                    textfont=dict(color="#00d4aa", size=12, family="JetBrains Mono"),
                    hovertemplate="Future: %{x:.3f} ans<br>Taux: %{y:.3f}%<extra></extra>",
                )
            )
            fig_yc.update_layout(
                **CHART_LAYOUT,
                height=400,
                title="Courbe des taux sans risque",
                xaxis_title=curve_axis_label,
                yaxis_title="Taux converti (%)",
            )
            st.plotly_chart(fig_yc, width="stretch")
            st.markdown(
                f"""
                <div class="alert-success">
                    <strong>Taux retenu pour le future a {future_years_label} :</strong><br>
                    <span style="font-family: 'JetBrains Mono'; font-size: 1.3rem; color: #f8fafc;">
                        r = {r_final * 100.0:.4f}%
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with tabs_yield[1]:
        if yc_ready:
            st.markdown("##### Points de courbe")
            combined_curve_table = build_curve_table_ui(build_curve_table(curve_display_df, market_curve_table))
            display_columns = [
                "Date de valeur",
                "Date d'echeance",
                "Maturite courbe (ans)",
                "Taux marche (%)",
                "Convention source",
                "Convention cible",
                "Taux converti (%)",
            ]
            st.dataframe(
                combined_curve_table[display_columns],
                width="stretch",
                hide_index=True,
                height=320,
            )
            if rate_source == RATE_SOURCE_MARKET and rate_warnings:
                with st.expander("Lignes ignorees a l'import"):
                    for warning in rate_warnings:
                        st.write(f"- {warning}")
        else:
            st.empty()


    st.markdown(
        """
        <div class="section-card" style="margin-top:2rem;">
            <div class="section-label"><span class="num">3</span> DIVIDENDES</div>
            <div class="section-heading">Taux de dividende</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    imported_factor_df = st.session_state.get("masi_factor_df_imported", pd.DataFrame())
    if isinstance(imported_factor_df, pd.DataFrame) and not imported_factor_df.empty:
        df_weights = imported_factor_df.copy()
        tickers_available = df_weights["Ticker_Short"].tolist()
        weights_source_label = "Base facteurs importee"
    else:
        tickers_available = []
        df_weights = None
        weights_source_label = "Base facteurs absente"

    div_data_key = "div_data_free"
    if div_data_key not in st.session_state:
        st.session_state[div_data_key] = pd.DataFrame(
            columns=["Ticker", "Cours Actuel (Ci)", "Dividende (Di)", "Capitalisation Boursiere"]
        )

    st.markdown(
        """
        <div class="alert-info">
            Saisissez les cours et dividendes ci-dessous ou importez-les en section 1.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Source facteurs MASI20 : {weights_source_label}")

    if bvc_ready and ("bvc_prices_imported" in st.session_state or "bvc_dividends_imported" in st.session_state):
        current_div_df = st.session_state[div_data_key].copy()
        imported_titles = st.session_state.get("bvc_titles_imported", [])
        if imported_titles and not current_div_df.empty:
            current_div_df = current_div_df[current_div_df["Ticker"].isin(imported_titles)].copy()
        imported_prices = st.session_state.get("bvc_prices_imported", {})
        imported_dividends = st.session_state.get("bvc_dividends_imported", {})
        imported_caps = st.session_state.get("bvc_caps_imported", {})
        for ticker in imported_titles:
            if ticker not in current_div_df["Ticker"].values:
                current_div_df = pd.concat(
                    [
                        current_div_df,
                        pd.DataFrame(
                            [
                                {
                                    "Ticker": ticker,
                                    "Cours Actuel (Ci)": imported_prices.get(ticker, None),
                                    "Dividende (Di)": imported_dividends.get(ticker, None),
                                    "Capitalisation Boursiere": imported_caps.get(ticker, None),
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
            else:
                idx = current_div_df[current_div_df["Ticker"] == ticker].index[0]
                if ticker in imported_prices:
                    current_div_df.at[idx, "Cours Actuel (Ci)"] = imported_prices[ticker]
                if ticker in imported_dividends:
                    current_div_df.at[idx, "Dividende (Di)"] = imported_dividends[ticker]
                if ticker in imported_caps:
                    current_div_df.at[idx, "Capitalisation Boursiere"] = imported_caps[ticker]
        st.session_state[div_data_key] = current_div_df

    edited_div_df = st.data_editor(
        st.session_state[div_data_key],
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker"),
            "Cours Actuel (Ci)": st.column_config.NumberColumn("Cours", min_value=0.0, format="%.2f"),
            "Dividende (Di)": st.column_config.NumberColumn("Dividende (MAD)", min_value=0.0, format="%.2f"),
            "Capitalisation Boursiere": st.column_config.NumberColumn("Cap. boursiere", min_value=0.0, format="%.2f"),
        },
        width="stretch",
        hide_index=True,
        num_rows="dynamic",
        key="div_editor_free",
    )

    stock_prices_dict = {}
    dividends = {}
    market_caps_dict = {}
    for _, row in edited_div_df.iterrows():
        ticker = str(row["Ticker"]).strip()
        if not ticker or ticker.lower() == "nan":
            continue
        if pd.notna(row["Cours Actuel (Ci)"]):
            stock_prices_dict[ticker] = float(row["Cours Actuel (Ci)"])
        if pd.notna(row["Dividende (Di)"]):
            dividends[ticker] = float(row["Dividende (Di)"])
        if pd.notna(row["Capitalisation Boursiere"]):
            market_caps_dict[ticker] = float(row["Capitalisation Boursiere"])

    weights_details_df = pd.DataFrame()
    dividend_input_warnings = []
    if df_weights is not None:
        weights, weights_details_df = compute_index_weights_from_caps(df_weights, market_caps_dict)
    else:
        weights = {}
        dividend_input_warnings.append(
            "Le taux de dividende auto reste indisponible tant que les donnees MASI20 ne sont pas chargees."
        )

    required_tickers = tickers_available if tickers_available else sorted(weights.keys())
    missing_caps_tickers = []
    missing_price_tickers = []
    missing_dividend_tickers = []

    if df_weights is not None:
        missing_caps_tickers = [ticker for ticker in required_tickers if (market_caps_dict.get(ticker, 0) or 0) <= 0]
        if missing_caps_tickers:
            dividend_input_warnings.append(
                "Capitalisations manquantes pour : " + ", ".join(missing_caps_tickers[:10]) +
                (" ..." if len(missing_caps_tickers) > 10 else "")
            )

    if weights:
        missing_price_tickers = [ticker for ticker in required_tickers if (stock_prices_dict.get(ticker, 0) or 0) <= 0]
        missing_dividend_tickers = [ticker for ticker in required_tickers if ticker not in dividends]
        if missing_price_tickers:
            dividend_input_warnings.append(
                "Cours manquants pour : " + ", ".join(missing_price_tickers[:10]) +
                (" ..." if len(missing_price_tickers) > 10 else "")
            )
        if missing_dividend_tickers:
            dividend_input_warnings.append(
                "Dividendes manquants pour : " + ", ".join(missing_dividend_tickers[:10]) +
                (" ..." if len(missing_dividend_tickers) > 10 else "")
            )

    has_prices = any(price is not None and price > 0 for price in stock_prices_dict.values())
    auto_dividend_ready = bool(weights) and not missing_caps_tickers and not missing_price_tickers and not missing_dividend_tickers

    if auto_dividend_ready:
        div_yield, div_details_df = compute_dividend_yield(stock_prices_dict, dividends, weights)
    else:
        div_yield = 0.0
        div_details_df = pd.DataFrame()

    if not div_details_df.empty and not weights_details_df.empty:
        div_details_df = div_details_df.merge(
            weights_details_df[
                [
                    column
                    for column in weights_details_df.columns
                    if column in {
                        "Ticker",
                        "Capitalisation brute",
                        "Valeur de secours",
                        "Source valeur",
                        "Flottant",
                        "Plafonnement",
                        "Capitalisation ajustee",
                        "Poids",
                    }
                ]
            ],
            on="Ticker",
            how="left",
        )

    for warning_message in dividend_input_warnings:
        st.warning(warning_message)


    st.markdown(
        """
        <div class="section-card" style="margin-top:2rem;">
            <div class="section-label"><span class="num">4</span> COURS THEORIQUES</div>
            <div class="section-heading">Valorisation des contrats</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not yc_ready or r_final is None:
        st.markdown(
            """
            <div style="text-align: center; padding: 3rem 1rem; border: 1px dashed rgba(255,255,255,0.1); border-radius: 12px;">
                <div style="font-size: 2rem; margin-bottom: 1rem;">Attente</div>
                <h4 style="color: #94a3b8; margin: 0;">Taux sans risque manquant</h4>
                <p style="color: #64748b; margin-top: 0.5rem; font-size: 0.9rem;">Importez un CSV ou saisissez un taux manuel.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if get_script_run_ctx() is not None:
            st.stop()
        raise SystemExit(0)

    if not has_prices:
        st.info("Le taux de dividende auto sera calcule une fois les titres renseignes.")

    spot_ready = pd.notna(spot_price)
    r_used = r_final
    multi_records: list[dict] = []

    st.markdown("##### Parametres")
    col_p1, col_p2, col_p3 = st.columns(3)

    with col_p1:
        st.metric("Spot", "-" if not spot_ready else f"{spot_price:.2f}")
        st.caption(spot_source_caption)

    with col_p2:
        d_override = st.number_input(
            "Taux dividende (%)",
            min_value=0.0,
            max_value=20.0,
            value=float(div_yield * 100.0),
            step=0.01,
            format="%.4f",
            help="Saisissez un taux manuel si besoin.",
        )
        d_used = d_override / 100.0

    with col_p3:
        st.metric("Nombre de contrats", str(len(selected_contracts)))
        st.caption(
            f"Calcul au {effective_eval_date.strftime('%d/%m/%Y')} | 1re echeance : {maturity_date.strftime('%d/%m/%Y') if hasattr(maturity_date, 'strftime') else maturity_date}"
        )

    if not spot_ready:
        missing_spot_message = (
            "Renseignez le spot dans la barre laterale pour lancer la valorisation."
            if spot_mode == SPOT_SOURCE_MANUAL
            else "Cliquez sur le bouton de fetch dans la barre laterale pour recuperer le spot."
        )
        st.markdown(
            f"""
            <div style="text-align: center; padding: 2rem 1rem; border: 1px dashed rgba(255,255,255,0.1); border-radius: 12px;">
                <h4 style="color: #94a3b8; margin: 0;">Spot MASI20 manquant</h4>
                <p style="color: #64748b; margin-top: 0.6rem; font-size: 0.92rem;">
                    {missing_spot_message}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif not selected_contracts:
        st.info("Aucun contrat trimestriel disponible pour la date de calcul choisie.")
    else:
        for contract in selected_contracts:
            r_maturity = interpolate_rate(
                contract["days"],
                yc_used,
                target_maturity=contract["days"],
                valuation_date=effective_eval_date,
            )
            result_maturity = price_future(spot_price, r_maturity, d_used, contract["days"])
            multi_records.append(
                {
                    "Contrat": contract["label"],
                    "Date d'echeance": contract["date"].strftime("%d/%m/%Y"),
                    "Maturite (ans)": days_to_display_years(contract["days"]),
                    "Taux utilise (%)": round(result_maturity["risk_free_rate_pricing"] * 100.0, 4),
                    "Prix theorique": round(result_maturity["future_price"], 4),
                }
            )

        st.caption(
            f"{len(multi_records)} contrats trimestriels selectionnes a partir du {effective_eval_date.strftime('%d/%m/%Y')}."
        )
        for start_idx in range(0, len(multi_records), 3):
            contract_chunk = multi_records[start_idx:start_idx + 3]
            chunk_columns = st.columns(len(contract_chunk))
            for column, row in zip(chunk_columns, contract_chunk):
                with column:
                    st.markdown(
                        contract_price_card(
                            row["Contrat"],
                            float(row["Prix theorique"]),
                            pd.to_datetime(row["Date d'echeance"], dayfirst=True, errors="coerce"),
                            float(row["Maturite (ans)"]),
                            float(row["Taux utilise (%)"]),
                        ),
                        unsafe_allow_html=True,
                    )


    st.markdown(
        f"""
        <div class="app-footer">
            <span style="color:#64748b">{APP_TITLE}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
