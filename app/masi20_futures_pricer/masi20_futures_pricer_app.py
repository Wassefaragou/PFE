# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import datetime
import json
import os
import re
import socket
import ssl
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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
MASI20_INDEX_CODE = "MSI20"
MASI20_INDEX_PAGE_URL = f"https://www.casablanca-bourse.com/fr/live-market/indices/{MASI20_INDEX_CODE}"
BVC_PROXY_BASE_URL = "https://www.casablanca-bourse.com/api/proxy"
MASI20_TICKER_ALIASES = {
    "adh": "DOUJA PROM ADDOHA",
    "adi": "ALLIANCES",
    "akt": "AKDITAL",
    "atw": "ATTIJARIWAFA BANK",
    "bcp": "BCP",
    "boa": "BANK OF AFRICA",
    "cfg": "CFG BANK",
    "cdm": "CDM",
    "cma": "CIMENTS DU MAROC",
    "csr": "COSUMAR",
    "cmg": "CMGP GROUP",
    "iam": "ITISSALAT AL-MAGHRIB",
    "lbv": "LABEL VIE",
    "lhm": "LAFARGEHOLCIM MAROC",
    "sid": "SONASID",
    "msa": "SODEP-Marsa Maroc",
    "jet": "JET CONTRACTORS",
    "rds": "RESIDENCES DAR SAADA",
    "tgc": "TGCC S.A",
    "tqm": "TAQA MOROCCO",
}


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


    def fetch_remote_payload(url: str, accept: str, timeout: float = 15.0) -> bytes:
        request = Request(
            url,
            headers={
                "Accept": accept,
                "Referer": MASI20_INDEX_PAGE_URL,
                "User-Agent": "MASI20-Futures-Pricer/1.0",
            },
        )
        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                with urlopen(request, timeout=timeout, context=ssl._create_unverified_context()) as response:
                    return response.read()
            except HTTPError as exc:
                raise RuntimeError(f"Erreur Casablanca Bourse ({exc.code}) sur {url}.") from exc
            except URLError as exc:
                reason = getattr(exc, "reason", None)
                is_timeout = isinstance(reason, (TimeoutError, socket.timeout))
                if is_timeout and attempt + 1 < max_attempts:
                    continue
                if is_timeout:
                    raise RuntimeError(
                        "La Bourse de Casablanca a mis trop de temps a repondre. Reessayez dans quelques secondes."
                    ) from exc
                raise RuntimeError(f"Connexion Casablanca Bourse impossible pour {url}.") from exc
            except (TimeoutError, socket.timeout) as exc:
                if attempt + 1 < max_attempts:
                    continue
                raise RuntimeError(
                    "La Bourse de Casablanca a mis trop de temps a repondre. Reessayez dans quelques secondes."
                ) from exc


    def fetch_remote_json(url: str, timeout: float = 15.0) -> dict:
        raw_payload = fetch_remote_payload(
            url,
            accept="application/json, application/vnd.api+json, text/plain, */*",
            timeout=timeout,
        )
        try:
            return json.loads(raw_payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Reponse JSON invalide pour {url}.") from exc


    def fetch_remote_text(url: str, timeout: float = 15.0) -> str:
        raw_payload = fetch_remote_payload(
            url,
            accept="text/html, application/json;q=0.9, */*;q=0.8",
            timeout=timeout,
        )
        return decode_text_file(raw_payload)


    def proxy_bvc_api_url(url: str) -> str:
        return (
            url.replace("https://api.casablanca-bourse.com", BVC_PROXY_BASE_URL)
            .replace("http://api.casablanca-bourse.com", BVC_PROXY_BASE_URL)
        )


    def extract_next_build_id(page_html: str) -> str:
        build_match = re.search(r'"buildId":"([^"]+)"', page_html)
        if build_match:
            return build_match.group(1)

        manifest_match = re.search(r"/_next/static/([^/]+)/_buildManifest\\.js", page_html)
        if manifest_match:
            return manifest_match.group(1)

        raise RuntimeError("Build Next.js introuvable sur la page MSI20.")


    def build_market_watch_rows(page_payload: dict) -> list[dict[str, float | str]]:
        included = {
            item.get("id"): item
            for item in page_payload.get("included", [])
            if item.get("type") == "instrument"
        }

        rows: list[dict[str, float | str]] = []
        for item in page_payload.get("data", []):
            attributes = item.get("attributes") or {}
            symbol_info = ((item.get("relationships") or {}).get("symbol") or {}).get("data") or {}
            instrument_id = symbol_info.get("id")
            instrument_attributes = (included.get(instrument_id) or {}).get("attributes") or {}
            ticker = clean_title(
                instrument_attributes.get("libelleFR")
                or instrument_attributes.get("libelleEN")
                or instrument_id
            )
            if not ticker:
                continue

            try:
                price_value = float(attributes["coursCourant"])
                cap_value = float(attributes["capitalisation"])
            except (KeyError, TypeError, ValueError):
                continue

            rows.append(
                {
                    "Ticker": ticker,
                    "Cours": price_value,
                    "Capitalisation": cap_value,
                }
            )

        return rows


    def fetch_masi20_market_snapshot(timeout: float = 20.0) -> tuple[list[dict[str, float | str]], str]:
        page_html = fetch_remote_text(MASI20_INDEX_PAGE_URL, timeout=timeout)
        build_id = extract_next_build_id(page_html)
        next_data_url = (
            f"https://www.casablanca-bourse.com/_next/data/{build_id}/fr/live-market/indices/{MASI20_INDEX_CODE}.json"
        )
        next_payload = fetch_remote_json(next_data_url, timeout=timeout)

        try:
            paragraphs = next_payload["pageProps"]["node"]["field_vactory_paragraphs"]
            composition_paragraph = next(
                paragraph
                for paragraph in paragraphs
                if (paragraph.get("field_vactory_component") or {}).get("widget_id")
                == "bourse_data_listing:index-composition"
            )
            widget_data = json.loads(composition_paragraph["field_vactory_component"]["widget_data"])
            first_page_payload = widget_data["components"][0]["collection"]["data"]
        except (KeyError, StopIteration, json.JSONDecodeError) as exc:
            raise RuntimeError("Structure inattendue pour la composition MSI20.") from exc

        rows = build_market_watch_rows(first_page_payload)
        next_page_url = (((first_page_payload.get("links") or {}).get("next") or {}).get("href"))
        visited_urls: set[str] = set()

        while next_page_url and next_page_url not in visited_urls:
            visited_urls.add(next_page_url)
            proxy_url = proxy_bvc_api_url(next_page_url)
            next_page_payload = fetch_remote_json(proxy_url, timeout=timeout)
            rows.extend(build_market_watch_rows(next_page_payload))
            next_page_url = (((next_page_payload.get("links") or {}).get("next") or {}).get("href"))

        deduplicated_rows: list[dict[str, float | str]] = []
        seen_titles: set[str] = set()
        for row in rows:
            ticker = str(row["Ticker"])
            if ticker in seen_titles:
                continue
            seen_titles.add(ticker)
            deduplicated_rows.append(row)

        if len(deduplicated_rows) != EXPECTED_MASI20_TITLES:
            raise RuntimeError(
                f"La composition fetchée contient {len(deduplicated_rows)} titres au lieu de {EXPECTED_MASI20_TITLES}."
            )

        fetched_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
        return deduplicated_rows, fetched_at


    def format_fetch_timestamp(value: object) -> str:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return "-"
        return parsed.strftime("%Y-%m-%d %H:%M:%S")


    def clean_title(value: object) -> str:
        if pd.isna(value):
            return ""
        text = str(value).strip()
        text = re.sub(r"\s+MC\s+Equity$", "", text, flags=re.IGNORECASE)
        if not text or text.lower() in {"nan", "na", "none", "-"}:
            return ""
        text = re.sub(r"\s+", " ", text)
        alias_key = normalize_key(text)
        return MASI20_TICKER_ALIASES.get(alias_key, text)


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


    def build_dividend_table(titles: list[str], existing_dividends: dict[str, float] | None = None) -> pd.DataFrame:
        dividend_map = existing_dividends or {}
        return pd.DataFrame(
            [
                {
                    "Ticker": ticker,
                    "Dividende (Di)": dividend_map.get(ticker, None),
                }
                for ticker in titles
            ]
        )


    def build_market_preview_table(
        titles: list[str],
        prices: Mapping[str, float],
        capitalisations: Mapping[str, float],
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Ticker": ticker,
                    "Cours": prices.get(ticker, float("nan")),
                    "Capitalisation": capitalisations.get(ticker, float("nan")),
                }
                for ticker in titles
            ]
        )


    def ensure_dividend_table_state(
        state_key: str,
        target_titles: list[str],
        existing_dividends: dict[str, float],
    ) -> None:
        expected_columns = ["Ticker", "Dividende (Di)"]
        if state_key not in st.session_state:
            st.session_state[state_key] = build_dividend_table(target_titles, existing_dividends)
            return

        current_dividend_df = st.session_state[state_key].copy()
        current_titles = current_dividend_df["Ticker"].tolist() if "Ticker" in current_dividend_df.columns else []
        if current_dividend_df.columns.tolist() != expected_columns:
            st.session_state[state_key] = build_dividend_table(target_titles, existing_dividends)
        elif target_titles and current_titles != target_titles:
            st.session_state[state_key] = build_dividend_table(target_titles, existing_dividends)


    def parse_dividend_file(source_df: pd.DataFrame) -> dict[str, float]:
        ticker_col = require_title_first_column(source_df, "dividendes MASI20")
        div_col = find_matching_column(source_df.columns.tolist(), [("divid",), ("div",), ("yield",)])
        if div_col is None:
            raise ValueError("Le fichier dividendes doit contenir Titre et Dividende.")

        dividends: dict[str, float] = {}
        for _, row in source_df.iterrows():
            ticker = clean_title(row[ticker_col])
            if not ticker:
                continue
            if pd.isna(row[div_col]):
                continue
            try:
                dividends[ticker] = parse_localized_float(row[div_col])
            except ValueError:
                continue

        if not dividends:
            raise ValueError("Aucun dividende valide n'a ete trouve dans le fichier.")
        return dividends


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


    def compact_future_contract_label(label: object) -> str:
        text = str(label).strip()
        prefix = "MASI20 FUTURE "
        if text.startswith(prefix):
            return text[len(prefix):]
        return text


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
    curve_display_df = pd.DataFrame()
    market_curve_table = pd.DataFrame()
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
                    la courbe affiche les taux BAM originaux, puis chaque future utilise
                    ensuite sa propre base cible pour determiner son taux sans risque.
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
        st.markdown("#### Donnees actions")
        st.markdown(
            """
            <div class="alert-info" style="font-size: 0.85rem; padding: 0.75rem;">
                Recuperez d'abord les <code>Cours</code> et <code>Capitalisations</code> du MASI20,
                puis importez la base facteurs avec <code>Titre</code> en premiere colonne :
                <code>Titre</code>, <code>Flottant</code>, <code>Plafonnement</code>.
            </div>
            """,
            unsafe_allow_html=True,
        )

        for state_key, default_value in (
            ("bvc_prices_imported", {}),
            ("bvc_dividends_imported", {}),
            ("bvc_caps_imported", {}),
            ("bvc_titles_imported", []),
            ("bvc_market_rows_imported", []),
            ("bvc_market_auto_fetched_at", None),
            ("masi_factor_df_imported", pd.DataFrame()),
        ):
            if state_key not in st.session_state:
                st.session_state[state_key] = default_value

        st.caption(f"API marche actions : {MASI20_INDEX_PAGE_URL}")
        if st.button("Fetch cours + caps MASI20", key="fetch_market_masi20", width="stretch"):
            try:
                fetched_market_rows, fetched_market_at = fetch_masi20_market_snapshot()
            except RuntimeError as exc:
                st.error(str(exc))
            else:
                fetched_titles = [str(row["Ticker"]) for row in fetched_market_rows]
                st.session_state["bvc_market_rows_imported"] = fetched_market_rows
                st.session_state["bvc_titles_imported"] = fetched_titles
                st.session_state["bvc_prices_imported"] = {
                    str(row["Ticker"]): float(row["Cours"]) for row in fetched_market_rows
                }
                st.session_state["bvc_caps_imported"] = {
                    str(row["Ticker"]): float(row["Capitalisation"]) for row in fetched_market_rows
                }
                st.session_state["bvc_market_auto_fetched_at"] = fetched_market_at

                existing_dividends = st.session_state.get("bvc_dividends_imported", {})
                st.session_state["div_data_free"] = build_dividend_table(fetched_titles, existing_dividends)
                st.rerun()

        fetched_market_rows = st.session_state.get("bvc_market_rows_imported", [])
        fetched_market_titles = st.session_state.get("bvc_titles_imported", [])
        fetched_market_at = st.session_state.get("bvc_market_auto_fetched_at")
        market_fetch_ready = len(fetched_market_titles) == EXPECTED_MASI20_TITLES

        st.text_input(
            "Dernier fetch marche",
            value=format_fetch_timestamp(fetched_market_at),
            disabled=True,
        )
        if market_fetch_ready:
            st.markdown(
                f"""
                <div class="alert-success" style="font-size: 0.85rem; padding: 0.5rem;">
                    Marche actions pret : <strong>{len(fetched_market_titles)} titres</strong> recuperes.
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(
                pd.DataFrame(fetched_market_rows),
                width="stretch",
                hide_index=True,
                height=260,
            )
        else:
            st.info("Cliquez sur le bouton de fetch pour charger les cours et capitalisations du MASI20.")

        factor_file = st.file_uploader(
            "Base facteurs MASI20 (CSV/XLSX)",
            type=["csv", "xlsx", "xls"],
            key="factor_file_upload",
        )

        if factor_file is None:
            st.session_state["masi_factor_df_imported"] = pd.DataFrame()
            if market_fetch_ready:
                st.info("Importez la base facteurs MASI20 pour lancer la validation.")
        elif market_fetch_ready:
            try:
                factor_source_df = read_uploaded_table(factor_file)
                factor_df = parse_masi_factors_file(factor_source_df)
                validated_titles = validate_masi20_title_lists(fetched_market_titles, factor_df["Titre"].tolist())
                title_order = {title_key(title): idx for idx, title in enumerate(validated_titles)}
                factor_df = (
                    factor_df.assign(_sort_key=factor_df["Titre"].map(lambda value: title_order[title_key(value)]))
                    .sort_values(by="_sort_key")
                    .drop(columns=["_sort_key"])
                    .reset_index(drop=True)
                )
                st.session_state["bvc_titles_imported"] = validated_titles
                st.session_state["masi_factor_df_imported"] = factor_df
                st.session_state["div_data_free"] = build_dividend_table(
                    validated_titles,
                    st.session_state.get("bvc_dividends_imported", {}),
                )
                st.markdown(
                    '<div class="alert-success" style="font-size: 0.85rem; padding: 0.5rem;">Validation OK : 20 titres identiques.</div>',
                    unsafe_allow_html=True,
                )
            except Exception as exc:
                st.session_state["masi_factor_df_imported"] = pd.DataFrame()
                st.error(f"Validation impossible : {exc}")
        else:
            st.info("Faites d'abord le fetch des cours et capitalisations, puis importez la base facteurs.")

    if yc_ready:
        try:
            curve_axis_label = "Horizon (ans)"
            r_final = interpolate_rate(
                effective_days_to_maturity,
                yc_used,
                target_maturity=effective_days_to_maturity,
                valuation_date=effective_eval_date,
            )
            curve_display_df = build_yield_curve_df(
                yc_used,
                valuation_date=effective_eval_date,
            )
        except Exception as exc:
            yc_ready = False
            r_final = None
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
        future_rate_points: list[dict[str, object]] = []
        for contract in selected_contracts:
            contract_rate = interpolate_rate(
                contract["days"],
                yc_used,
                target_maturity=contract["days"],
                valuation_date=effective_eval_date,
            )
            future_rate_points.append(
                {
                    "label": contract["label"],
                    "label_short": compact_future_contract_label(contract["label"]),
                    "days": contract["days"],
                    "years": contract["days"] / DISPLAY_YEAR_DAY_COUNT,
                    "rate_pct": contract_rate * 100.0,
                }
            )

        fig_yc = go.Figure()

        if rate_source == RATE_SOURCE_MARKET:
            fig_yc.add_trace(
                go.Scatter(
                    x=yc_years.tolist(),
                    y=yc_df["Taux source (%)"].tolist(),
                    mode="lines+markers+text",
                    name="Courbe BAM - taux originaux",
                    line=dict(color=COLORS["rate"], width=2.5),
                    marker=dict(color="#fcd34d", size=10, line=dict(width=2, color="#f59e0b")),
                    text=yc_df["Maturite courbe (jours)"].apply(format_display_years).tolist(),
                    textposition="top center",
                    textfont=dict(color="#fcd34d", size=10),
                    hovertemplate="Pilier: %{text}<br>Taux BAM: %{y:.3f}%<extra></extra>",
                )
            )
            yaxis_title = "Taux BAM originaux (%)"
        else:
            fine_days = list(range(1, min(10800, horizon_max) + 1, 5))
            fine_years = [day / DISPLAY_YEAR_DAY_COUNT for day in fine_days]
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
                    y=yc_df["Taux source (%)"].tolist(),
                    mode="markers+text",
                    name="Pilier",
                    marker=dict(color="#fcd34d", size=10, line=dict(width=2, color="#f59e0b")),
                    text=yc_df["Maturite courbe (jours)"].apply(format_display_years).tolist(),
                    textposition="top center",
                    textfont=dict(color="#fcd34d", size=10),
                    hovertemplate="Pilier: %{text}<br>Horizon: %{x:.3f} ans<br>Taux: %{y:.3f}%<extra></extra>",
                )
            )
            yaxis_title = "Taux (%)"

        if future_rate_points:
            fig_yc.add_trace(
                go.Scatter(
                    x=[point["years"] for point in future_rate_points],
                    y=[point["rate_pct"] for point in future_rate_points],
                    mode="markers+text",
                    name="Taux futures - base cible",
                    marker=dict(color="#00d4aa", size=14, symbol="star", line=dict(width=2, color="white")),
                    text=[str(point["label_short"]) for point in future_rate_points],
                    textposition="top center",
                    textfont=dict(color="#00d4aa", size=11, family="JetBrains Mono"),
                    hovertemplate="Future: %{text}<br>Horizon: %{x:.3f} ans<br>Taux retenu: %{y:.3f}%<extra></extra>",
                )
            )

        fig_yc.update_layout(
            **CHART_LAYOUT,
            height=400,
            title="Courbe des taux sans risque",
            xaxis_title=curve_axis_label,
            yaxis_title=yaxis_title,
        )
        st.plotly_chart(fig_yc, width="stretch")

        if rate_source == RATE_SOURCE_MARKET and rate_warnings:
            with st.expander("Lignes ignorees a l'import"):
                for warning in rate_warnings:
                    st.write(f"- {warning}")


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
    imported_titles = st.session_state.get("bvc_titles_imported", [])
    imported_prices = st.session_state.get("bvc_prices_imported", {})
    imported_caps = st.session_state.get("bvc_caps_imported", {})
    market_source_label = (
        f"Fetch auto ({len(imported_titles)} titres)"
        if imported_titles
        else "Cours/capitalisations non charges"
    )

    target_dividend_titles = imported_titles or tickers_available
    existing_dividend_map = st.session_state.get("bvc_dividends_imported", {})
    ensure_dividend_table_state(div_data_key, target_dividend_titles, existing_dividend_map)

    st.markdown(
        """
        <div class="alert-info">
            Les cours et capitalisations proviennent du fetch de la section 1.
            Renseignez ici uniquement les dividendes par ticker.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Source facteurs MASI20 : {weights_source_label}")
    st.caption(f"Source marche actions : {market_source_label}")

    if imported_titles:
        with st.expander("Apercu cours / capitalisations fetches", expanded=False):
            st.dataframe(
                build_market_preview_table(imported_titles, imported_prices, imported_caps),
                width="stretch",
                hide_index=True,
                height=260,
            )

    dividend_file = st.file_uploader(
        "Importer un fichier dividendes (CSV/XLSX)",
        type=["csv", "xlsx", "xls"],
        key="dividend_file_upload",
        help="Colonnes attendues : Titre, Dividende. Les tickers type ADH MC Equity sont acceptes.",
    )

    if st.button(
        "Charger le fichier dividendes dans la table",
        key="load_dividend_file",
        width="stretch",
        disabled=dividend_file is None,
    ):
        try:
            dividend_source_df = read_uploaded_table(dividend_file)
            imported_dividend_map = parse_dividend_file(dividend_source_df)
            current_div_df = st.session_state[div_data_key].copy()
            if current_div_df.empty:
                current_div_df = build_dividend_table(target_dividend_titles or list(imported_dividend_map.keys()))

            index_by_key = {
                title_key(row["Ticker"]): idx
                for idx, row in current_div_df.iterrows()
                if clean_title(row["Ticker"])
            }
            ignored_titles: list[str] = []
            imported_count = 0

            for ticker, dividend_value in imported_dividend_map.items():
                ticker_key = title_key(ticker)
                row_idx = index_by_key.get(ticker_key)
                if row_idx is not None:
                    current_div_df.at[row_idx, "Dividende (Di)"] = dividend_value
                    imported_count += 1
                elif not target_dividend_titles:
                    current_div_df = pd.concat(
                        [
                            current_div_df,
                            pd.DataFrame([{"Ticker": ticker, "Dividende (Di)": dividend_value}]),
                        ],
                        ignore_index=True,
                    )
                    index_by_key[ticker_key] = len(current_div_df) - 1
                    imported_count += 1
                else:
                    ignored_titles.append(ticker)

            st.session_state[div_data_key] = current_div_df.reset_index(drop=True)
            st.session_state["bvc_dividends_imported"] = {
                str(row["Ticker"]).strip(): float(row["Dividende (Di)"])
                for _, row in st.session_state[div_data_key].iterrows()
                if clean_title(row["Ticker"]) and pd.notna(row["Dividende (Di)"])
            }
            if ignored_titles:
                ignored_preview = ", ".join(sorted(set(ignored_titles))[:5])
                suffix = " ..." if len(set(ignored_titles)) > 5 else ""
                st.warning(f"Titres ignores (hors table courante) : {ignored_preview}{suffix}")
            st.success(f"{imported_count} dividendes importes dans la table.")
            st.rerun()
        except Exception as exc:
            st.error(f"Import dividendes impossible : {exc}")

    edited_div_df = st.data_editor(
        st.session_state[div_data_key],
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker"),
            "Dividende (Di)": st.column_config.NumberColumn("Dividende (MAD)", min_value=0.0, format="%.2f"),
        },
        width="stretch",
        hide_index=True,
        num_rows="fixed" if imported_titles else "dynamic",
        key="div_editor_free",
    )

    st.session_state[div_data_key] = edited_div_df.copy()

    stock_prices_dict = {
        ticker: float(value)
        for ticker, value in imported_prices.items()
        if value is not None and pd.notna(value)
    }
    dividends = {}
    market_caps_dict = {
        ticker: float(value)
        for ticker, value in imported_caps.items()
        if value is not None and pd.notna(value)
    }
    for _, row in edited_div_df.iterrows():
        ticker = str(row["Ticker"]).strip()
        if not ticker or ticker.lower() == "nan":
            continue
        if pd.notna(row["Dividende (Di)"]):
            dividends[ticker] = float(row["Dividende (Di)"])
    st.session_state["bvc_dividends_imported"] = dividends.copy()

    dividend_input_warnings = []
    if df_weights is not None:
        weights, _ = compute_index_weights_from_caps(
            df_weights,
            market_caps_dict,
            include_details=False,
        )
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
        div_yield, _ = compute_dividend_yield(
            stock_prices_dict,
            dividends,
            weights,
            include_details=False,
        )
    else:
        div_yield = 0.0

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
            result_maturity = price_future(
                spot_price,
                r_maturity,
                d_used,
                contract["days"],
                include_breakdown=False,
            )
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
