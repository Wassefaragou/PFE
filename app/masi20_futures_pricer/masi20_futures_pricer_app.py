# -*- coding: utf-8 -*-
"""
Streamlit App - Pricer Futures MASI 20

Le taux sans risque n'expose que deux modes:
1. Courbe construite a partir d'un CSV de marche secondaire
2. Taux manuel saisi par l'utilisateur
"""

from __future__ import annotations

from datetime import datetime
import io
import os
import re
import unicodedata

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    generate_term_structure,
    interpolate_rate,
    price_future,
    sensitivity_analysis,
)

APP_TITLE = "MASI20 Futures Pricer"

NAVIGATION_STATE_KEY = 'selected_app'


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


    def metric_card(label: str, value: str, glow: str = "gold") -> str:
        return f"""<div class="glass-metric">
            <div class="glow glow-{glow}"></div>
            <h4>{label}</h4>
            <div class="val">{value}</div>
        </div>"""


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
        if "Value Date" in formatted.columns:
            formatted["Value Date"] = formatted["Value Date"].dt.strftime("%d/%m/%Y")
        formatted["Maturity Date"] = formatted["Maturity Date"].dt.strftime("%d/%m/%Y")
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
        return combined


    def render_curve_empty_state(message: str) -> None:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 3rem 1rem; border: 1px dashed rgba(255,255,255,0.1); border-radius: 12px;">
                <div style="font-size: 2rem; margin-bottom: 1rem;">Courbe</div>
                <h4 style="color: #94a3b8; margin: 0;">Courbe indisponible</h4>
                <p style="color: #64748b; margin-top: 0.5rem; font-size: 0.9rem;">{message}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


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
        return f"FMASI20{month_codes[maturity_date.month]}{str(maturity_date.year)[-2:]}"


    with st.sidebar:
        default_eval_date = datetime.now().date()
        default_maturity_date = max(default_eval_date, datetime(2026, 6, 19).date())

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

        spot_price = st.number_input(
            "Spot MASI20 (S)",
            min_value=0.0,
            value=1423.18,
            step=1.0,
            format="%.2f",
        )

        st.markdown("---")

        eval_date = st.date_input(
            "Date de calcul",
            value=default_eval_date,
            help="Date de depart du calcul (t=0).",
        )

        maturity_date = st.date_input(
            "Date d'echeance",
            value=default_maturity_date,
            min_value=eval_date,
            help="0 jour est autorise pour tester le cas limite.",
        )
        days_to_maturity = max(0, (maturity_date - eval_date).days)
        future_contract_code = format_future_contract_code(maturity_date)
        st.caption(f"Code contrat : {future_contract_code}")

        st.markdown("---")
        st.markdown("##### Resume")
        st.markdown(
            f"""
            <div class="sidebar-config-grid">
                <div class="sidebar-config-item">
                    <div class="cfg-val">{days_to_maturity}j</div>
                    <div class="cfg-label">Maturite future</div>
                </div>
                <div class="sidebar-config-item">
                    <div class="cfg-val">{days_to_maturity / 360:.6f}</div>
                    <div class="cfg-label">t = j/360</div>
                </div>
                <div class="sidebar-config-item">
                    <div class="cfg-val" style="font-size: 0.95rem;">{future_contract_code}</div>
                    <div class="cfg-label">Code</div>
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
    curve_axis_label = "Horizon (jours)"
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
            curve_axis_label = "Horizon (jours)"
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
            <div class="section-heading">Vue et controle</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs_yield = st.tabs(["Graphique", "Tables"])

    with tabs_yield[0]:
        if not yc_ready:
            render_curve_empty_state("Choisissez une source de taux en section 1.")
        else:
            yc_df = curve_display_df.sort_values(by="Maturite courbe (jours)").reset_index(drop=True)
            horizon_max = max(
                360,
                int(effective_days_to_maturity) + 360,
                int(yc_df["Maturite courbe (jours)"].max()),
            )
            fine_days = list(range(1, min(10800, horizon_max) + 1, 5))
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
                    x=fine_days,
                    y=fine_rates,
                    mode="lines",
                    name="Courbe taux",
                    line=dict(color=COLORS["rate"], width=2.5),
                    fill="tozeroy",
                    fillcolor="rgba(248, 113, 113, 0.08)",
                )
            )
            fig_yc.add_trace(
                go.Scatter(
                    x=yc_df["Maturite courbe (jours)"].tolist(),
                    y=yc_df["Taux converti (%)"].tolist(),
                    mode="markers+text",
                    name="Piliers",
                    marker=dict(color="#fcd34d", size=10, line=dict(width=2, color="#f59e0b")),
                    text=yc_df["Label"].tolist(),
                    textposition="top center",
                    textfont=dict(color="#fcd34d", size=10),
                )
            )
            fig_yc.add_trace(
                go.Scatter(
                    x=[effective_days_to_maturity],
                    y=[r_final * 100.0],
                    mode="markers+text",
                    name=f"Taux future @ {effective_days_to_maturity}j",
                    marker=dict(color="#00d4aa", size=14, symbol="star", line=dict(width=2, color="white")),
                    text=[f"{r_final * 100.0:.3f}%"],
                    textposition="top center",
                    textfont=dict(color="#00d4aa", size=12, family="JetBrains Mono"),
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
                    <strong>Taux retenu pour le future a {effective_days_to_maturity} jours :</strong><br>
                    <span style="font-family: 'JetBrains Mono'; font-size: 1.3rem; color: #f8fafc;">
                        r = {r_final * 100.0:.4f}%
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with tabs_yield[1]:
        if yc_ready:
            st.markdown("##### Courbe des taux")
            combined_curve_table = build_curve_table(curve_display_df, market_curve_table)
            display_columns = [
                "Date de valeur",
                "Date d'echeance",
                "Maturite courbe (jours)",
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
                with st.expander("Lignes ignorees"):
                    for warning in rate_warnings:
                        st.write(f"- {warning}")
        else:
            st.info("Aucune courbe a afficher.")


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
    has_caps = any(cap is not None and cap > 0 for cap in market_caps_dict.values())
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
            <div class="section-label"><span class="num">4</span> PRICING</div>
            <div class="section-heading">Prix theorique</div>
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

    st.markdown("##### Parametres")
    col_p1, col_p2, col_p3 = st.columns(3)

    with col_p1:
        st.metric("Taux utilise", f"{r_pricing_final * 100.0:.4f}%")
        st.caption(f"Taux de courbe applique tel quel : {r_final * 100.0:.4f}%")
        r_used = r_final

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
        st.metric("Maturite future", f"{effective_days_to_maturity}j")
        st.metric("t (annee)", f"{effective_days_to_maturity / 360:.6f}")
        st.caption(f"Calcul au {effective_eval_date.strftime('%d/%m/%Y')}")

    result = price_future(spot_price, r_used, d_used, effective_days_to_maturity)

    st.markdown(
        f"""
        <div class="pricing-result">
            <div class="future-label">Prix theorique du future MASI20</div>
            <div class="future-price">{result['future_price']:.2f}</div>
            <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 0.5rem;">
                Echeance : <strong style="color: #f8fafc;">{maturity_date.strftime('%d/%m/%Y') if hasattr(maturity_date, 'strftime') else maturity_date}</strong>
                &nbsp;&nbsp;|&nbsp;&nbsp;
                Spot : <strong style="color: #f8fafc;">{spot_price:.2f}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(metric_card("Spot", f"{result['spot']:.2f}", "purple"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("Future", f"{result['future_price']:.2f}", "gold"), unsafe_allow_html=True)
    with c3:
        basis_sign = "+" if result["basis"] >= 0 else ""
        st.markdown(metric_card("Base", f"{basis_sign}{result['basis']:.2f}", "green"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("Carry", f"{result['cost_of_carry'] * 100.0:.3f}%", "blue"), unsafe_allow_html=True)
    with c5:
        st.markdown(metric_card("Base %", f"{result['basis_pct']:.4f}%", "pink"), unsafe_allow_html=True)


    st.markdown(
        """
        <div class="section-card" style="margin-top:2rem;">
            <div class="section-label"><span class="num">5</span> ANALYSES</div>
            <div class="section-heading">Scenarios et details</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs_analysis = st.tabs(
        [
            "Terme",
            "Sensibilite",
            "Dividendes",
            "Calcul",
            "Export",
        ]
    )

    with tabs_analysis[0]:
        col_ts1, col_ts2 = st.columns([3, 1])
        with col_ts2:
            ts_max_days = st.slider(
                "Horizon (jours)",
                30,
                1800,
                min(720, max(effective_days_to_maturity + 90, 360)),
            )
            ts_step = st.slider("Pas (jours)", 1, 30, 1)

        with col_ts1:
            ts_df = generate_term_structure(
                spot_price,
                yc_used,
                d_used,
                ts_max_days,
                ts_step,
                valuation_date=effective_eval_date,
            )
            fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
            fig_ts.add_trace(
                go.Scatter(
                    x=ts_df["Jours"],
                    y=ts_df["Future"],
                    name="Future",
                    line=dict(color=COLORS["future"], width=2.5),
                ),
                secondary_y=False,
            )
            fig_ts.add_hline(
                y=spot_price,
                line_dash="dash",
                line_color=COLORS["spot"],
                annotation_text=f"Spot = {spot_price:.2f}",
                annotation_font=dict(color=COLORS["spot"], size=11),
                secondary_y=False,
            )
            fig_ts.add_trace(
                go.Scatter(
                    x=ts_df["Jours"],
                    y=ts_df["Basis (%)"],
                    name="Base (%)",
                    line=dict(color=COLORS["basis"], width=1.5, dash="dot"),
                ),
                secondary_y=True,
            )
            if effective_days_to_maturity <= ts_max_days:
                current_row = ts_df[ts_df["Jours"] == effective_days_to_maturity]
                if not current_row.empty:
                    fig_ts.add_trace(
                        go.Scatter(
                            x=[effective_days_to_maturity],
                            y=[current_row["Future"].iloc[0]],
                            mode="markers",
                            name=f"Echeance ({effective_days_to_maturity}j)",
                            marker=dict(color="#fcd34d", size=14, symbol="star", line=dict(width=2, color="white")),
                        ),
                        secondary_y=False,
                    )
            fig_ts.update_layout(**CHART_LAYOUT, height=500, title="Structure a terme - MASI20")
            fig_ts.update_xaxes(title_text="Maturite (jours)")
            fig_ts.update_yaxes(title_text="Prix future", secondary_y=False)
            fig_ts.update_yaxes(title_text="Base (%)", secondary_y=True)
            st.plotly_chart(fig_ts, width="stretch")
            st.dataframe(ts_df, width="stretch", hide_index=True, height=300)

    with tabs_analysis[1]:
        st.markdown(
            """
            <div class="alert-info">
                Variation du prix future selon le spot (lignes) et le taux (colonnes).
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            spot_range = st.slider("Spot (+/- %)", 1.0, 20.0, 5.0, 0.5)
        with col_s2:
            rate_range = st.slider("Taux (+/- bps)", 10, 200, 50, 10)
        with col_s3:
            n_steps = st.slider("Nb. de pas", 5, 21, 11, 2)

        sens_df = sensitivity_analysis(
            spot_price,
            r_used,
            d_used,
            effective_days_to_maturity,
            spot_range_pct=spot_range,
            rate_range_bps=rate_range,
            n_steps=n_steps,
        )
        fig_heat = go.Figure(
            data=go.Heatmap(
                z=sens_df.values,
                x=sens_df.columns.tolist(),
                y=sens_df.index.tolist(),
                colorscale=[
                    [0, "#0f172a"],
                    [0.25, "#1e1b4b"],
                    [0.5, "#f59e0b"],
                    [0.75, "#fbbf24"],
                    [1, "#fef3c7"],
                ],
                text=[[f"{value:.2f}" if pd.notna(value) else "NA" for value in row] for row in sens_df.values],
                texttemplate="%{text}",
                textfont=dict(size=9, color="white"),
                hoverongaps=False,
                colorbar=dict(title=dict(text="Prix Future", font=dict(color="#94a3b8")), tickfont=dict(color="#94a3b8")),
            )
        )
        fig_heat.update_layout(
            **CHART_LAYOUT,
            height=500,
            title="Sensibilite du prix future",
            xaxis_title="Variation Taux (bps)",
            yaxis_title="Variation Spot (%)",
        )
        st.plotly_chart(fig_heat, width="stretch")
        st.dataframe(sens_df, width="stretch", height=350)

    with tabs_analysis[2]:
        st.markdown("#### Detail du dividende")
        if auto_dividend_ready and not div_details_df.empty:
            col_d1, col_d2 = st.columns([2, 1])
            with col_d1:
                fig_div = go.Figure()
                fig_div.add_trace(
                    go.Bar(
                        x=div_details_df["Ticker"],
                        y=div_details_df["Contribution (%)"],
                        marker_color=COLORS["bar1"],
                        marker_line=dict(width=0),
                        name="Contribution",
                    )
                )
                fig_div.update_layout(
                    **CHART_LAYOUT,
                    height=400,
                    title="Contribution par titre",
                    yaxis_title="Contribution (%)",
                )
                st.plotly_chart(fig_div, width="stretch")
            with col_d2:
                st.markdown(
                    f"""
                    <div class="glass-metric">
                        <div class="glow glow-green"></div>
                        <h4>Taux total</h4>
                        <div class="val">{div_yield * 100.0:.4f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    <div class="glass-metric" style="margin-top: 1rem;">
                        <div class="glow glow-gold"></div>
                        <h4>Titres avec dividende</h4>
                        <div class="val">{len(div_details_df[div_details_df['Dividende (Di)'] > 0])}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.dataframe(div_details_df, width="stretch", hide_index=True)
            fig_pie = go.Figure(
                go.Pie(
                    labels=div_details_df["Ticker"],
                    values=div_details_df["Poids (pi)"],
                    hole=0.45,
                    marker=dict(colors=px.colors.qualitative.Pastel),
                    textinfo="label+percent",
                    textfont=dict(size=10),
                )
            )
            fig_pie.update_layout(**CHART_LAYOUT, height=400, title="Repartition des poids")
            st.plotly_chart(fig_pie, width="stretch")
        else:
            st.info(
                "Le taux auto n'est disponible que si les 20 titres ont un cours, un dividende et une capitalisation."
            )

        st.markdown("#### Poids MASI20")
        if not weights_details_df.empty:
            weight_columns = [
                column
                for column in [
                    "Ticker",
                    "Capitalisation brute",
                    "Flottant",
                    "Plafonnement",
                    "Capitalisation ajustee",
                    "Capitalisation disponible",
                    "Poids",
                ]
                if column in weights_details_df.columns
            ]
            st.dataframe(weights_details_df[weight_columns], width="stretch", hide_index=True, height=320)
        else:
            st.info("Aucune table de poids disponible.")

    with tabs_analysis[3]:
        st.markdown("#### Detail du calcul")
        detail_data = {
            "Parametre": [
                "Source taux",
                "Convention",
                "Source poids",
                "Spot (S)",
                "Taux courbe",
                "Taux utilise",
                "Taux dividende (d)",
                "Carry (r - d)",
                "Maturite future (j)",
                "t = jours/360",
                "Facteur expo",
                "(r - d) x t",
                "e^((r - d) x t)",
                "Prix theorique",
                "Base (F - S)",
                "Base (%)",
                "Spread theorique",
            ],
            "Valeur": [
                rate_source_label,
                result["rate_basis"],
                weights_source_label,
                f"{result['spot']:.4f}",
                f"{result['risk_free_rate_curve'] * 100.0:.4f}%",
                f"{result['risk_free_rate_pricing'] * 100.0:.4f}%",
                f"{result['dividend_yield'] * 100.0:.4f}%",
                f"{result['cost_of_carry'] * 100.0:.4f}%",
                f"{result['days_to_maturity']}",
                f"{result['t_fraction']:.6f}",
                f"{result['carry_factor']:.8f}",
                f"{result['cost_of_carry'] * result['t_fraction']:.6f}",
                f"{result['carry_factor']:.8f}",
                f"{result['future_price']:.4f}",
                f"{result['basis']:.4f}",
                f"{result['basis_pct']:.4f}%",
                f"{result['fair_value_spread']:.4f}",
            ],
        }
        st.dataframe(pd.DataFrame(detail_data), width="stretch", hide_index=True, height=520)
        st.markdown(
            f"""
            <div class="formula-box" style="margin-top: 1rem;">
                <div class="formula-label">Formule</div>
                <div class="formula-main" style="font-size: 1.1rem;">
                    F = {spot_price:.2f} x e<sup>({result['risk_free_rate_pricing'] * 100.0:.4f}% - {d_used * 100.0:.4f}%) x ({effective_days_to_maturity}/360)</sup>
                </div>
                <div style="font-family: 'JetBrains Mono'; color: #fcd34d; font-size: 1.2rem; margin-top: 0.8rem;">
                    F = {spot_price:.2f} x e<sup>{result['cost_of_carry'] * result['t_fraction']:.6f}</sup>
                    = {spot_price:.2f} x {result['carry_factor']:.8f}
                    = <strong>{result['future_price']:.4f}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tabs_analysis[4]:
        st.markdown("#### Export")
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            export_curve_table = build_curve_table(curve_display_df, market_curve_table)
            pd.DataFrame(detail_data).to_excel(writer, sheet_name="Resume Pricing", index=False)
            ts_df.to_excel(writer, sheet_name="Structure par Terme", index=False)
            sens_df.to_excel(writer, sheet_name="Sensibilite", index=True)
            if not div_details_df.empty:
                div_details_df.to_excel(writer, sheet_name="Dividend Yield", index=False)
            export_curve_table.to_excel(writer, sheet_name="Courbe des Taux", index=False)
            if not market_curve_table.empty and rate_source != RATE_SOURCE_MARKET:
                market_curve_table.to_excel(writer, sheet_name="Taux Source", index=False)
            if not weights_details_df.empty:
                weights_details_df.to_excel(writer, sheet_name="Poids Indice", index=False)
            elif df_weights is not None:
                df_weights.to_excel(writer, sheet_name="Poids Indice", index=False)
        output.seek(0)
        st.download_button(
            "Telecharger le rapport Excel",
            data=output,
            file_name=f"pricing_future_masi20_{effective_eval_date.strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )


    st.markdown(
        """
        <div class="section-card" style="margin-top:2rem;">
            <div class="section-label"><span class="num">6</span> ECHEANCES</div>
            <div class="section-heading">Vue multi-contrats</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    all_maturities = generate_maturity_schedule(datetime.combine(effective_eval_date, datetime.min.time()))
    if all_maturities:
        multi_records = []
        for maturity in all_maturities:
            r_maturity = interpolate_rate(
                maturity["days"],
                yc_used,
                target_maturity=maturity["days"],
                valuation_date=effective_eval_date,
            )
            result_maturity = price_future(spot_price, r_maturity, d_used, maturity["days"])
            multi_records.append(
                {
                    "Echeance": maturity["label"],
                    "Date": maturity["date"].strftime("%Y-%m-%d"),
                    "Jours": maturity["days"],
                    "Taux courbe (%)": f"{r_maturity * 100.0:.3f}",
                    "Taux utilise (%)": f"{result_maturity['risk_free_rate_pricing'] * 100.0:.3f}",
                    "Future": f"{result_maturity['future_price']:.2f}",
                    "Base": f"{result_maturity['basis']:.2f}",
                    "Base (%)": f"{result_maturity['basis_pct']:.4f}",
                }
            )
        multi_df = pd.DataFrame(multi_records)
        st.dataframe(multi_df, width="stretch", hide_index=True)

        fig_multi = go.Figure()
        fig_multi.add_trace(
            go.Bar(
                x=[row["Echeance"] for row in multi_records],
                y=[float(row["Future"]) for row in multi_records],
                marker_color=[
                    f"rgba(245, 158, 11, {0.4 + 0.6 * idx / len(multi_records)})"
                    for idx in range(len(multi_records))
                ],
                marker_line=dict(width=0),
                text=[row["Future"] for row in multi_records],
                textposition="outside",
                textfont=dict(color="#fcd34d", size=11, family="JetBrains Mono"),
            )
        )
        fig_multi.add_hline(
            y=spot_price,
            line_dash="dash",
            line_color=COLORS["spot"],
            annotation_text=f"Spot = {spot_price:.2f}",
            annotation_font=dict(color=COLORS["spot"], size=11),
        )
        fig_multi.update_layout(**CHART_LAYOUT, height=400, title="MASI20 - Toutes echeances", yaxis_title="Prix")
        st.plotly_chart(fig_multi, width="stretch")
    else:
        st.info("Aucune echeance disponible depuis la date de calcul.")


    st.markdown(
        f"""
        <div class="app-footer">
            <span style="color:#64748b">{APP_TITLE}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
