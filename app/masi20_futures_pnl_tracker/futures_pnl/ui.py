from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from .analytics import (
    build_dashboard_alerts,
    build_cmp_portfolio_view,
    compute_cmp_sequential,
    compute_cmp_global_metrics,
    compute_confirmed_positions,
    compute_contract_metrics,
    compute_global_metrics,
)
from .config import APP_TITLE, CONTRACT_COLUMNS, TRANSACTION_COLUMNS
from .contracts import prepare_contracts_for_valuation
from .storage import ensure_storage, load_contracts, load_settings, load_transactions, reset_storage
from .validators import validate_contracts, validate_transactions

STYLE_PATH = Path(__file__).resolve().parents[1] / "style.css"
NAVIGATION_STATE_KEY = "selected_app"
PAGE_ICON = "https://www.google.com/s2/favicons?domain=attijariwafabank.com&sz=128"

DISPLAY_LABELS = {
    "parameter": "Parametre",
    "value": "Valeur",
    "severity": "Niveau",
    "code": "Code",
    "row": "Ligne",
    "entity": "Element",
    "message": "Message",
    "default_tick_value": "Tick global",
    "default_initial_margin_per_lot": "Marge init./lot globale",
    "commission_bvc_rt": "Commission BVC AR",
    "commission_broker_rt": "Commission broker AR",
    "commission_sgmat_rt": "Commission SGMAT AR",
    "default_position_limit_per_contract": "Limite position globale",
    "contract_code": "Ticker",
    "underlying_name": "Sous-jacent",
    "expiry_date": "Echeance",
    "tick_value": "Tick",
    "initial_margin_per_lot": "Marge init./lot",
    "settlement_price_points": "Cours valorisation",
    "position_limit_per_contract": "Limite pos.",
    "comments": "Commentaire",
    "effective_tick_value": "Tick effectif",
    "effective_initial_margin_per_lot": "Marge effect.",
    "effective_position_limit_per_contract": "Limite effect.",
    "mtm_price": "Cours retenu",
    "mtm_source": "Source cours",
    "days_to_expiry": "Jours restants",
    "expiry_alert": "Alerte echeance",
    "execution_id": "ID exec",
    "trade_date": "Date",
    "trade_time": "Heure",
    "contract": "Contrat",
    "side_lp": "Sens",
    "quantity_lots": "Qt lots",
    "price_points": "Prix pts",
    "counterparty": "Contrepartie",
    "counterparty_type": "Type cpty",
    "status": "Statut",
    "notional_mad": "Exposition brute",
    "signed_notional_mad": "Exposition nette",
    "is_confirmed": "Confirmee",
    "is_valid_for_calc": "Valide calcul",
    "is_official_for_calc": "Inclus P&L officiel",
    "total_buys_lots": "Lots achetes",
    "total_sells_lots": "Lots vendus",
    "official_net_position": "Pos. officielle",
    "confirmed_net_position": "Pos. confirmee",
    "net_position": "Pos. nette",
    "delta_vs_all": "Ecart vs officiel",
    "side_label": "Sens",
    "direction": "Direction",
    "abs_position": "Position ouverte",
    "wap_buys": "PRU achats",
    "wap_sells": "PRU ventes",
    "entry_wap": "CMP WAP",
    "delta_points": "Ecart cours/CMP WAP",
    "pnl_unrealized_mad": "P&L latent",
    "matched_qty": "Quantite cloturee",
    "pnl_realized_mad": "P&L réalisé",
    "pnl_accounting_mad": "P&L comptable",
    "commissions_mad": "Commissions",
    "pnl_management_mad": "P&L économique",
    "margin_mad": "Marge mobilisée",
    "leverage": "Effet de levier",
    "position_limit_breach": "Limite dépassée",
    "cmp_realized_total": "CMP réalisé",
    "cmp_final_position": "Position finale",
    "cmp_final_cost": "CMP sequentiel",
    "cmp_unrealized": "P&L latent sequentiel",
    "cmp_total": "P&L total sequentiel",
    "wap_accounting_total": "P&L total WAP",
    "difference_vs_wap": "Ecart vs WAP",
    "within_tolerance": "Tol. OK",
    "signed_qty": "Quantite signee",
    "pos_before": "Position avant",
    "cmp_before": "CMP seq avant",
    "closed_qty": "Quantite cloturee",
    "trade_realized_pnl": "P&L réalisé trade",
    "pos_after": "Position après",
    "cmp_after": "CMP après trade",
    "date": "Date",
    "contract_count": "Nb contrats",
    "nb_transactions": "Nb transactions",
    "quantite_totale": "Quantité totale",
    "notionnel_total_mad": "Notionnel total MAD",
    "inclus_pnl_officiel": "Nb inclus P&L officiel",
    "daily_delta": "Delta jour",
    "daily_variation_mad": "Var. jour MAD",
    "total_daily_variation": "Variation totale",
    "cumulative_margin_pnl": "Cumul marge",
    "round_trip_fee_per_lot": "Frais AR / lot",
    "is_valid": "Ligne valide",
    "chrono_key": "Cle chrono",
    "peak_abs_position": "Pic de position",
    "capital_engaged_mad": "Capital engage",
}

VALUE_LABELS = {
    "severity": {
        "error": "Erreur",
        "warning": "Alerte",
        "info": "Info",
        "success": "Succes",
    },
    "side_lp": {
        "BUY": "Achat",
        "SELL": "Vente",
    },
    "side_label": {
        "LONG": "Long",
        "SHORT": "Short",
        "FLAT": "Neutre",
    },
    "mtm_source": {
        "contract": "Referentiel",
        "missing": "Non renseigne",
    },
    "expiry_alert": {
        "Expired": "Expire",
        "Near expiry": "Proche echeance",
        "OK": "OK",
    },
    "status": {
        "CONFIRME": "Confirme",
        "CONFIRMED": "Confirme",
        "CONFIRMED_OK": "Confirme",
        "ATTENTE": "En attente",
        "PENDING": "En attente",
        "REJETE": "Rejete",
        "REJECTED": "Rejete",
        "CANCELLED": "Rejete",
        "CANCELED": "Rejete",
        "FILLED": "Confirme",
        "DONE": "Confirme",
        "PARTIAL": "Confirme",
        "PARTIEL": "Confirme",
    },
}

POSITIVE_FLAG_COLUMNS = {"is_confirmed", "is_valid_for_calc", "is_official_for_calc", "within_tolerance"}
NEGATIVE_FLAG_COLUMNS = {"position_limit_breach"}
BOOLEAN_COLUMNS = POSITIVE_FLAG_COLUMNS | NEGATIVE_FLAG_COLUMNS | {
    "is_valid",
}
MONEY_COLUMNS = {
    "notional_mad",
    "notionnel_total_mad",
    "signed_notional_mad",
    "pnl_unrealized_mad",
    "pnl_realized_mad",
    "pnl_accounting_mad",
    "commissions_mad",
    "pnl_management_mad",
    "margin_mad",
    "cmp_realized_total",
    "cmp_unrealized",
    "cmp_total",
    "wap_accounting_total",
    "trade_realized_pnl",
    "daily_variation_mad",
    "total_daily_variation",
    "cumulative_margin_pnl",
    "effective_initial_margin_per_lot",
    "initial_margin_per_lot",
    "round_trip_fee_per_lot",
    "capital_engaged_mad",
}
POINT_COLUMNS = {
    "settlement_price_points",
    "price_points",
    "mtm_price",
    "wap_buys",
    "wap_sells",
    "entry_wap",
    "delta_points",
    "cmp_before",
    "cmp_after",
    "cmp_final_cost",
}
QUANTITY_COLUMNS = {
    "tick_value",
    "effective_tick_value",
    "nb_transactions",
    "quantite_totale",
    "inclus_pnl_officiel",
    "position_limit_per_contract",
    "effective_position_limit_per_contract",
    "quantity_lots",
    "total_buys_lots",
    "total_sells_lots",
    "net_position",
    "official_net_position",
    "confirmed_net_position",
    "delta_vs_all",
    "abs_position",
    "matched_qty",
    "signed_qty",
    "pos_before",
    "closed_qty",
    "pos_after",
    "cmp_final_position",
    "days_to_expiry",
    "direction",
    "contract_count",
    "peak_abs_position",
}
RATIO_COLUMNS = {"leverage"}


@st.cache_data(show_spinner=False)
def _load_css() -> str:
    if not STYLE_PATH.exists():
        return ""
    return STYLE_PATH.read_text(encoding="utf-8")


def init_page(page_title: str) -> None:
    st.set_page_config(
        page_title=f"{APP_TITLE} - {page_title}",
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    css = _load_css()
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    if st.button(
        "\u2b05\ufe0f Retour a l'accueil",
        key=f"back_to_home_{page_title.lower().replace(' ', '_')}",
    ):
        st.session_state[NAVIGATION_STATE_KEY] = None
        st.rerun()


def format_currency(value: float) -> str:
    return f"{value:,.2f} MAD"


def format_pct(value: float) -> str:
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric_value) or not np.isfinite(float(numeric_value)):
        return "-"
    return f"{float(numeric_value):.2%}"


def format_number(value: float) -> str:
    return f"{value:,.2f}"


def label_for(name: str) -> str:
    return DISPLAY_LABELS.get(name, name.replace("_", " ").strip().capitalize())


def resolved_label(name: str, label_overrides: dict[str, str] | None = None) -> str:
    if label_overrides and name in label_overrides:
        return label_overrides[name]
    return label_for(name)


def format_choice_value(column: str, value: object) -> str:
    if pd.isna(value):
        return "-"
    text = str(value).strip()
    if not text:
        return "-"
    if column in VALUE_LABELS:
        return VALUE_LABELS[column].get(text, text.replace("_", " ").title())
    return text


def _display_bool(value: object) -> str:
    if pd.isna(value):
        return "-"
    return "Oui" if bool(value) else "Non"


def prepare_table_view(
    dataframe: pd.DataFrame,
    columns: list[str] | None = None,
    label_overrides: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_view = dataframe.copy()
    if columns is not None:
        existing_columns = [column for column in columns if column in source_view.columns]
        source_view = source_view[existing_columns]

    display_view = source_view.copy()
    for column in display_view.columns:
        if column in BOOLEAN_COLUMNS or pd.api.types.is_bool_dtype(display_view[column]):
            display_view[column] = display_view[column].map(_display_bool)
        elif column in VALUE_LABELS:
            display_view[column] = display_view[column].map(lambda value, current=column: format_choice_value(current, value))
        elif pd.api.types.is_datetime64_any_dtype(display_view[column]):
            display_view[column] = display_view[column].dt.strftime("%Y-%m-%d")
        elif display_view[column].dtype == "object":
            display_view[column] = display_view[column].replace("", pd.NA).fillna("-")

    display_view = display_view.rename(
        columns={column: resolved_label(column, label_overrides) for column in display_view.columns}
    )
    return source_view, display_view


def _format_value_for_table(value: object, column: str) -> object:
    if pd.isna(value):
        return "-"
    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.notna(numeric_value):
        numeric_value = float(numeric_value)
        if column in MONEY_COLUMNS:
            return f"{numeric_value:,.2f}"
        if column in POINT_COLUMNS:
            return f"{numeric_value:,.4f}"
        if column in RATIO_COLUMNS:
            return f"{numeric_value:,.2f}"
        if column in QUANTITY_COLUMNS:
            return f"{numeric_value:,.2f}" if not float(numeric_value).is_integer() else f"{int(numeric_value):,}"
    return str(value)


def _style_table_value(value: object, column: str) -> str:
    if pd.isna(value):
        return "color: #6b7280;"

    if column in MONEY_COLUMNS | POINT_COLUMNS | RATIO_COLUMNS:
        numeric_value = pd.to_numeric(value, errors="coerce")
        if pd.notna(numeric_value):
            if float(numeric_value) > 0:
                return "color: #86efac; font-weight: 700;"
            if float(numeric_value) < 0:
                return "color: #fca5a5; font-weight: 700;"
            return "color: #cbd5e1;"

    if column in POSITIVE_FLAG_COLUMNS:
        return (
            "color: #86efac; font-weight: 700;"
            if bool(value)
            else "color: #94a3b8;"
        )
    if column in NEGATIVE_FLAG_COLUMNS:
        return (
            "color: #fca5a5; font-weight: 700;"
            if bool(value)
            else "color: #94a3b8;"
        )
    if column == "severity":
        tone = str(value).strip().lower()
        palette = {
            "error": "color: #fca5a5; font-weight: 700;",
            "warning": "color: #fcd34d; font-weight: 700;",
            "info": "color: #93c5fd; font-weight: 700;",
            "success": "color: #86efac; font-weight: 700;",
        }
        return palette.get(tone, "color: #cbd5e1;")
    if column == "side_lp":
        return "color: #93c5fd; font-weight: 700;" if str(value).strip().upper() == "BUY" else "color: #fca5a5; font-weight: 700;"
    if column == "side_label":
        tone = str(value).strip().upper()
        palette = {
            "LONG": "color: #86efac; font-weight: 700;",
            "SHORT": "color: #fca5a5; font-weight: 700;",
            "FLAT": "color: #cbd5e1; font-weight: 700;",
        }
        return palette.get(tone, "color: #cbd5e1;")
    if column == "status":
        tone = str(value).strip().upper()
        if tone in {"CONFIRMED", "CONFIRME", "CONFIRMED_OK", "FILLED", "DONE", "PARTIAL", "PARTIEL"}:
            return "color: #86efac; font-weight: 700;"
        if tone in {"PENDING", "ATTENTE"}:
            return "color: #fcd34d; font-weight: 700;"
        if tone in {"REJECTED", "REJETE", "CANCELLED", "CANCELED", "ANNULE", "ANNULEE"}:
            return "color: #fca5a5; font-weight: 700;"
    if column == "expiry_alert":
        tone = str(value).strip()
        if tone == "Expired":
            return "color: #fca5a5; font-weight: 700;"
        if tone == "Near expiry":
            return "color: #fcd34d; font-weight: 700;"
        if tone == "OK":
            return "color: #86efac; font-weight: 700;"
    if column == "mtm_source":
        tone = str(value).strip().lower()
        if tone == "contract":
            return "color: #fcd34d; font-weight: 700;"
        if tone == "missing":
            return "color: #94a3b8;"
    return ""


def _metric_card_html(label: str, value: str, glow: str = "gold", unit: str = "") -> str:
    unit_html = f'<div class="unit">{escape(unit)}</div>' if unit else ""
    return (
        f'<div class="glass-metric">'
        f'<div class="glow glow-{escape(glow)}"></div>'
        f"<h4>{escape(label)}</h4>"
        f'<div class="val">{escape(value)}</div>'
        f"{unit_html}"
        f"</div>"
    )


def render_metric_cards(cards: list[dict], columns: int = 4) -> None:
    if not cards:
        return
    columns = max(1, columns)
    for start in range(0, len(cards), columns):
        row_cards = cards[start : start + columns]
        row = st.columns(len(row_cards))
        for column, card in zip(row, row_cards):
            with column:
                st.markdown(
                    _metric_card_html(
                        label=str(card.get("label", "")),
                        value=str(card.get("value", "")),
                        glow=str(card.get("glow", "gold")),
                        unit=str(card.get("unit", "")),
                    ),
                    unsafe_allow_html=True,
                )


def _badge_html(label: str, badge_class: str = "") -> str:
    klass = f"hero-badge {badge_class}".strip()
    return f'<span class="{escape(klass)}">{escape(label)}</span>'


def render_hero(title: str, subtitle: str = "", badges: list | None = None) -> None:
    badges = badges or []
    badges_html_parts: list[str] = []
    for badge in badges:
        if isinstance(badge, tuple):
            badge_label, badge_class = badge
            badges_html_parts.append(_badge_html(str(badge_label), str(badge_class)))
        else:
            badges_html_parts.append(_badge_html(str(badge)))
    badges_html = "".join(badges_html_parts)
    subtitle_html = f'<div class="hero-sub">{escape(subtitle)}</div>' if subtitle else ""
    badge_block = f'<div class="hero-badges">{badges_html}</div>' if badges_html else ""
    st.markdown(
        f"""
        <div class="hero-wrap">
            <h1 class="hero-title">{escape(title)}</h1>
            {subtitle_html}
            {badge_block}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(
    title: str,
    subtitle: str = "",
    step: str | int | None = None,
    label: str = "Section",
) -> None:
    num_html = f'<span class="num">{escape(str(step))}</span>' if step is not None else ""
    subtitle_html = f'<div class="section-subtitle">{escape(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-label">{num_html}{escape(label)}</div>
            <div class="section-heading">{escape(title)}</div>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_form_group(title: str, subtitle: str = "") -> None:
    subtitle_html = f'<div class="form-group-subtitle">{escape(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="form-group-title">{escape(title)}</div>
        {subtitle_html}
        """,
        unsafe_allow_html=True,
    )


def render_micro_note(title: str, body: str, tone: str = "info") -> None:
    tone_class = f"micro-note micro-note-{tone}"
    st.markdown(
        f"""
        <div class="{escape(tone_class)}">
            <div class="micro-note-title">{escape(title)}</div>
            <div class="micro-note-body">{escape(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_box(message: str, kind: str = "info") -> None:
    kind_class = {
        "info": "alert-info",
        "success": "alert-success",
        "warning": "alert-warning",
        "error": "alert-error",
    }.get(kind, "alert-info")
    st.markdown(f'<div class="{kind_class}">{escape(message)}</div>', unsafe_allow_html=True)


def render_sidebar_tools(app_state: dict | None = None) -> None:
    ensure_storage()
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-logo">
                <div class="sidebar-logo-mark">PnL</div>
                <div class="sidebar-logo-text">
                    <div class="sidebar-logo-title">Index Futures Tracker</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if app_state is not None:
            contract_count = len(app_state["contracts_raw"])
            trade_count = len(app_state["transactions_raw"])
            open_contracts = (
                int((app_state["contract_metrics"]["abs_position"] > 0).sum())
                if not app_state["contract_metrics"].empty
                else 0
            )
            st.markdown("##### Synthese")
            st.markdown(
                f"""
                <div class="sidebar-config-grid">
                    <div class="sidebar-config-item">
                        <div class="cfg-val">{contract_count}</div>
                        <div class="cfg-label">Contrats</div>
                    </div>
                    <div class="sidebar-config-item">
                        <div class="cfg-val">{trade_count}</div>
                        <div class="cfg-label">Transactions</div>
                    </div>
                    <div class="sidebar-config-item">
                        <div class="cfg-val">{open_contracts}</div>
                        <div class="cfg-label">Positions ouvertes</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("##### Donnees")
        if st.button("Reinitialiser les donnees locales", width="stretch"):
            reset_storage()
            st.rerun()


def show_issues(title: str, issues_df: pd.DataFrame) -> None:
    st.subheader(title)
    if issues_df.empty:
        render_status_box("Aucun probleme detecte.", kind="success")
        return
    render_data_table(issues_df)


def render_footer(text: str = "Stockage local generique uniquement. Aucun jeu de donnees embarque.") -> None:
    st.markdown(f'<div class="app-footer">{escape(text)}</div>', unsafe_allow_html=True)


def get_empty_contracts() -> pd.DataFrame:
    return pd.DataFrame(columns=CONTRACT_COLUMNS)


def get_empty_transactions() -> pd.DataFrame:
    return pd.DataFrame(columns=TRANSACTION_COLUMNS)


def labeled_view(
    dataframe: pd.DataFrame,
    columns: list[str] | None = None,
    label_overrides: dict[str, str] | None = None,
) -> pd.DataFrame:
    _, display_view = prepare_table_view(dataframe, columns, label_overrides)
    return display_view


def render_data_table(
    dataframe: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    height: int | None = None,
    label_overrides: dict[str, str] | None = None,
) -> None:
    source_view, display_view = prepare_table_view(dataframe, columns, label_overrides)
    styles = pd.DataFrame("", index=display_view.index, columns=display_view.columns)
    for source_column in source_view.columns:
        display_column = resolved_label(source_column, label_overrides)
        styles[display_column] = source_view[source_column].map(
            lambda value, current=source_column: _style_table_value(value, current)
        )
        display_view[display_column] = display_view[display_column].map(
            lambda value, current=source_column: _format_value_for_table(value, current)
        )

    styler = (
        display_view.style
        .apply(lambda _: styles, axis=None)
        .set_properties(**{"font-size": "0.88rem"})
    )
    dataframe_kwargs = {
        "width": "stretch",
        "hide_index": True,
    }
    if height is not None:
        dataframe_kwargs["height"] = height
    st.dataframe(styler, **dataframe_kwargs)


def append_dataframe_rows(
    existing_df: pd.DataFrame | None,
    incoming_df: pd.DataFrame | None,
    columns: list[str],
) -> pd.DataFrame:
    existing = existing_df.copy() if existing_df is not None else pd.DataFrame(columns=columns)
    incoming = incoming_df.copy() if incoming_df is not None else pd.DataFrame(columns=columns)

    for dataframe in (existing, incoming):
        for column in columns:
            if column not in dataframe.columns:
                dataframe[column] = pd.NA

    existing = existing[columns]
    incoming = incoming[columns]

    if incoming.empty:
        return existing.reset_index(drop=True)
    if existing.empty:
        return incoming.reset_index(drop=True)

    return pd.concat([existing, incoming], ignore_index=True)

def load_app_state() -> dict:
    ensure_storage()
    settings = load_settings()
    legacy_default_margin = pd.to_numeric(settings.get("default_initial_margin_per_lot"), errors="coerce")
    contracts_raw = load_contracts(fallback_initial_margin_per_lot=legacy_default_margin)
    transactions_raw = load_transactions()

    contracts_validated, contract_issues = validate_contracts(contracts_raw, settings)
    contracts_ready = prepare_contracts_for_valuation(contracts_validated)
    transactions_validated, transaction_issues = validate_transactions(transactions_raw, contracts_ready)
    contract_metrics = compute_contract_metrics(contracts_ready, transactions_validated, settings)
    confirmed_positions = compute_confirmed_positions(transactions_validated, contract_metrics)
    cmp_detail, cmp_summary = compute_cmp_sequential(transactions_validated, contract_metrics)
    cmp_portfolio = build_cmp_portfolio_view(contract_metrics, cmp_summary)
    global_metrics = compute_global_metrics(contract_metrics)
    cmp_global_metrics = compute_cmp_global_metrics(cmp_portfolio)
    alerts = build_dashboard_alerts(
        contracts_ready,
        contract_issues,
        transactions_validated,
        transaction_issues,
        contract_metrics,
    )

    return {
        "settings": settings,
        "contracts_raw": contracts_raw,
        "transactions_raw": transactions_raw,
        "contracts_validated": contracts_validated,
        "contracts_ready": contracts_ready,
        "contract_issues": contract_issues,
        "transactions_validated": transactions_validated,
        "transaction_issues": transaction_issues,
        "contract_metrics": contract_metrics,
        "confirmed_positions": confirmed_positions,
        "cmp_detail": cmp_detail,
        "cmp_summary": cmp_summary,
        "cmp_portfolio": cmp_portfolio,
        "global_metrics": global_metrics,
        "cmp_global_metrics": cmp_global_metrics,
        "alerts": alerts,
    }
