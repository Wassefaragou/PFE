import streamlit as st

from futures_pnl.history import (
    dashboard_state_from_snapshot,
    get_dashboard_snapshot,
    upsert_today_dashboard_snapshot,
)
from futures_pnl.ui import (
    DASHBOARD_HISTORY_FILTER_KEY,
    format_currency,
    format_pct,
    init_page,
    load_app_state,
    render_data_table,
    render_footer,
    render_hero,
    render_metric_cards,
    render_section_header,
    render_sidebar_tools,
    render_status_box,
)
init_page("Dashboard")
live_state = load_app_state()
dashboard_history = upsert_today_dashboard_snapshot(live_state)
preselected_history_date = st.session_state.get(DASHBOARD_HISTORY_FILTER_KEY)
sidebar_snapshot = get_dashboard_snapshot(dashboard_history, preselected_history_date)
sidebar_state = dashboard_state_from_snapshot(sidebar_snapshot, live_state)
selected_history_date = render_sidebar_tools(sidebar_state, dashboard_history=dashboard_history)
selected_snapshot = get_dashboard_snapshot(dashboard_history, selected_history_date)
state = dashboard_state_from_snapshot(selected_snapshot, live_state)

global_metrics = state["global_metrics"]
contract_metrics = state["contract_metrics"]
alerts = state["alerts"]
dashboard_history_date = state.get("dashboard_history_date", "")

portfolio_view = state.get("dashboard_portfolio_view", contract_metrics.copy())
open_cmp_view = state.get("dashboard_open_cmp_view")
if open_cmp_view is None:
    open_cmp_view = (
        portfolio_view.loc[portfolio_view["abs_position"] > 0].copy()
        if not portfolio_view.empty and "abs_position" in portfolio_view.columns
        else portfolio_view.head(0).copy()
    )

render_hero(
    "Index Futures P&L Tracker",
    "Vue d'ensemble du portefeuille avec le calcul P&L principal du tracker. La page CMP sequentiel presente l'autre methode.",
    badges=[f"Jour {dashboard_history_date}"] if dashboard_history_date else None,
)

render_metric_cards(
    [
        {"label": "P&L economique", "value": format_currency(global_metrics["total_management_pnl"]), "glow": "gold"},
        {"label": "P&L comptable", "value": format_currency(global_metrics["total_accounting_pnl"]), "glow": "purple"},
        {"label": "P&L latent", "value": format_currency(global_metrics["total_unrealized_pnl"]), "glow": "blue"},
        {"label": "P&L realise", "value": format_currency(global_metrics["total_realized_pnl"]), "glow": "green"},
        {
            "label": "Notionnel futures long",
            "value": format_currency(global_metrics["open_notional_futures_long"]),
            "glow": "green",
        },
        {
            "label": "Notionnel futures short",
            "value": format_currency(global_metrics["open_notional_futures_short"]),
            "glow": "purple",
        },
        {
            "label": "Exposition globale",
            "value": format_currency(global_metrics.get("global_exposure", 0.0)),
            "glow": "blue",
        },
        {"label": "Marge mobilisee", "value": format_currency(global_metrics["total_margin"]), "glow": "pink"},
        {"label": "Levier global", "value": f"{global_metrics['global_leverage']:.2f}x", "glow": "gold"},
        {"label": "Commissions", "value": format_currency(global_metrics["total_commissions"]), "glow": "red"},
        {"label": "ROI sur marge", "value": format_pct(global_metrics["roi_on_margin"]), "glow": "green"},
    ],
    columns=5,
)

render_section_header(
    "Vue portefeuille",
    "Alertes, etat des donnees et lecture rapide des positions.",
    step="01",
    label="Dashboard",
)

col_alerts, col_health = st.columns([1.4, 1])
with col_alerts:
    st.subheader("Alertes")
    if not alerts:
        render_status_box("Aucune alerte detectee.", kind="success")
    else:
        for alert in alerts:
            render_status_box(alert["message"], kind=alert["severity"])

with col_health:
    st.subheader("Chiffres cles")
    st.metric("Contrats", len(state["contracts_raw"]))
    st.metric("Transactions", len(state["transactions_raw"]))
    st.metric(
        "Contrats ouverts",
        int((contract_metrics["abs_position"] > 0).sum()) if not contract_metrics.empty else 0,
    )

render_section_header(
    "Portefeuille par contrat",
    "Vue P&L par contrat avec notionnel absolu et sens oppose a prendre dans le portefeuille de replication.",
    step="02",
    label="Portfolio",
)

if portfolio_view.empty:
    render_status_box(
        "Aucune metrique disponible pour le moment. Commencez par creer des contrats et des transactions.",
        kind="info",
    )
else:
    render_data_table(
        portfolio_view.sort_values(["pnl_management_mad", "contract_code"], ascending=[False, True]),
        [
            "contract_code",
            "underlying_name",
            "side_label",
            "replication_side_label",
            "abs_position",
            "entry_wap",
            "mtm_price",
            "pnl_unrealized_mad",
            "pnl_realized_mad",
            "pnl_management_mad",
            "margin_mad",
            "notional_mad",
            "position_limit_breach",
            "expiry_alert",
        ],
        label_overrides={
            "notional_mad": "Notionnel abs.",
        },
    )

render_section_header(
    "CMP WAP par contrat",
    "Lecture rapide du CMP WAP sur les positions ouvertes. La methode sequentielle reste sur sa page dediee.",
    step="03",
    label="WAP",
)

if open_cmp_view.empty:
    render_status_box("Aucun CMP a afficher pour le moment.", kind="info")
else:
    render_data_table(
        open_cmp_view.sort_values(["abs_position", "contract_code"], ascending=[False, True]),
        [
            "contract_code",
            "underlying_name",
            "side_label",
            "replication_side_label",
            "abs_position",
            "entry_wap",
            "mtm_price",
            "delta_points",
            "expiry_alert",
        ],
        label_overrides={
            "abs_position": "Position",
            "delta_points": "Ecart cours/CMP WAP",
        },
    )

render_section_header(
    "Positions confirmees",
    "Cette vue est informative uniquement et ne remplace jamais le moteur officiel.",
    step="04",
    label="Informative",
)
if state["confirmed_positions"].empty:
    render_status_box("Aucune vue confirmee disponible.", kind="info")
else:
    render_data_table(
        state["confirmed_positions"].sort_values(["delta_vs_all", "contract_code"], ascending=[False, True]),
        [
            "contract_code",
            "official_net_position",
            "confirmed_net_position",
            "delta_vs_all",
        ],
    )

render_footer()
