import altair as alt
import pandas as pd
import streamlit as st

from futures_pnl.ui import (
    format_currency,
    format_pct,
    init_page,
    load_app_state,
    project_tree_text,
    render_data_table,
    render_footer,
    render_hero,
    render_metric_cards,
    render_section_header,
    render_sidebar_tools,
    render_status_box,
)

POSITIVE_COLOR = "#00d4aa"
NEGATIVE_COLOR = "#ff6b6b"
ACCENT_BLUE = "#4facfe"
ACCENT_GOLD = "#f59e0b"
ACCENT_PURPLE = "#8b5cf6"
NEUTRAL_TEXT = "#cbd5e1"
ZERO_LINE = "#64748b"


def _chart_theme(chart: alt.Chart, *, height: int) -> alt.Chart:
    return chart.properties(height=height).configure_axis(
        gridColor="rgba(148, 163, 184, 0.12)",
        domain=False,
        labelColor=NEUTRAL_TEXT,
        titleColor=NEUTRAL_TEXT,
        titleFontSize=12,
        titleFontWeight="normal",
        labelFontSize=11,
        tickColor="rgba(148, 163, 184, 0.25)",
    ).configure_view(strokeOpacity=0).configure_legend(
        labelColor=NEUTRAL_TEXT,
        titleColor=NEUTRAL_TEXT,
        labelFontSize=11,
        titleFontSize=12,
        orient="top",
        direction="horizontal",
        strokeColor="transparent",
    )


def _dashboard_pnl_chart(dataframe: pd.DataFrame) -> alt.Chart:
    chart_data = dataframe[["contract_code", "pnl_management_mad", "pnl_unrealized_mad", "pnl_realized_mad"]].copy()
    # Sort by total P&L for clarity
    chart_data = chart_data.sort_values("pnl_management_mad", ascending=True).reset_index(drop=True)
    order = chart_data["contract_code"].tolist()
    chart_data = chart_data.melt(
        id_vars="contract_code",
        value_vars=["pnl_management_mad", "pnl_unrealized_mad", "pnl_realized_mad"],
        var_name="component",
        value_name="amount",
    )
    labels_map = {
        "pnl_management_mad": "P&L economique",
        "pnl_unrealized_mad": "P&L latent",
        "pnl_realized_mad": "P&L realise",
    }
    chart_data["component_label"] = chart_data["component"].map(labels_map)

    zero_rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
        color=ZERO_LINE, strokeDash=[6, 4]
    ).encode(x="x:Q")

    base = alt.Chart(chart_data)
    bars = base.mark_bar(cornerRadiusEnd=5, size=10).encode(
        y=alt.Y("contract_code:N", title=None, sort=order, axis=alt.Axis(labelFontSize=11, labelLimit=200)),
        yOffset=alt.YOffset("component_label:N"),
        x=alt.X("amount:Q", title="Montant (MAD)", axis=alt.Axis(format="~s")),
        color=alt.Color(
            "component_label:N",
            title="Composante",
            scale=alt.Scale(
                domain=["P&L economique", "P&L latent", "P&L realise"],
                range=[ACCENT_GOLD, ACCENT_BLUE, POSITIVE_COLOR],
            ),
        ),
        tooltip=[
            alt.Tooltip("contract_code:N", title="Contrat"),
            alt.Tooltip("component_label:N", title="Composante"),
            alt.Tooltip("amount:Q", title="Montant (MAD)", format=",.2f"),
        ],
    )
    labels = base.mark_text(align="left", dx=4, fontSize=10, color=NEUTRAL_TEXT).encode(
        y=alt.Y("contract_code:N", sort=order),
        yOffset=alt.YOffset("component_label:N"),
        x=alt.X("amount:Q"),
        text=alt.Text("amount:Q", format=",.0f"),
    )
    n_contracts = dataframe["contract_code"].nunique()
    dynamic_height = max(280, n_contracts * 100)
    return _chart_theme(zero_rule + bars + labels, height=dynamic_height)


def _dashboard_capital_chart(dataframe: pd.DataFrame) -> alt.Chart:
    chart_data = dataframe[["contract_code", "margin_mad", "notional_mad"]].copy()
    chart_data = chart_data.sort_values("notional_mad", ascending=True).reset_index(drop=True)
    order = chart_data["contract_code"].tolist()
    chart_data = chart_data.melt(
        id_vars="contract_code",
        value_vars=["margin_mad", "notional_mad"],
        var_name="metric",
        value_name="amount",
    )
    labels_map = {"margin_mad": "Marge", "notional_mad": "Exposition ouverte"}
    chart_data["metric_label"] = chart_data["metric"].map(labels_map)

    base = alt.Chart(chart_data)
    bars = base.mark_bar(cornerRadiusEnd=5, size=14).encode(
        y=alt.Y("contract_code:N", title=None, sort=order, axis=alt.Axis(labelFontSize=11, labelLimit=200)),
        yOffset=alt.YOffset("metric_label:N"),
        x=alt.X("amount:Q", title="Montant (MAD)", axis=alt.Axis(format="~s")),
        color=alt.Color(
            "metric_label:N",
            title="Mesure",
            scale=alt.Scale(
                domain=["Marge", "Exposition ouverte"],
                range=[ACCENT_PURPLE, ACCENT_BLUE],
            ),
        ),
        tooltip=[
            alt.Tooltip("contract_code:N", title="Contrat"),
            alt.Tooltip("metric_label:N", title="Mesure"),
            alt.Tooltip("amount:Q", title="Montant (MAD)", format=",.2f"),
        ],
    )
    labels = base.mark_text(align="left", dx=4, fontSize=10, color=NEUTRAL_TEXT).encode(
        y=alt.Y("contract_code:N", sort=order),
        yOffset=alt.YOffset("metric_label:N"),
        x=alt.X("amount:Q"),
        text=alt.Text("amount:Q", format=",.0f"),
    )
    n_contracts = dataframe["contract_code"].nunique()
    dynamic_height = max(250, n_contracts * 80)
    return _chart_theme(bars + labels, height=dynamic_height)


init_page("Dashboard")
state = load_app_state()
render_sidebar_tools(state)

global_metrics = state["global_metrics"]
contract_metrics = state["contract_metrics"]
alerts = state["alerts"]
cmp_summary = state["cmp_summary"]

render_hero(
    "Index Futures P&L Tracker",
    "Vue d'ensemble du portefeuille avec moteur WAP officiel, controle CMP et pilotage des positions.",
    badges=[("WAP officiel", ""), ("Controle CMP", "purple"), ("Stockage local", "blue")],
)

render_metric_cards(
    [
        {"label": "P&L economique", "value": format_currency(global_metrics["total_management_pnl"]), "glow": "gold"},
        {"label": "P&L comptable", "value": format_currency(global_metrics["total_accounting_pnl"]), "glow": "purple"},
        {"label": "P&L latent", "value": format_currency(global_metrics["total_unrealized_pnl"]), "glow": "blue"},
        {"label": "P&L realise", "value": format_currency(global_metrics["total_realized_pnl"]), "glow": "green"},
        {"label": "Exposition ouverte (abs.)", "value": format_currency(global_metrics["total_notional"]), "glow": "blue"},
        {"label": "Exposition nette (signee)", "value": format_currency(global_metrics["total_net_notional"]), "glow": "purple"},
        {"label": "Marge totale", "value": format_currency(global_metrics["total_margin"]), "glow": "pink"},
        {"label": "Levier global", "value": f"{global_metrics['global_leverage']:.2f}x", "glow": "gold"},
        {"label": "Commissions", "value": format_currency(global_metrics["total_commissions"]), "glow": "red"},
        {"label": "ROI marge", "value": format_pct(global_metrics["roi_on_margin"]), "glow": "green"},
    ],
    columns=5,
)

render_section_header(
    "Radar portefeuille",
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
    st.subheader("Vue rapide")
    st.metric("Contrats", len(state["contracts_raw"]))
    st.metric("Transactions", len(state["transactions_raw"]))
    st.metric(
        "Contrats ouverts",
        int((contract_metrics["abs_position"] > 0).sum()) if not contract_metrics.empty else 0,
    )

render_section_header(
    "Synthese portefeuille",
    "Vue officielle par contrat avec WAP d'entree, MtM, marge et alertes de limite.",
    step="02",
    label="Portfolio",
)

if contract_metrics.empty:
    render_status_box(
        "Aucune metrique disponible pour le moment. Commencez par creer des contrats et des transactions.",
        kind="info",
    )
else:
    display_columns = [
        "contract_code",
        "side_label",
        "abs_position",
        "entry_wap",
        "mtm_price",
        "pnl_unrealized_mad",
        "pnl_realized_mad",
        "pnl_management_mad",
        "margin_mad",
        "signed_notional_mad",
        "position_limit_breach",
        "expiry_alert",
    ]
    render_data_table(
        contract_metrics,
        display_columns,
        label_overrides={
            "signed_notional_mad": "Exposition nette (signee)",
        },
    )

    chart_col_left, chart_col_right = st.columns(2)
    with chart_col_left:
        st.subheader("P&L par contrat")
        st.altair_chart(_dashboard_pnl_chart(contract_metrics), width='stretch')
    with chart_col_right:
        st.subheader("Efficacite du capital")
        st.altair_chart(_dashboard_capital_chart(contract_metrics), width='stretch')

render_section_header(
    "Controles",
    "Vue confirmee, controle CMP et arborescence locale du projet.",
    step="03",
    label="Controles",
)

tab_confirmed, tab_cmp, tab_tree = st.tabs(["Confirmes", "Controle CMP", "Arborescence"])

with tab_confirmed:
    st.caption("Informatif uniquement. Cette vue ne remplace pas le moteur officiel.")
    if state["confirmed_positions"].empty:
        render_status_box("Aucune position confirmee a afficher.", kind="info")
    else:
        render_data_table(state["confirmed_positions"])

with tab_cmp:
    if cmp_summary.empty:
        render_status_box("Aucun resultat CMP disponible pour le moment.", kind="info")
    else:
        ok_count = int(cmp_summary["within_tolerance"].fillna(False).sum())
        render_metric_cards(
            [
                {"label": "Contrats testes", "value": str(len(cmp_summary)), "glow": "purple"},
                {"label": "Dans la tolerance", "value": str(ok_count), "glow": "green"},
                {"label": "Ecarts", "value": str(len(cmp_summary) - ok_count), "glow": "red"},
            ],
            columns=3,
        )
        render_data_table(cmp_summary)

with tab_tree:
    with st.expander("Arborescence du projet", expanded=False):
        st.code(project_tree_text(), language="text")

render_footer()
