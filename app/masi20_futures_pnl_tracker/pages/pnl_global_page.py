import altair as alt
import pandas as pd
import streamlit as st

from futures_pnl.ui import (
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


def _ranked_pnl_chart(dataframe: pd.DataFrame) -> alt.Chart:
    chart_data = dataframe[["contract_code", "pnl_management_mad", "abs_position", "mtm_source", "leverage"]].copy()
    chart_data = chart_data.sort_values("pnl_management_mad", ascending=True).reset_index(drop=True)
    order = chart_data["contract_code"].tolist()
    base = alt.Chart(chart_data)
    zero_rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=ZERO_LINE, strokeDash=[6, 4]).encode(x="x:Q")
    bars = base.mark_bar(cornerRadiusEnd=6, cornerRadiusTopLeft=6, cornerRadiusBottomLeft=6, size=28).encode(
        x=alt.X("pnl_management_mad:Q", title="P&L economique (MAD)", axis=alt.Axis(format="~s")),
        y=alt.Y("contract_code:N", title=None, sort=order, axis=alt.Axis(labelFontSize=12, labelLimit=200)),
        color=alt.condition(
            "datum.pnl_management_mad >= 0",
            alt.value(POSITIVE_COLOR),
            alt.value(NEGATIVE_COLOR),
        ),
        tooltip=[
            alt.Tooltip("contract_code:N", title="Contrat"),
            alt.Tooltip("pnl_management_mad:Q", title="P&L economique", format=",.2f"),
            alt.Tooltip("abs_position:Q", title="Position", format=",.0f"),
            alt.Tooltip("mtm_source:N", title="Source MtM"),
            alt.Tooltip("leverage:Q", title="Levier", format=",.2f"),
        ],
    )
    labels = base.mark_text(align="left", baseline="middle", dx=6, fontSize=12, fontWeight="bold").encode(
        x=alt.X("pnl_management_mad:Q"),
        y=alt.Y("contract_code:N", sort=order),
        text=alt.Text("pnl_management_mad:Q", format=",.0f"),
        color=alt.condition(
            "datum.pnl_management_mad >= 0",
            alt.value(POSITIVE_COLOR),
            alt.value(NEGATIVE_COLOR),
        ),
    )
    return _chart_theme(zero_rule + bars + labels, height=max(220, 56 * len(chart_data)))


def _realized_vs_unrealized_chart(dataframe: pd.DataFrame) -> alt.Chart:
    chart_data = dataframe[["contract_code", "pnl_realized_mad", "pnl_unrealized_mad"]].copy()
    chart_data = chart_data.sort_values("pnl_realized_mad", ascending=True).reset_index(drop=True)
    order = chart_data["contract_code"].tolist()
    chart_data = chart_data.melt(
        id_vars="contract_code",
        value_vars=["pnl_realized_mad", "pnl_unrealized_mad"],
        var_name="component",
        value_name="amount",
    )
    component_labels = {
        "pnl_realized_mad": "P&L realise",
        "pnl_unrealized_mad": "P&L latent",
    }
    chart_data["component_label"] = chart_data["component"].map(component_labels)
    base = alt.Chart(chart_data)
    zero_rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=ZERO_LINE, strokeDash=[6, 4]).encode(x="x:Q")
    bars = base.mark_bar(cornerRadiusEnd=5, size=18).encode(
        x=alt.X("amount:Q", title="Montant (MAD)", axis=alt.Axis(format="~s")),
        y=alt.Y("contract_code:N", title=None, sort=order, axis=alt.Axis(labelFontSize=12, labelLimit=200)),
        yOffset=alt.YOffset("component_label:N"),
        color=alt.Color(
            "component_label:N",
            title="Composante",
            scale=alt.Scale(
                domain=["P&L realise", "P&L latent"],
                range=[ACCENT_GOLD, ACCENT_BLUE],
            ),
        ),
        tooltip=[
            alt.Tooltip("contract_code:N", title="Contrat"),
            alt.Tooltip("component_label:N", title="Composante"),
            alt.Tooltip("amount:Q", title="Montant", format=",.2f"),
        ],
    )
    labels = base.mark_text(align="left", dx=4, fontSize=10, color=NEUTRAL_TEXT).encode(
        x=alt.X("amount:Q"),
        y=alt.Y("contract_code:N", sort=order),
        yOffset=alt.YOffset("component_label:N"),
        text=alt.Text("amount:Q", format=",.0f"),
    )
    return _chart_theme(zero_rule + bars + labels, height=max(220, 60 * len(dataframe)))


def _capital_scatter_chart(dataframe: pd.DataFrame) -> alt.Chart:
    chart_data = dataframe[
        ["contract_code", "margin_mad", "notional_mad", "abs_position", "leverage", "side_label"]
    ].copy()
    circles = alt.Chart(chart_data).mark_circle(opacity=0.85, stroke="#0b0f19", strokeWidth=2).encode(
        x=alt.X("margin_mad:Q", title="Marge (MAD)", axis=alt.Axis(format="~s")),
        y=alt.Y("notional_mad:Q", title="Exposition ouverte (MAD)", axis=alt.Axis(format="~s")),
        size=alt.Size(
            "abs_position:Q",
            title="Position (lots)",
            scale=alt.Scale(range=[200, 1200]),
        ),
        color=alt.Color(
            "side_label:N",
            title="Sens",
            scale=alt.Scale(
                domain=["LONG", "SHORT", "FLAT"],
                range=[POSITIVE_COLOR, NEGATIVE_COLOR, ACCENT_PURPLE],
            ),
        ),
        tooltip=[
            alt.Tooltip("contract_code:N", title="Contrat"),
            alt.Tooltip("margin_mad:Q", title="Marge", format=",.2f"),
            alt.Tooltip("notional_mad:Q", title="Exposition ouverte", format=",.2f"),
            alt.Tooltip("abs_position:Q", title="Position", format=",.0f"),
            alt.Tooltip("leverage:Q", title="Levier", format=",.2f"),
            alt.Tooltip("side_label:N", title="Sens"),
        ],
    )
    labels = alt.Chart(chart_data).mark_text(
        align="center",
        baseline="bottom",
        dy=-12,
        color=NEUTRAL_TEXT,
        fontSize=11,
        fontWeight="bold",
    ).encode(
        x="margin_mad:Q",
        y="notional_mad:Q",
        text="contract_code:N",
    )
    return _chart_theme(circles + labels, height=360)


def _leverage_chart(dataframe: pd.DataFrame) -> alt.Chart:
    chart_data = dataframe[["contract_code", "leverage", "abs_position", "side_label"]].copy()
    chart_data = chart_data.sort_values("leverage", ascending=True).reset_index(drop=True)
    order = chart_data["contract_code"].tolist()
    base = alt.Chart(chart_data)
    bars = base.mark_bar(cornerRadiusEnd=6, cornerRadiusTopLeft=6, cornerRadiusBottomLeft=6, size=28).encode(
        x=alt.X("leverage:Q", title="Levier (x)", axis=alt.Axis(format=",.2f")),
        y=alt.Y("contract_code:N", title=None, sort=order, axis=alt.Axis(labelFontSize=12, labelLimit=200)),
        color=alt.Color(
            "side_label:N",
            title="Sens",
            scale=alt.Scale(
                domain=["LONG", "SHORT", "FLAT"],
                range=[POSITIVE_COLOR, NEGATIVE_COLOR, ACCENT_PURPLE],
            ),
        ),
        tooltip=[
            alt.Tooltip("contract_code:N", title="Contrat"),
            alt.Tooltip("leverage:Q", title="Levier", format=",.2f"),
            alt.Tooltip("abs_position:Q", title="Position", format=",.0f"),
            alt.Tooltip("side_label:N", title="Sens"),
        ],
    )
    labels = base.mark_text(align="left", baseline="middle", dx=6, fontSize=12, fontWeight="bold", color=NEUTRAL_TEXT).encode(
        x="leverage:Q",
        y=alt.Y("contract_code:N", sort=order),
        text=alt.Text("leverage:Q", format=",.2f"),
    )
    return _chart_theme(bars + labels, height=max(220, 56 * len(chart_data)))


def _mtm_source_chart(dataframe: pd.DataFrame) -> alt.Chart:
    chart_data = dataframe.copy()
    chart_data["source_label"] = chart_data["mtm_source"].map(
        {"settlement": "Settlement", "theoretical": "Theorique"}
    ).fillna(chart_data["mtm_source"])
    chart_data = chart_data.melt(
        id_vars="source_label",
        value_vars=["pnl_management_mad", "margin_mad"],
        var_name="metric",
        value_name="amount",
    )
    chart_data["metric_label"] = chart_data["metric"].map(
        {"pnl_management_mad": "P&L economique", "margin_mad": "Marge"}
    )
    base = alt.Chart(chart_data)
    bars = base.mark_bar(cornerRadiusEnd=6, size=26).encode(
        x=alt.X("amount:Q", title="Montant (MAD)", axis=alt.Axis(format="~s")),
        y=alt.Y("source_label:N", title=None, axis=alt.Axis(labelFontSize=12, labelLimit=200)),
        yOffset=alt.YOffset("metric_label:N"),
        color=alt.Color(
            "metric_label:N",
            title="Mesure",
            scale=alt.Scale(
                domain=["P&L economique", "Marge"],
                range=[ACCENT_GOLD, ACCENT_PURPLE],
            ),
        ),
        tooltip=[
            alt.Tooltip("source_label:N", title="Source MtM"),
            alt.Tooltip("metric_label:N", title="Mesure"),
            alt.Tooltip("amount:Q", title="Montant", format=",.2f"),
        ],
    )
    labels = base.mark_text(align="left", dx=4, fontSize=10, color=NEUTRAL_TEXT).encode(
        x="amount:Q",
        y=alt.Y("source_label:N"),
        yOffset=alt.YOffset("metric_label:N"),
        text=alt.Text("amount:Q", format=",.0f"),
    )
    return _chart_theme(bars + labels, height=250)


init_page("PnL Global")
state = load_app_state()
render_sidebar_tools(state)

global_metrics = state["global_metrics"]
contract_metrics = state["contract_metrics"]
active_contracts = int((contract_metrics["abs_position"] > 0).sum()) if not contract_metrics.empty else 0
profitable_contracts = int((contract_metrics["pnl_management_mad"] > 0).sum()) if not contract_metrics.empty else 0
settlement_sourced = int((contract_metrics["mtm_source"] == "settlement").sum()) if not contract_metrics.empty else 0

render_hero(
    "PnL global",
    "Vue de pilotage du portefeuille avec contribution par contrat, structure du capital et lecture des sources MtM.",
    badges=[("Vue portefeuille", ""), ("Capital", "purple"), ("Sources MtM", "green")],
)

render_metric_cards(
    [
        {"label": "P&L economique", "value": format_currency(global_metrics["total_management_pnl"]), "glow": "gold"},
        {"label": "P&L latent", "value": format_currency(global_metrics["total_unrealized_pnl"]), "glow": "blue"},
        {"label": "P&L realise", "value": format_currency(global_metrics["total_realized_pnl"]), "glow": "green"},
        {"label": "P&L comptable", "value": format_currency(global_metrics["total_accounting_pnl"]), "glow": "purple"},
        {"label": "Commissions", "value": format_currency(global_metrics["total_commissions"]), "glow": "red"},
        {"label": "Exposition ouverte (abs.)", "value": format_currency(global_metrics["total_notional"]), "glow": "blue"},
        {"label": "Exposition nette (signee)", "value": format_currency(global_metrics["total_net_notional"]), "glow": "purple"},
        {"label": "Marge", "value": format_currency(global_metrics["total_margin"]), "glow": "pink"},
        {"label": "Levier", "value": f"{global_metrics['global_leverage']:.2f}x", "glow": "gold"},
        {"label": "ROI marge", "value": format_pct(global_metrics["roi_on_margin"]), "glow": "green"},
        {"label": "Contrats ouverts", "value": str(active_contracts), "glow": "purple"},
        {"label": "Contrats gagnants", "value": str(profitable_contracts), "glow": "green"},
        {"label": "MtM settlement", "value": str(settlement_sourced), "glow": "blue"},
    ],
    columns=5,
)

render_section_header(
    "Pilotage portefeuille",
    "Visualisations clarifiees du P&L, du capital engage par contrat et des sources de valorisation.",
    step="01",
    label="Global",
)

if contract_metrics.empty:
    render_status_box("Aucune metrique contrat disponible.", kind="info")
else:
    ordered_metrics = contract_metrics.sort_values("pnl_management_mad", ascending=False).reset_index(drop=True)
    winner = ordered_metrics.iloc[0]
    loser = ordered_metrics.sort_values("pnl_management_mad", ascending=True).iloc[0]
    levered = ordered_metrics.sort_values("leverage", ascending=False).iloc[0]

    render_metric_cards(
        [
            {
                "label": f"Meilleur contrat - {winner['contract_code']}",
                "value": format_currency(float(winner["pnl_management_mad"])),
                "glow": "green",
            },
            {
                "label": f"Moins bon contrat - {loser['contract_code']}",
                "value": format_currency(float(loser["pnl_management_mad"])),
                "glow": "red",
            },
            {
                "label": f"Levier max - {levered['contract_code']}",
                "value": f"{float(levered['leverage']):.2f}x",
                "glow": "purple",
            },
        ],
        columns=3,
    )

    portfolio_tab, capital_tab, mtm_tab = st.tabs(["Portefeuille", "Capital", "Sources MtM"])

    with portfolio_tab:
        render_data_table(
            ordered_metrics,
            [
                "contract_code",
                "mtm_source",
                "side_label",
                "abs_position",
                "entry_wap",
                "mtm_price",
                "pnl_unrealized_mad",
                "pnl_realized_mad",
                "commissions_mad",
                "pnl_management_mad",
                "margin_mad",
                "leverage",
                "expiry_alert",
            ],
        )
        pnl_col, breakdown_col = st.columns(2)
        with pnl_col:
            st.subheader("Classement P&L economique")
            st.altair_chart(_ranked_pnl_chart(ordered_metrics), width='stretch')
        with breakdown_col:
            st.subheader("Structure realisee vs latente")
            st.altair_chart(_realized_vs_unrealized_chart(ordered_metrics), width='stretch')

    with capital_tab:
        render_data_table(
            ordered_metrics,
            [
                "contract_code",
                "abs_position",
                "margin_mad",
                "notional_mad",
                "signed_notional_mad",
                "leverage",
                "position_limit_breach",
                "expiry_alert",
            ],
            label_overrides={
                "notional_mad": "Exposition ouverte (abs.)",
                "signed_notional_mad": "Exposition nette (signee)",
            },
        )
        capital_col, leverage_col = st.columns(2)
        with capital_col:
            st.subheader("Carte marge / exposition ouverte")
            st.altair_chart(_capital_scatter_chart(ordered_metrics), width='stretch')
        with leverage_col:
            st.subheader("Classement du levier")
            st.altair_chart(_leverage_chart(ordered_metrics), width='stretch')

    with mtm_tab:
        mtm_summary = (
            ordered_metrics.groupby("mtm_source", as_index=False)
            .agg(
                contract_count=("contract_code", "nunique"),
                abs_position=("abs_position", "sum"),
                notional_mad=("notional_mad", "sum"),
                margin_mad=("margin_mad", "sum"),
                pnl_management_mad=("pnl_management_mad", "sum"),
            )
            .sort_values("pnl_management_mad", ascending=False)
        )
        summary_col, composition_col = st.columns([1.05, 1])
        with summary_col:
            render_data_table(mtm_summary)
        with composition_col:
            st.subheader("Impact par source MtM")
            st.altair_chart(_mtm_source_chart(mtm_summary), width='stretch')

render_footer()
