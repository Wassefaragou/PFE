import streamlit as st

from futures_pnl.ui import (
    format_currency,
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
init_page("PnL Global")
state = load_app_state()
render_sidebar_tools(state)

global_metrics = state["global_metrics"]
contract_metrics = state["contract_metrics"]
portfolio_view = contract_metrics.copy()
active_contracts = int((contract_metrics["abs_position"] > 0).sum()) if not contract_metrics.empty else 0
profitable_contracts = int((contract_metrics["pnl_management_mad"] > 0).sum()) if not contract_metrics.empty else 0
contracts_with_mtm = int(contract_metrics["mtm_price"].notna().sum()) if not contract_metrics.empty else 0

render_hero(
    "PnL global",
    "Vue de pilotage du portefeuille en methode CMP sequentiel.",
)

render_metric_cards(
    [
        {"label": "P&L economique", "value": format_currency(global_metrics["total_management_pnl"]), "glow": "gold"},
        {"label": "P&L latent", "value": format_currency(global_metrics["total_unrealized_pnl"]), "glow": "blue"},
        {"label": "P&L realise", "value": format_currency(global_metrics["total_realized_pnl"]), "glow": "green"},
        {"label": "P&L comptable", "value": format_currency(global_metrics["total_accounting_pnl"]), "glow": "purple"},
        {"label": "Commissions", "value": format_currency(global_metrics["total_commissions"]), "glow": "red"},
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
        {"label": "Contrats ouverts", "value": str(active_contracts), "glow": "purple"},
        {"label": "Contrats en gain", "value": str(profitable_contracts), "glow": "green"},
        {"label": "Contrats avec cours", "value": str(contracts_with_mtm), "glow": "blue"},
    ],
    columns=5,
)

render_section_header(
    "Synthese globale",
    "Lecture en tables du P&L sequentiel et du sens oppose a prendre dans le portefeuille de replication.",
    step="01",
    label="Global",
)

if contract_metrics.empty:
    render_status_box("Aucune metrique contrat disponible.", kind="info")
else:
    ordered_metrics = contract_metrics.sort_values("pnl_management_mad", ascending=False).reset_index(drop=True)
    ordered_portfolio = portfolio_view.sort_values("pnl_management_mad", ascending=False).reset_index(drop=True)
    winner = ordered_metrics.iloc[0]
    loser = ordered_metrics.sort_values("pnl_management_mad", ascending=True).iloc[0]

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
        ],
        columns=2,
    )

    portfolio_tab, notional_tab, mtm_tab = st.tabs(["Portefeuille", "Notionnel", "Disponibilite cours"])

    with portfolio_tab:
        render_data_table(
            ordered_portfolio,
            [
                "contract_code",
                "underlying_name",
                "side_label",
                "replication_side_label",
                "abs_position",
                "cmp_final_cost",
                "mtm_price",
                "cmp_unrealized",
                "cmp_realized_total",
                "commissions_mad",
                "pnl_management_mad",
                "expiry_alert",
            ],
        )

    with notional_tab:
        render_data_table(
            ordered_portfolio.sort_values(["notional_mad", "contract_code"], ascending=[False, True]),
            [
                "contract_code",
                "underlying_name",
                "replication_side_label",
                "abs_position",
                "cmp_final_cost",
                "notional_mad",
                "position_limit_breach",
                "expiry_alert",
            ],
            label_overrides={
                "cmp_final_cost": "CMP sequentiel",
                "notional_mad": "Notionnel abs.",
            },
        )

    with mtm_tab:
        mtm_summary = ordered_metrics.copy()
        mtm_summary["mtm_coverage"] = mtm_summary["mtm_price"].notna().map(
            {True: "Cours renseigne", False: "Cours manquant"}
        )
        mtm_summary = (
            mtm_summary.groupby("mtm_coverage", as_index=False)
            .agg(
                contract_count=("contract_code", "nunique"),
                abs_position=("abs_position", "sum"),
                notional_mad=("notional_mad", "sum"),
                pnl_management_mad=("pnl_management_mad", "sum"),
            )
            .sort_values(["contract_count", "pnl_management_mad"], ascending=[False, False])
        )
        render_data_table(
            mtm_summary,
            [
                "mtm_coverage",
                "contract_count",
                "abs_position",
                "notional_mad",
                "pnl_management_mad",
            ],
            label_overrides={
                "mtm_coverage": "Disponibilite cours",
                "notional_mad": "Notionnel abs.",
            },
        )

render_footer()
