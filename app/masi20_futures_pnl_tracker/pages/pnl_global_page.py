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


def _official_portfolio_view(contract_metrics: pd.DataFrame) -> pd.DataFrame:
    if contract_metrics.empty:
        return contract_metrics
    return contract_metrics.copy()


init_page("PnL Global")
state = load_app_state()
render_sidebar_tools(state)

global_metrics = state["global_metrics"]
contract_metrics = state["contract_metrics"]
portfolio_view = _official_portfolio_view(contract_metrics)
active_contracts = int((contract_metrics["abs_position"] > 0).sum()) if not contract_metrics.empty else 0
profitable_contracts = int((contract_metrics["pnl_management_mad"] > 0).sum()) if not contract_metrics.empty else 0
contracts_with_mtm = int(contract_metrics["mtm_price"].notna().sum()) if not contract_metrics.empty else 0

render_hero(
    "PnL global",
    "Vue de pilotage du portefeuille en methode WAP. La page CMP sequentiel presente l'autre methode de calcul.",
)

render_metric_cards(
    [
        {"label": "P&L economique", "value": format_currency(global_metrics["total_management_pnl"]), "glow": "gold"},
        {"label": "P&L latent", "value": format_currency(global_metrics["total_unrealized_pnl"]), "glow": "blue"},
        {"label": "P&L realise", "value": format_currency(global_metrics["total_realized_pnl"]), "glow": "green"},
        {"label": "P&L comptable", "value": format_currency(global_metrics["total_accounting_pnl"]), "glow": "purple"},
        {"label": "Commissions", "value": format_currency(global_metrics["total_commissions"]), "glow": "red"},
        {"label": "Exposition brute", "value": format_currency(global_metrics["total_notional"]), "glow": "blue"},
        {"label": "Exposition nette", "value": format_currency(global_metrics["total_net_notional"]), "glow": "purple"},
        {"label": "Marge mobilisee", "value": format_currency(global_metrics["total_margin"]), "glow": "pink"},
        {"label": "Levier global", "value": f"{global_metrics['global_leverage']:.2f}x", "glow": "gold"},
        {"label": "ROI sur marge", "value": format_pct(global_metrics["roi_on_margin"]), "glow": "green"},
        {"label": "Contrats ouverts", "value": str(active_contracts), "glow": "purple"},
        {"label": "Contrats en gain", "value": str(profitable_contracts), "glow": "green"},
        {"label": "Contrats avec cours", "value": str(contracts_with_mtm), "glow": "blue"},
    ],
    columns=5,
)

render_section_header(
    "Synthese globale",
    "Lecture en tables du P&L, du capital engage par contrat et de la disponibilite des cours.",
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

    portfolio_tab, capital_tab, mtm_tab = st.tabs(["Portefeuille", "Capital", "Disponibilite cours"])

    with portfolio_tab:
        render_data_table(
            ordered_portfolio,
            [
                "contract_code",
                "underlying_name",
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

    with capital_tab:
        render_data_table(
            ordered_portfolio.sort_values(["capital_engaged_mad", "contract_code"], ascending=[False, True]),
            [
                "contract_code",
                "underlying_name",
                "abs_position",
                "peak_abs_position",
                "entry_wap",
                "margin_mad",
                "capital_engaged_mad",
                "notional_mad",
                "signed_notional_mad",
                "leverage",
                "position_limit_breach",
                "expiry_alert",
            ],
            label_overrides={
                "entry_wap": "CMP WAP",
                "notional_mad": "Exposition brute",
                "signed_notional_mad": "Exposition nette",
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
                margin_mad=("margin_mad", "sum"),
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
                "margin_mad",
                "pnl_management_mad",
            ],
            label_overrides={
                "mtm_coverage": "Disponibilite cours",
            },
        )

render_footer()
