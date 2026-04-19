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

init_page("Position Par Contrat")
state = load_app_state()
render_sidebar_tools(state)

contract_metrics = state["contract_metrics"]
confirmed_positions = state["confirmed_positions"]

render_hero(
    "Position par contrat",
    "Vue officielle du portefeuille par contrat, avec une lecture informative des seuls trades confirmes.",
)

long_count = int((contract_metrics["net_position"] > 0).sum()) if not contract_metrics.empty else 0
short_count = int((contract_metrics["net_position"] < 0).sum()) if not contract_metrics.empty else 0
breach_count = int(contract_metrics["position_limit_breach"].fillna(False).sum()) if not contract_metrics.empty else 0
mgmt_pnl = float(contract_metrics["pnl_management_mad"].sum()) if not contract_metrics.empty else 0.0

render_metric_cards(
    [
        {"label": "Contrats ouverts", "value": str(int((contract_metrics['abs_position'] > 0).sum())) if not contract_metrics.empty else "0", "glow": "gold"},
        {"label": "Contrats longs", "value": str(long_count), "glow": "green"},
        {"label": "Contrats shorts", "value": str(short_count), "glow": "purple"},
        {"label": "P&L économique", "value": format_currency(mgmt_pnl), "glow": "blue"},
        {"label": "Limites dépassées", "value": str(breach_count), "glow": "red"},
    ],
    columns=5,
)

render_section_header(
    "Moteur officiel",
    "Calcul officiel par contrat base sur l'agregation WAP des achats et des ventes. La page CMP sequentiel presente l'autre methode.",
    step="01",
    label="Official",
)

if contract_metrics.empty:
    render_status_box("Aucune metrique contrat a afficher.", kind="info")
else:
    render_data_table(
        contract_metrics.sort_values(["pnl_management_mad", "contract_code"], ascending=[False, True]),
        [
            "contract_code",
            "underlying_name",
            "total_buys_lots",
            "total_sells_lots",
            "net_position",
            "side_label",
            "entry_wap",
            "mtm_price",
            "delta_points",
            "pnl_unrealized_mad",
            "pnl_realized_mad",
            "pnl_accounting_mad",
            "commissions_mad",
            "pnl_management_mad",
            "notional_mad",
            "margin_mad",
            "leverage",
            "position_limit_breach",
            "expiry_alert",
        ],
        label_overrides={
            "notional_mad": "Exposition ouverte (abs.)",
        },
    )

render_section_header(
    "Vue confirmee",
    "Cette vue est informative uniquement et ne remplace jamais le moteur officiel.",
    step="02",
    label="Informative",
)
if confirmed_positions.empty:
    render_status_box("Aucune vue confirmee disponible.", kind="info")
else:
    render_data_table(
        confirmed_positions.sort_values(["delta_vs_all", "contract_code"], ascending=[False, True]),
        [
            "contract_code",
            "official_net_position",
            "confirmed_net_position",
            "delta_vs_all",
        ],
    )

render_footer()
