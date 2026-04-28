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

init_page("CMP Sequentiel")
state = load_app_state()
render_sidebar_tools(state)

cmp_summary = state["cmp_summary"]
cmp_detail = state["cmp_detail"]
contract_metrics = state["contract_metrics"]

render_hero(
    "CMP sequentiel",
    "Methode officielle de calcul CMP, traitee trade par trade.",
)

checked_count = len(cmp_summary)
open_count = int((contract_metrics["abs_position"] > 0).sum()) if not contract_metrics.empty else 0
cmp_realized = float(cmp_summary["cmp_realized_total"].sum()) if not cmp_summary.empty else 0.0
cmp_unrealized = float(cmp_summary["cmp_unrealized"].sum()) if not cmp_summary.empty else 0.0
cmp_total = float(cmp_summary["cmp_total"].sum()) if not cmp_summary.empty else 0.0

render_metric_cards(
    [
        {"label": "Contrats calcules", "value": str(checked_count), "glow": "gold"},
        {"label": "Contrats ouverts", "value": str(open_count), "glow": "green"},
        {"label": "P&L realise sequentiel", "value": format_currency(cmp_realized), "glow": "blue"},
        {"label": "P&L latent sequentiel", "value": format_currency(cmp_unrealized), "glow": "green"},
        {"label": "P&L total sequentiel", "value": format_currency(cmp_total), "glow": "purple"},
    ],
    columns=5,
)

render_section_header(
    "Synthese sequentielle",
    "Position finale, CMP final et P&L calcule trade par trade.",
    step="01",
    label="CMP",
)

if cmp_summary.empty:
    render_status_box("Aucun resultat CMP disponible.", kind="info")
else:
    ordered_summary = (
        cmp_summary.assign(abs_cmp_total=cmp_summary["cmp_total"].abs())
        .sort_values(["abs_cmp_total", "contract_code"], ascending=[False, True])
        .drop(columns=["abs_cmp_total"])
    )

    render_data_table(
        ordered_summary,
        [
            "contract_code",
            "cmp_final_position",
            "cmp_final_cost",
            "cmp_realized_total",
            "cmp_unrealized",
            "cmp_total",
        ],
        label_overrides={
            "cmp_realized_total": "P&L realise sequentiel",
            "cmp_final_cost": "CMP sequentiel",
            "cmp_unrealized": "P&L latent sequentiel",
            "cmp_total": "P&L total sequentiel",
        },
    )

    render_section_header(
        "Lecture trade par trade",
        "Detail des positions, clotures et CMP sequentiel apres chaque execution.",
        step="02",
        label="Details",
    )
    contract_options = cmp_summary["contract_code"].tolist()
    selected_contract = st.selectbox("Contrat a inspecter", options=contract_options)
    detail_view = cmp_detail.loc[cmp_detail["contract_code"] == selected_contract]
    render_data_table(
        detail_view,
        [
            "execution_id",
            "trade_date",
            "side_lp",
            "quantity_lots",
            "price_points",
            "signed_qty",
            "pos_before",
            "cmp_before",
            "closed_qty",
            "trade_realized_pnl",
            "pos_after",
            "cmp_after",
        ],
        label_overrides={
            "cmp_before": "CMP seq avant",
            "cmp_after": "CMP seq apres trade",
        },
    )

render_footer()
