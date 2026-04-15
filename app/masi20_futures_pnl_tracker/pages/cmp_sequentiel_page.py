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

render_hero(
    "CMP sequentiel",
    "Methode alternative de calcul CMP, traitee trade par trade et comparee au moteur WAP.",
)

checked_count = len(cmp_summary)
ok_count = int(cmp_summary["within_tolerance"].fillna(False).sum()) if not cmp_summary.empty else 0
mismatch_count = checked_count - ok_count
cmp_total = float(cmp_summary["cmp_total"].sum()) if not cmp_summary.empty else 0.0

render_metric_cards(
    [
        {"label": "Contrats testés", "value": str(checked_count), "glow": "gold"},
        {"label": "Dans la tolérance", "value": str(ok_count), "glow": "green"},
        {"label": "Écarts", "value": str(mismatch_count), "glow": "red"},
        {"label": "P&L total sequentiel", "value": format_currency(cmp_total), "glow": "purple"},
    ],
    columns=4,
)

render_section_header(
    "Comparaison des deux methodes",
    "Verification du total sequentiel contre le total WAP, contrat par contrat.",
    step="01",
    label="Comparaison",
)

if cmp_summary.empty:
    render_status_box("Aucun resultat CMP disponible.", kind="info")
else:
    if mismatch_count == 0:
        render_status_box(
            "Les deux methodes concordent sur le total P&L dans la tolerance.",
            kind="success",
        )
    else:
        render_status_box(
            "Certains contrats divergent entre la methode sequentielle et la methode WAP.",
            kind="error",
        )

    ordered_summary = (
        cmp_summary.assign(abs_difference_vs_wap=cmp_summary["difference_vs_wap"].abs())
        .sort_values(["abs_difference_vs_wap", "contract_code"], ascending=[False, True])
        .drop(columns=["abs_difference_vs_wap"])
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
            "wap_accounting_total",
            "difference_vs_wap",
            "within_tolerance",
        ],
        label_overrides={
            "cmp_realized_total": "P&L realise sequentiel",
            "cmp_final_cost": "CMP sequentiel",
            "cmp_unrealized": "P&L latent sequentiel",
            "cmp_total": "P&L total sequentiel",
            "wap_accounting_total": "P&L total WAP",
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
