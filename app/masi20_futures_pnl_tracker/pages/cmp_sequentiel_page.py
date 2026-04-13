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
    "Moteur alternatif traite trade par trade pour controler le P&L comptable du moteur WAP.",
    badges=[("Trade par trade", ""), ("Tolerance", "purple"), ("Controle", "green")],
)

checked_count = len(cmp_summary)
ok_count = int(cmp_summary["within_tolerance"].fillna(False).sum()) if not cmp_summary.empty else 0
mismatch_count = checked_count - ok_count
cmp_total = float(cmp_summary["cmp_total"].sum()) if not cmp_summary.empty else 0.0

render_metric_cards(
    [
        {"label": "Contrats testes", "value": str(checked_count), "glow": "gold"},
        {"label": "Dans la tolerance", "value": str(ok_count), "glow": "green"},
        {"label": "Ecarts", "value": str(mismatch_count), "glow": "red"},
        {"label": "CMP total", "value": format_currency(cmp_total), "glow": "purple"},
    ],
    columns=4,
)

render_section_header(
    "Synthese du controle CMP",
    "Verification du total CMP contre le total comptable WAP, contrat par contrat.",
    step="01",
    label="Controle",
)

if cmp_summary.empty:
    render_status_box("Aucun resultat CMP disponible.", kind="info")
else:
    if mismatch_count == 0:
        render_status_box(
            "Tous les totaux CMP concordent avec le moteur WAP dans la tolerance.",
            kind="success",
        )
    else:
        render_status_box(
            "Certains contrats sont hors tolerance par rapport au moteur WAP.",
            kind="error",
        )

    render_data_table(cmp_summary)

    render_section_header(
        "Lecture trade par trade",
        "Detail des positions, clotures et CMP apres chaque execution.",
        step="02",
        label="Details",
    )
    contract_options = cmp_summary["contract_code"].tolist()
    selected_contract = st.selectbox("Contrat a inspecter", options=contract_options)
    detail_view = cmp_detail.loc[cmp_detail["contract_code"] == selected_contract]
    render_data_table(detail_view)

render_footer()
