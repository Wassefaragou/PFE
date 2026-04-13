import pandas as pd
import streamlit as st

from futures_pnl.contracts import clear_contract_overrides, enrich_contract_reference
from futures_pnl.storage import (
    load_daily_prices,
    load_transactions,
    save_contracts,
    save_daily_prices,
    save_transactions,
)
from futures_pnl.ui import (
    get_empty_contracts,
    init_page,
    label_for,
    load_app_state,
    render_data_table,
    render_footer,
    render_micro_note,
    render_hero,
    render_metric_cards,
    render_section_header,
    render_sidebar_tools,
    show_issues,
)

init_page("Referentiel Contrats")
state = load_app_state()
render_sidebar_tools(state)

contracts_raw = state["contracts_raw"]
contracts_priced = state["contracts_priced"]
contract_issues = state["contract_issues"]


def _sync_generated_tickers(previous_contracts, next_contracts) -> None:
    previous = enrich_contract_reference(previous_contracts)
    current = enrich_contract_reference(next_contracts, force_contract_code=True)

    mapping = (
        previous.loc[previous["expiry_date"].notna(), ["expiry_date", "contract_code"]]
        .rename(columns={"contract_code": "old_code"})
        .merge(
            current.loc[current["expiry_date"].notna(), ["expiry_date", "contract_code"]].rename(
                columns={"contract_code": "new_code"}
            ),
            on="expiry_date",
            how="inner",
        )
        .dropna(subset=["old_code", "new_code"])
        .drop_duplicates(subset=["expiry_date", "old_code", "new_code"])
    )
    mapping = mapping.loc[mapping["old_code"] != mapping["new_code"]]
    if mapping.empty:
        return

    code_map = dict(zip(mapping["old_code"], mapping["new_code"]))

    transactions = load_transactions()
    if not transactions.empty and "contract" in transactions.columns:
        transactions["contract"] = transactions["contract"].replace(code_map)
        save_transactions(transactions)

    daily_prices = load_daily_prices()
    if not daily_prices.empty and "contract_code" in daily_prices.columns:
        daily_prices["contract_code"] = daily_prices["contract_code"].replace(code_map)
        save_daily_prices(daily_prices)

render_hero(
    "Referentiel contrats",
    "Ajoutez, modifiez et supprimez librement les contrats suivis par l'application.",
    badges=[("Univers dynamique", ""), ("Prix", "purple"), ("Validation", "green")],
)

active_count = int(contracts_priced["is_active_lp"].fillna(False).sum()) if not contracts_priced.empty else 0
with_settlement = int(contracts_priced["settlement_price_points"].notna().sum()) if not contracts_priced.empty else 0
invalid_count = int(contracts_priced["is_valid"].eq(False).sum()) if not contracts_priced.empty else 0

render_metric_cards(
    [
        {"label": "Contrats", "value": str(len(contracts_raw)), "glow": "gold"},
        {"label": "LP actifs", "value": str(active_count), "glow": "green"},
        {"label": "Avec settlement", "value": str(with_settlement), "glow": "blue"},
        {"label": "Lignes invalides", "value": str(invalid_count), "glow": "red"},
    ],
    columns=4,
)

render_section_header(
    "Modifier le referentiel",
    step="01",
    label="Contrats",
)

editor_source = clear_contract_overrides(enrich_contract_reference(contracts_raw, force_contract_code=True))
if editor_source.empty:
    editor_source = clear_contract_overrides(enrich_contract_reference(get_empty_contracts(), force_contract_code=True))

editor_columns = [
    "expiry_date",
    "initial_margin_per_lot",
    "settlement_price_points",
    "is_active_lp",
    "comments",
]
editor_input = editor_source[editor_columns].copy()
editor_input["expiry_date"] = editor_input["expiry_date"].fillna("").astype(str)
editor_input["comments"] = editor_input["comments"].fillna("").astype(str)
editor_input["is_active_lp"] = editor_input["is_active_lp"].fillna(False).astype(bool)
editor_input["initial_margin_per_lot"] = pd.to_numeric(editor_input["initial_margin_per_lot"], errors="coerce")
editor_input["settlement_price_points"] = pd.to_numeric(editor_input["settlement_price_points"], errors="coerce")

edited_contracts = st.data_editor(
    editor_input,
    num_rows="dynamic",
    width="stretch",
    hide_index=True,
    column_config={
        "expiry_date": st.column_config.TextColumn(label_for("expiry_date"), help="Format YYYY-MM-DD"),
        "initial_margin_per_lot": st.column_config.NumberColumn(
            label_for("initial_margin_per_lot"),
            format="%.2f",
            help="Marge initiale requise par lot pour ce contrat.",
        ),
        "settlement_price_points": st.column_config.NumberColumn(
            label_for("settlement_price_points"),
            format="%.4f",
            help="Prix settlement / MtM du jour. Vous pouvez le modifier a chaque ouverture de l'application.",
        ),
        "is_active_lp": st.column_config.CheckboxColumn(label_for("is_active_lp")),
        "comments": st.column_config.TextColumn(label_for("comments")),
    },
)

editor_preview = clear_contract_overrides(enrich_contract_reference(edited_contracts, force_contract_code=True))
if not editor_preview.empty:
    render_micro_note(
        "Apercu ticker",
        "Le ticker est recalcule a partir de l'echeance courante. L'enregistrement applique cette valeur au registre.",
        tone="warning",
    )
    render_data_table(editor_preview, ["expiry_date", "contract_code"])

if st.button("Enregistrer le referentiel", width="stretch"):
    _sync_generated_tickers(contracts_raw, editor_preview)
    save_contracts(editor_preview)
    st.rerun()

render_section_header(
    "Controle et pricing",
    "Validation des lignes et apercu des prix theoriques et MtM.",
    step="02",
    label="Controles",
)
show_issues("Validation des contrats", contract_issues)

display_columns = [
    "contract_code",
    "expiry_date",
    "initial_margin_per_lot",
    "effective_tick_value",
    "theoretical_price",
    "mtm_price",
    "mtm_source",
    "days_to_expiry",
    "expiry_alert",
]
if contracts_priced.empty:
    st.info("Aucun contrat disponible.")
else:
    render_data_table(contracts_priced, display_columns)

render_footer()
