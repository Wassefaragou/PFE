import pandas as pd
import streamlit as st

from futures_pnl.config import current_valuation_date_str
from futures_pnl.contracts import clear_contract_overrides, enrich_contract_reference, upcoming_contract_schedule
from futures_pnl.storage import (
    load_daily_prices,
    load_transactions,
    save_contracts,
    save_daily_prices,
    save_transactions,
)
from futures_pnl.ui import (
    format_expiry_month,
    get_empty_contracts,
    init_page,
    label_for,
    load_app_state,
    render_data_table,
    render_footer,
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
contracts_ready = state["contracts_ready"]
contract_issues = state["contract_issues"]


@st.cache_resource(show_spinner=False)
def _server_reference_date() -> object:
    reference = pd.to_datetime(current_valuation_date_str(), errors="coerce")
    if pd.isna(reference):
        return None
    return pd.Timestamp(reference).date()


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


def _expiry_selection_label(value: object) -> str:
    formatted_value = format_expiry_month(value)
    return "" if formatted_value == "-" else formatted_value


def _build_expiry_option_map(expiry_values: list[str]) -> dict[str, str]:
    option_map = {"": ""}
    for raw_value in expiry_values:
        text = str(raw_value).strip()
        if not text:
            continue
        option_map[_expiry_selection_label(text)] = text
    return option_map

render_hero(
    "Referentiel contrats",
    "Ajoutez, modifiez et supprimez les contrats suivis par le tracker.",
)

with_settlement = int(contracts_ready["settlement_price_points"].notna().sum()) if not contracts_ready.empty else 0
invalid_count = int(contracts_ready["is_valid"].eq(False).sum()) if not contracts_ready.empty else 0

render_metric_cards(
    [
        {"label": "Contrats", "value": str(len(contracts_raw)), "glow": "gold"},
        {"label": "Avec cours", "value": str(with_settlement), "glow": "blue"},
        {"label": "Lignes invalides", "value": str(invalid_count), "glow": "red"},
    ],
    columns=3,
)

render_section_header(
    "Modifier le referentiel",
    step="01",
    label="Contrats",
)

reference_date = _server_reference_date()

editor_source = clear_contract_overrides(enrich_contract_reference(contracts_raw, force_contract_code=True))
if editor_source.empty:
    editor_source = clear_contract_overrides(enrich_contract_reference(get_empty_contracts(), force_contract_code=True))

editor_columns = [
    "expiry_date",
    "initial_margin_per_lot",
    "settlement_price_points",
    "comments",
]
editor_input = editor_source[editor_columns].copy()
editor_input["expiry_date"] = editor_input["expiry_date"].map(_expiry_selection_label)
editor_input["comments"] = editor_input["comments"].fillna("").astype(str)
editor_input["initial_margin_per_lot"] = pd.to_numeric(editor_input["initial_margin_per_lot"], errors="coerce")
editor_input["settlement_price_points"] = pd.to_numeric(editor_input["settlement_price_points"], errors="coerce")

upcoming_contracts = upcoming_contract_schedule(reference_date=reference_date, contract_count=5)
upcoming_expiry_dates = [contract["expiry_date"] for contract in upcoming_contracts]
existing_expiry_dates = [
    value
    for value in editor_source["expiry_date"].tolist()
    if isinstance(value, str) and value.strip() and value not in upcoming_expiry_dates
]
expiry_option_map = _build_expiry_option_map(upcoming_expiry_dates + sorted(set(existing_expiry_dates)))
expiry_options = list(expiry_option_map.keys())

if upcoming_contracts:
    upcoming_contract_cards = []
    palette = ["blue", "purple", "gold", "green", "red"]
    for index, contract in enumerate(upcoming_contracts):
        upcoming_contract_cards.append(
            {
                "label": _expiry_selection_label(contract["expiry_date"]),
                "value": contract["contract_code"],
                "glow": palette[index % len(palette)],
            }
        )
    render_metric_cards(upcoming_contract_cards, columns=5)

edited_contracts = st.data_editor(
    editor_input,
    num_rows="dynamic",
    width="stretch",
    hide_index=True,
    column_config={
        "expiry_date": st.column_config.SelectboxColumn(
            label_for("expiry_date"),
            help="Selectionnez une des 5 prochaines echeances trimestrielles proposees.",
            options=expiry_options,
            required=False,
        ),
        "initial_margin_per_lot": st.column_config.NumberColumn(
            label_for("initial_margin_per_lot"),
            format="%.2f",
            help="Marge initiale requise par lot pour ce contrat.",
        ),
        "settlement_price_points": st.column_config.NumberColumn(
            label_for("settlement_price_points"),
            format="%.4f",
            help="Cours de valorisation utilise pour le suivi quotidien.",
        ),
        "comments": st.column_config.TextColumn(label_for("comments")),
    },
)

editor_preview_input = edited_contracts.copy()
editor_preview_input["expiry_date"] = editor_preview_input["expiry_date"].map(
    lambda value: expiry_option_map.get(str(value).strip(), "")
)
editor_preview = clear_contract_overrides(enrich_contract_reference(editor_preview_input, force_contract_code=True))

if st.button("Enregistrer le referentiel", width="stretch"):
    _sync_generated_tickers(contracts_raw, editor_preview)
    save_contracts(editor_preview)
    st.rerun()

render_section_header(
    "Controle du referentiel",
    "Validation des lignes et apercu du cours retenu par contrat.",
    step="02",
    label="Controles",
)
show_issues("Validation des contrats", contract_issues)

display_columns = [
    "contract_code",
    "expiry_date",
    "initial_margin_per_lot",
    "settlement_price_points",
    "effective_tick_value",
    "expiry_alert",
]
if contracts_ready.empty:
    st.info("Aucun contrat disponible.")
else:
    render_data_table(contracts_ready, display_columns)

render_footer()
