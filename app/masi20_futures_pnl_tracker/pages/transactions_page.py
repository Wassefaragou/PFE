import pandas as pd
import streamlit as st

from futures_pnl.config import STATUS_ALIASES, TRANSACTION_COLUMNS, TRANSACTION_STATUS_OPTIONS
from futures_pnl.importers import prepare_transaction_import
from futures_pnl.storage import save_transactions
from futures_pnl.ui import (
    append_dataframe_rows,
    format_choice_value,
    format_currency,
    get_empty_transactions,
    init_page,
    label_for,
    load_app_state,
    render_data_table,
    render_footer,
    render_form_group,
    render_hero,
    render_metric_cards,
    render_micro_note,
    render_section_header,
    render_sidebar_tools,
    show_issues,
)

init_page("Transactions")
state = load_app_state()
render_sidebar_tools(state)

transactions_raw = state["transactions_raw"]
transactions_validated = state["transactions_validated"]
transaction_issues = state["transaction_issues"]


def _transaction_option_label(row: pd.Series) -> str:
    execution_id = str(row.get("execution_id") or "-").strip() or "-"
    trade_date = str(row.get("trade_date") or "-").strip() or "-"
    trade_time = str(row.get("trade_time") or "-").strip() or "-"
    contract = str(row.get("contract") or "-").strip() or "-"
    side = str(row.get("side_lp") or "-").strip() or "-"
    quantity = "-" if pd.isna(row.get("quantity_lots")) else f"{float(row.get('quantity_lots')):,.0f}"
    price = "-" if pd.isna(row.get("price_points")) else f"{float(row.get('price_points')):,.4f}"
    return f"{execution_id} | {trade_date} {trade_time} | {contract} | {side} {quantity} @ {price}"


render_hero(
    "Transactions",
    "Saisie, import et edition du registre qui alimente directement le moteur officiel.",
)

confirmed_count = int(transactions_validated["is_confirmed"].fillna(False).sum()) if not transactions_validated.empty else 0
invalid_count = int(transactions_validated["is_valid_for_calc"].eq(False).sum()) if not transactions_validated.empty else 0
pending_count = int(transactions_validated["status"].eq("ATTENTE").fillna(False).sum()) if not transactions_validated.empty else 0
rejected_count = int(transactions_validated["status"].eq("REJETE").fillna(False).sum()) if not transactions_validated.empty else 0
excluded_count = (
    int(
        (
            transactions_validated["is_valid_for_calc"].fillna(False)
            & ~transactions_validated["is_official_for_calc"].fillna(False)
        ).sum()
    )
    if not transactions_validated.empty
    else 0
)
futures_long_notional = float(state["global_metrics"].get("open_notional_futures_long", 0.0))
futures_short_notional = float(state["global_metrics"].get("open_notional_futures_short", 0.0))

render_metric_cards(
    [
        {"label": "Transactions", "value": str(len(transactions_raw)), "glow": "gold"},
        {"label": "Confirmees", "value": str(confirmed_count), "glow": "green"},
        {"label": "En attente", "value": str(pending_count), "glow": "gold"},
        {"label": "Rejetees", "value": str(rejected_count), "glow": "red"},
        {"label": "Exclues du P&L", "value": str(excluded_count), "glow": "purple"},
        {"label": "Lignes invalides", "value": str(invalid_count), "glow": "red"},
        {"label": "Notionnel futures long", "value": format_currency(futures_long_notional), "glow": "green"},
        {"label": "Notionnel futures short", "value": format_currency(futures_short_notional), "glow": "purple"},
    ],
    columns=6,
)

valid_contracts = (
    state["contracts_ready"]
    .loc[state["contracts_ready"]["is_valid"].fillna(False), "contract_code"]
    .drop_duplicates()
    .tolist()
)
status_options = TRANSACTION_STATUS_OPTIONS.copy()

render_section_header(
    "Saisie et import",
    "Ajoutez les executions a la main ou importez un CSV au bon format.",
    step="01",
    label="Capture",
)

tab_manual, tab_import, tab_register = st.tabs(["Saisie manuelle", "Import CSV", "Registre"])

with tab_manual:
    with st.form("manual_transaction_form"):
        render_form_group("Identification", "ID unique, date, heure et contrat.")
        col1, col2, col3 = st.columns(3)
        execution_id = col1.text_input("ID execution", help="Identifiant unique de l'execution.")
        trade_date = col2.date_input("Date", help="Date de l'execution.")
        trade_time = col3.time_input("Heure", help="Heure de l'execution.")

        render_form_group("Execution", "Sens, quantite et prix du trade.")
        col4, col5, col6 = st.columns(3)
        contract = col4.selectbox("Contrat", options=valid_contracts if valid_contracts else [""], help="Le contrat doit exister dans le referentiel.")
        side_lp = col5.selectbox(
            "Sens",
            options=["BUY", "SELL"],
            format_func=lambda value: format_choice_value("side_lp", value),
            help="BUY = Achat, SELL = Vente.",
        )
        quantity_lots = col6.number_input("Quantite lots", min_value=0.0, step=1.0, help="Quantite strictement positive.")

        render_form_group("Contexte de trade", "Informations de contrepartie et statut d'execution.")
        col7, col8, col9 = st.columns(3)
        price_points = col7.number_input("Prix points", min_value=0.0, step=1.0, help="Prix du trade en points d'indice.")
        counterparty = col8.text_input("Contrepartie", placeholder="Nom de la contrepartie")
        counterparty_type = col9.text_input("Type contrepartie", placeholder="Broker, client, desk...")
        status = st.selectbox(
            "Statut",
            options=status_options,
            index=0 if status_options else None,
            format_func=lambda value: format_choice_value("status", value),
            help="Seul le statut CONFIRME alimente le P&L officiel.",
        )

        submitted = st.form_submit_button("Ajouter la transaction", width="stretch")

    if submitted:
        new_row = pd.DataFrame(
            [
                {
                    "execution_id": execution_id,
                    "trade_date": trade_date.strftime("%Y-%m-%d"),
                    "trade_time": trade_time.strftime("%H:%M:%S"),
                    "contract": contract,
                    "side_lp": side_lp,
                    "quantity_lots": quantity_lots,
                    "price_points": price_points,
                    "counterparty": counterparty,
                    "counterparty_type": counterparty_type,
                    "status": status,
                }
            ]
        )
        updated = append_dataframe_rows(transactions_raw, new_row, TRANSACTION_COLUMNS)
        save_transactions(updated)
        st.rerun()

with tab_import:
    render_micro_note(
        "Formats acceptes",
        "L'import supporte soit le registre interne, soit l'export broker de type Symbol / Side / Executed Size / Average Price / Execution ID / Transact Time. Le statut est force a CONFIRME pour ce format.",
        tone="info",
    )
    uploaded_file = st.file_uploader("Importer un CSV transactions", type=["csv"])
    if uploaded_file is not None:
        try:
            raw_imported = pd.read_csv(uploaded_file, sep=None, engine="python", dtype=str)
            imported, detected_format, ignored_columns = prepare_transaction_import(raw_imported)
        except Exception as exc:
            st.error(f"Import impossible: {exc}")
        else:
            missing_required = int(
                imported[
                    [
                        "execution_id",
                        "trade_date",
                        "trade_time",
                        "contract",
                        "side_lp",
                        "quantity_lots",
                        "price_points",
                    ]
                ].isna().any(axis=1).sum()
            )
            render_metric_cards(
                [
                    {"label": "Format detecte", "value": detected_format, "glow": "purple"},
                    {"label": "Lignes importees", "value": str(len(imported)), "glow": "gold"},
                    {"label": "Lignes a verifier", "value": str(missing_required), "glow": "red"},
                ],
                columns=3,
            )
            st.write("Apercu du registre importe")
            render_data_table(imported)
            if ignored_columns:
                st.caption(f"Colonnes ignorees: {', '.join(ignored_columns)}")
            if st.button("Ajouter les transactions importees", width="stretch"):
                updated = append_dataframe_rows(transactions_raw, imported, TRANSACTION_COLUMNS)
                save_transactions(updated)
                st.rerun()

with tab_register:
    editor_source = transactions_raw.copy()
    if editor_source.empty:
        editor_source = get_empty_transactions()
        
    for col in ["execution_id", "trade_date", "trade_time", "contract", "side_lp", "counterparty", "counterparty_type", "status"]:
        if col in editor_source.columns:
            editor_source[col] = editor_source[col].fillna("").astype(str)
    if "status" in editor_source.columns:
        editor_source["status"] = editor_source["status"].str.upper().map(
            lambda value: STATUS_ALIASES.get(value, value) if value else ""
        )
            
    for col in ["quantity_lots", "price_points"]:
        if col in editor_source.columns:
            editor_source[col] = pd.to_numeric(editor_source[col], errors="coerce")

    edited_transactions = st.data_editor(
        editor_source,
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        column_config={
            "execution_id": st.column_config.TextColumn(label_for("execution_id")),
            "trade_date": st.column_config.TextColumn(label_for("trade_date"), help="Format YYYY-MM-DD"),
            "trade_time": st.column_config.TextColumn(label_for("trade_time"), help="Format HH:MM:SS"),
            "contract": st.column_config.TextColumn(label_for("contract")),
            "side_lp": st.column_config.TextColumn(label_for("side_lp")),
            "quantity_lots": st.column_config.NumberColumn(label_for("quantity_lots"), format="%.2f"),
            "price_points": st.column_config.NumberColumn(label_for("price_points"), format="%.4f"),
            "counterparty": st.column_config.TextColumn(label_for("counterparty")),
            "counterparty_type": st.column_config.TextColumn(label_for("counterparty_type")),
            "status": st.column_config.SelectboxColumn(
                label_for("status"),
                options=status_options,
                required=False,
            ),
        },
    )
    working_transactions = edited_transactions.reset_index(drop=True).copy()
    if not working_transactions.empty:
        working_transactions["row_id"] = working_transactions.index
        row_options = working_transactions["row_id"].tolist()
        selected_rows = st.multiselect(
            "Transactions a supprimer",
            options=row_options,
            format_func=lambda row_id: _transaction_option_label(
                working_transactions.loc[working_transactions["row_id"] == row_id].iloc[0]
            ),
            help="Selectionnez une ou plusieurs lignes a supprimer du registre.",
        )
    else:
        selected_rows = []

    action_col1, action_col2 = st.columns(2)
    with action_col1:
        if st.button("Enregistrer le registre", width="stretch"):
            save_transactions(edited_transactions)
            st.rerun()
    with action_col2:
        if st.button("Supprimer la selection", width="stretch"):
            if not selected_rows:
                st.warning("Selectionnez au moins une transaction a supprimer.")
            else:
                remaining_transactions = working_transactions.loc[
                    ~working_transactions["row_id"].isin(selected_rows),
                    TRANSACTION_COLUMNS,
                ].reset_index(drop=True)
                save_transactions(remaining_transactions)
                st.rerun()

render_section_header(
    "Controle des transactions",
    "Le moteur officiel filtre maintenant les transactions par statut en plus des controles de validite technique.",
    step="02",
    label="Controles",
)
show_issues("Validation des transactions", transaction_issues)

status_order = pd.CategoricalDtype(["CONFIRME", "ATTENTE", "REJETE"], ordered=True)
transactions_by_status = transactions_validated.copy()
if not transactions_by_status.empty and "status" in transactions_by_status.columns:
    transactions_by_status["status"] = transactions_by_status["status"].astype("object")
    transactions_by_status["status_sort"] = transactions_by_status["status"].astype(status_order)
    transactions_by_status = transactions_by_status.sort_values(
        ["status_sort", "contract", "trade_date", "trade_time", "execution_id"],
        na_position="last",
    ).drop(columns=["status_sort"])

status_contract_view = pd.DataFrame()
if not transactions_by_status.empty:
    status_contract_view = (
        transactions_by_status.groupby(["status", "contract"], dropna=False, as_index=False)
        .agg(
            nb_transactions=("execution_id", "count"),
            quantite_totale=("quantity_lots", "sum"),
            inclus_pnl_officiel=("is_official_for_calc", "sum"),
        )
    )
    status_contract_view["status_sort"] = status_contract_view["status"].astype(status_order)
    status_contract_view = status_contract_view.sort_values(
        ["status_sort", "contract"], na_position="last"
    ).drop(columns=["status_sort"])

display_columns = [
    "execution_id",
    "trade_date",
    "trade_time",
    "contract",
    "side_lp",
    "quantity_lots",
    "price_points",
    "status",
    "is_confirmed",
    "is_valid_for_calc",
    "is_official_for_calc",
]
if transactions_validated.empty:
    st.info("Aucune transaction disponible.")
else:
    synthese_tab, detail_tab = st.tabs(["Contrats par statut", "Transactions triees par statut"])
    with synthese_tab:
        if status_contract_view.empty:
            st.info("Aucune synthese par statut disponible.")
        else:
            render_data_table(
                status_contract_view,
                label_overrides={
                    "status": "Statut",
                    "contract": "Contrat",
                    "nb_transactions": "Nb transactions",
                    "quantite_totale": "Quantite totale",
                    "inclus_pnl_officiel": "Nb inclus P&L officiel",
                },
            )
    with detail_tab:
        render_data_table(transactions_by_status, display_columns)

render_footer()
