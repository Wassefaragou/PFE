import numpy as np
import pandas as pd

from .config import (
    CONFIRMED_STATUSES,
    CONTRACT_COLUMNS,
    OFFICIAL_PNL_STATUSES,
    STATUS_ALIASES,
    TRANSACTION_COLUMNS,
)
from .contracts import clear_contract_overrides, enrich_contract_reference


def _blank_to_na(value):
    if pd.isna(value):
        return pd.NA
    if isinstance(value, str) and value.strip() == "":
        return pd.NA
    return value


def _normalize_text(series: pd.Series) -> pd.Series:
    return series.map(_blank_to_na).astype("object")


def _normalize_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y", "oui"}


def _issues_to_dataframe(issues: list[dict]) -> pd.DataFrame:
    if not issues:
        return pd.DataFrame(columns=["severity", "code", "row", "entity", "message"])
    return pd.DataFrame(issues)


def validate_contracts(contracts_df: pd.DataFrame, settings: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    contracts = contracts_df.copy() if contracts_df is not None else pd.DataFrame(columns=CONTRACT_COLUMNS)
    for column in CONTRACT_COLUMNS:
        if column not in contracts.columns:
            contracts[column] = pd.NA
    contracts = clear_contract_overrides(enrich_contract_reference(contracts[CONTRACT_COLUMNS].copy()))
    contracts["row_number"] = np.arange(1, len(contracts) + 1)
    contracts["comments"] = _normalize_text(contracts["comments"]).map(
        lambda value: str(value).strip() if pd.notna(value) else pd.NA
    )
    contracts["expiry_date_parsed"] = pd.to_datetime(contracts["expiry_date"], errors="coerce").dt.normalize()
    contracts["expiry_date"] = contracts["expiry_date_parsed"].dt.strftime("%Y-%m-%d")
    contracts.loc[contracts["expiry_date_parsed"].isna(), "expiry_date"] = pd.NA

    numeric_columns = ["initial_margin_per_lot", "settlement_price_points"]
    for column in numeric_columns:
        contracts[column] = pd.to_numeric(contracts[column], errors="coerce")
    contracts["is_active_lp"] = contracts["is_active_lp"].map(_normalize_bool)

    issues: list[dict] = []
    for record in contracts.itertuples():
        if pd.isna(record.expiry_date_parsed):
            issues.append(
                {
                    "severity": "error",
                    "code": "invalid_expiry_date",
                    "row": record.row_number,
                    "entity": record.contract_code if pd.notna(record.contract_code) else "",
                    "message": "expiry_date doit etre une date valide.",
                }
            )
        if pd.notna(record.settlement_price_points) and record.settlement_price_points <= 0:
            issues.append(
                {
                    "severity": "error",
                    "code": "invalid_settlement_price",
                    "row": record.row_number,
                    "entity": record.contract_code if pd.notna(record.contract_code) else "",
                    "message": "settlement_price_points doit etre strictement positif.",
                }
            )
    duplicate_codes = (
        contracts["contract_code"]
        .dropna()
        .loc[lambda series: series.duplicated(keep=False)]
        .unique()
        .tolist()
    )
    for duplicate_code in duplicate_codes:
        duplicate_rows = contracts.loc[contracts["contract_code"] == duplicate_code, "row_number"].tolist()
        for row_number in duplicate_rows:
            issues.append(
                {
                    "severity": "error",
                    "code": "duplicate_contract_code",
                    "row": row_number,
                    "entity": duplicate_code,
                    "message": "contract_code doit etre unique.",
                }
            )

    contracts["effective_tick_value"] = pd.to_numeric(settings.get("default_tick_value"), errors="coerce")
    contracts["effective_initial_margin_per_lot"] = pd.to_numeric(contracts["initial_margin_per_lot"], errors="coerce")
    legacy_default_margin = pd.to_numeric(settings.get("default_initial_margin_per_lot"), errors="coerce")
    if pd.notna(legacy_default_margin):
        contracts["effective_initial_margin_per_lot"] = contracts["effective_initial_margin_per_lot"].fillna(
            float(legacy_default_margin)
        )
    contracts["effective_position_limit_per_contract"] = pd.to_numeric(
        settings.get("default_position_limit_per_contract"), errors="coerce"
    )

    for record in contracts.itertuples():
        if pd.isna(record.effective_tick_value) or record.effective_tick_value <= 0:
            issues.append(
                {
                    "severity": "error",
                    "code": "missing_effective_tick_value",
                    "row": record.row_number,
                    "entity": record.contract_code if pd.notna(record.contract_code) else "",
                    "message": "Aucun tick global valide dans les parametres.",
                }
            )
        if pd.isna(record.effective_initial_margin_per_lot) or record.effective_initial_margin_per_lot <= 0:
            issues.append(
                {
                    "severity": "error",
                    "code": "invalid_initial_margin_per_lot",
                    "row": record.row_number,
                    "entity": record.contract_code if pd.notna(record.contract_code) else "",
                    "message": "initial_margin_per_lot doit etre renseigne et strictement positif pour chaque contrat.",
                }
            )

    issue_rows = {issue["row"] for issue in issues if issue["severity"] == "error"}
    contracts["is_valid"] = ~contracts["row_number"].isin(issue_rows)
    return contracts, _issues_to_dataframe(issues)


def validate_transactions(
    transactions_df: pd.DataFrame, contracts_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    transactions = (
        transactions_df.copy() if transactions_df is not None else pd.DataFrame(columns=TRANSACTION_COLUMNS)
    )
    for column in TRANSACTION_COLUMNS:
        if column not in transactions.columns:
            transactions[column] = pd.NA
    transactions = transactions[TRANSACTION_COLUMNS].copy()
    transactions["row_number"] = np.arange(1, len(transactions) + 1)

    for column in ["execution_id", "contract", "side_lp", "counterparty", "counterparty_type", "status"]:
        transactions[column] = _normalize_text(transactions[column]).map(
            lambda value: str(value).strip() if pd.notna(value) else pd.NA
        )

    transactions["execution_id"] = transactions["execution_id"].map(
        lambda value: value.upper() if pd.notna(value) else pd.NA
    )
    transactions["contract"] = transactions["contract"].map(
        lambda value: value.upper() if pd.notna(value) else pd.NA
    )
    transactions["side_lp"] = transactions["side_lp"].map(
        lambda value: value.upper() if pd.notna(value) else pd.NA
    )
    transactions["status"] = transactions["status"].map(
        lambda value: value.upper() if pd.notna(value) else pd.NA
    )
    transactions["status"] = transactions["status"].map(
        lambda value: STATUS_ALIASES.get(value, value) if pd.notna(value) else pd.NA
    )

    transactions["trade_date_parsed"] = pd.to_datetime(transactions["trade_date"], errors="coerce").dt.normalize()
    transactions["trade_time_parsed"] = pd.to_datetime(transactions["trade_time"], format="%H:%M:%S", errors="coerce")
    transactions["trade_date"] = transactions["trade_date_parsed"].dt.strftime("%Y-%m-%d")
    transactions["trade_time"] = transactions["trade_time_parsed"].dt.strftime("%H:%M:%S")
    transactions.loc[transactions["trade_date_parsed"].isna(), "trade_date"] = pd.NA
    transactions.loc[transactions["trade_time_parsed"].isna(), "trade_time"] = pd.NA
    transactions["quantity_lots"] = pd.to_numeric(transactions["quantity_lots"], errors="coerce")
    transactions["price_points"] = pd.to_numeric(transactions["price_points"], errors="coerce")

    valid_contracts = (
        contracts_df.loc[contracts_df["is_valid"].fillna(False)]
        .drop_duplicates("contract_code")
        .set_index("contract_code")
    )
    transactions["contract_exists"] = transactions["contract"].isin(valid_contracts.index)
    transactions["tick_value_contract"] = transactions["contract"].map(valid_contracts["effective_tick_value"])
    transactions["notional_mad"] = (
        transactions["quantity_lots"] * transactions["price_points"] * transactions["tick_value_contract"]
    )

    trade_time_offsets = transactions["trade_time_parsed"].dt.hour.fillna(0) * 3600
    trade_time_offsets += transactions["trade_time_parsed"].dt.minute.fillna(0) * 60
    trade_time_offsets += transactions["trade_time_parsed"].dt.second.fillna(0)
    transactions["trade_datetime"] = transactions["trade_date_parsed"] + pd.to_timedelta(
        trade_time_offsets, unit="s"
    )

    sort_columns = ["trade_datetime", "execution_id", "row_number"]
    transactions = transactions.sort_values(sort_columns, na_position="last").reset_index(drop=True)
    duplicate_datetime_rank = transactions.groupby("trade_datetime", dropna=False).cumcount()
    transactions["chrono_key"] = transactions["trade_datetime"] + pd.to_timedelta(
        duplicate_datetime_rank, unit="us"
    )

    issues: list[dict] = []
    for record in transactions.itertuples():
        if pd.isna(record.execution_id):
            issues.append(
                {
                    "severity": "error",
                    "code": "missing_execution_id",
                    "row": record.row_number,
                    "entity": "",
                    "message": "execution_id est obligatoire.",
                }
            )
        if pd.isna(record.trade_date_parsed):
            issues.append(
                {
                    "severity": "error",
                    "code": "invalid_trade_date",
                    "row": record.row_number,
                    "entity": record.execution_id if pd.notna(record.execution_id) else "",
                    "message": "trade_date doit etre une date valide.",
                }
            )
        if pd.isna(record.trade_time_parsed):
            issues.append(
                {
                    "severity": "error",
                    "code": "invalid_trade_time",
                    "row": record.row_number,
                    "entity": record.execution_id if pd.notna(record.execution_id) else "",
                    "message": "trade_time doit etre au format HH:MM:SS.",
                }
            )
        if pd.isna(record.contract) or not record.contract_exists:
            issues.append(
                {
                    "severity": "error",
                    "code": "unknown_contract",
                    "row": record.row_number,
                    "entity": record.execution_id if pd.notna(record.execution_id) else "",
                    "message": "Le contrat reference doit exister dans le referentiel valide.",
                }
            )
        if pd.isna(record.side_lp) or record.side_lp not in {"BUY", "SELL"}:
            issues.append(
                {
                    "severity": "error",
                    "code": "invalid_side_lp",
                    "row": record.row_number,
                    "entity": record.execution_id if pd.notna(record.execution_id) else "",
                    "message": "side_lp doit etre BUY ou SELL.",
                }
            )
        if pd.isna(record.quantity_lots) or record.quantity_lots <= 0:
            issues.append(
                {
                    "severity": "error",
                    "code": "invalid_quantity_lots",
                    "row": record.row_number,
                    "entity": record.execution_id if pd.notna(record.execution_id) else "",
                    "message": "quantity_lots doit etre strictement positif.",
                }
            )
        if pd.isna(record.price_points) or record.price_points <= 0:
            issues.append(
                {
                    "severity": "error",
                    "code": "invalid_price_points",
                    "row": record.row_number,
                    "entity": record.execution_id if pd.notna(record.execution_id) else "",
                    "message": "price_points doit etre strictement positif.",
                }
            )
        if pd.notna(record.contract) and pd.isna(record.tick_value_contract):
            issues.append(
                {
                    "severity": "error",
                    "code": "missing_contract_tick_value",
                    "row": record.row_number,
                    "entity": record.execution_id if pd.notna(record.execution_id) else "",
                    "message": "Impossible de calculer le notionnel sans tick value contrat.",
                }
            )

    duplicate_ids = (
        transactions["execution_id"]
        .dropna()
        .loc[lambda series: series.duplicated(keep=False)]
        .unique()
        .tolist()
    )
    for duplicate_id in duplicate_ids:
        duplicate_rows = transactions.loc[transactions["execution_id"] == duplicate_id, "row_number"].tolist()
        for row_number in duplicate_rows:
            issues.append(
                {
                    "severity": "error",
                    "code": "duplicate_execution_id",
                    "row": row_number,
                    "entity": duplicate_id,
                    "message": "execution_id doit etre unique.",
                }
            )

    issue_rows = {issue["row"] for issue in issues if issue["severity"] == "error"}
    transactions["is_valid_for_calc"] = ~transactions["row_number"].isin(issue_rows)
    transactions["is_confirmed"] = transactions["status"].isin(CONFIRMED_STATUSES)
    transactions["is_official_for_calc"] = (
        transactions["is_valid_for_calc"] & transactions["status"].isin(OFFICIAL_PNL_STATUSES)
    )
    return transactions, _issues_to_dataframe(issues)
