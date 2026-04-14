import re

import pandas as pd

from .config import TRANSACTION_COLUMNS
from .contracts import normalize_contract_code


def _normalize_header(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()


def _empty_series(length: int) -> pd.Series:
    return pd.Series([pd.NA] * length, dtype="object")


def _column_lookup(dataframe: pd.DataFrame) -> dict[str, str]:
    return {_normalize_header(column): column for column in dataframe.columns}


def _find_column(column_lookup: dict[str, str], *candidates: str) -> str | None:
    for candidate in candidates:
        normalized = _normalize_header(candidate)
        if normalized in column_lookup:
            return column_lookup[normalized]
    return None


def _text_series(dataframe: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None or column not in dataframe.columns:
        return _empty_series(len(dataframe))

    return dataframe[column].map(
        lambda value: pd.NA if pd.isna(value) or not str(value).strip() else str(value).strip()
    ).astype("object")


def _numeric_series(dataframe: pd.DataFrame, column: str | None) -> pd.Series:
    text = _text_series(dataframe, column).astype("string")
    cleaned = (
        text.str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("\u00a0", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _first_non_empty(dataframe: pd.DataFrame, columns: list[str | None]) -> pd.Series:
    values = _empty_series(len(dataframe))
    for column in columns:
        candidate = _text_series(dataframe, column)
        fill_mask = values.isna() & candidate.notna()
        values.loc[fill_mask] = candidate.loc[fill_mask]
    return values


def _parse_datetime_column(timestamp_series: pd.Series) -> tuple[pd.Series, pd.Series]:
    raw = timestamp_series.astype("string").fillna("").str.strip()
    cleaned = raw.str.replace(r"(?<=\d{4})-(?=\d{2}:\d{2}:\d{2}$)", " ", regex=True)
    parsed = pd.to_datetime(cleaned, dayfirst=True, errors="coerce")

    trade_date = parsed.dt.strftime("%Y-%m-%d").astype("object")
    trade_time = parsed.dt.strftime("%H:%M:%S").astype("object")
    trade_date.loc[parsed.isna()] = pd.NA
    trade_time.loc[parsed.isna()] = pd.NA
    return trade_date, trade_time


def _parse_date_and_time_columns(
    date_series: pd.Series,
    time_series: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    parsed_date = pd.to_datetime(date_series.astype("string").fillna("").str.strip(), dayfirst=True, errors="coerce")
    parsed_time = pd.to_datetime(time_series.astype("string").fillna("").str.strip(), errors="coerce")

    trade_date = parsed_date.dt.strftime("%Y-%m-%d").astype("object")
    trade_time = parsed_time.dt.strftime("%H:%M:%S").astype("object")
    trade_date.loc[parsed_date.isna()] = pd.NA
    trade_time.loc[parsed_time.isna()] = pd.NA
    return trade_date, trade_time


def _detect_internal_mapping(dataframe: pd.DataFrame) -> dict[str, str] | None:
    columns = _column_lookup(dataframe)
    mapping: dict[str, str] = {}
    for field in TRANSACTION_COLUMNS:
        mapping[field] = _find_column(columns, field, field.replace("_", " "))
    return mapping if all(mapping.values()) else None


def _format_missing_columns_message() -> str:
    return (
        "Format CSV non reconnu. Colonnes attendues: soit le registre interne "
        f"{', '.join(TRANSACTION_COLUMNS)}, soit un export broker avec au minimum "
        "'Symbol', 'Side', 'Executed Size', 'Average Price', 'Execution ID' et 'Transact Time'."
    )


def prepare_transaction_import(raw_dataframe: pd.DataFrame) -> tuple[pd.DataFrame, str, list[str]]:
    dataframe = raw_dataframe.copy()
    if dataframe.empty:
        return pd.DataFrame(columns=TRANSACTION_COLUMNS), "Fichier vide", []

    internal_mapping = _detect_internal_mapping(dataframe)
    if internal_mapping is not None:
        imported = pd.DataFrame(
            {field: dataframe[internal_mapping[field]] for field in TRANSACTION_COLUMNS}
        )
        imported["execution_id"] = _text_series(imported, "execution_id").map(
            lambda value: value.upper() if pd.notna(value) else pd.NA
        )
        imported["contract"] = _text_series(imported, "contract").map(normalize_contract_code)
        imported["side_lp"] = _text_series(imported, "side_lp").map(
            lambda value: value.upper() if pd.notna(value) else pd.NA
        )
        imported["counterparty"] = _text_series(imported, "counterparty")
        imported["counterparty_type"] = _text_series(imported, "counterparty_type")
        imported["status"] = _text_series(imported, "status").map(
            lambda value: value.upper() if pd.notna(value) else pd.NA
        )
        imported["quantity_lots"] = pd.to_numeric(imported["quantity_lots"], errors="coerce")
        imported["price_points"] = pd.to_numeric(imported["price_points"], errors="coerce")
        return imported[TRANSACTION_COLUMNS], "Registre interne", []

    columns = _column_lookup(dataframe)
    symbol_col = _find_column(columns, "symbol")
    side_col = _find_column(columns, "side")
    executed_size_col = _find_column(columns, "executed size", "quantity", "qty", "lots")
    average_price_col = _find_column(columns, "average price", "avg price", "executed value", "price")
    execution_id_col = _find_column(columns, "execution id")
    client_order_id_col = _find_column(columns, "client order id")
    trade_report_id_col = _find_column(columns, "trade report id")
    transact_time_col = _find_column(columns, "transact time", "execution time", "transaction time")
    trade_date_col = _find_column(columns, "trade date", "execution date", "date")
    trade_time_col = _find_column(columns, "trade time", "time")
    contra_firm_col = _find_column(columns, "contra firm", "counterparty", "contra broker")
    client_id_col = _find_column(columns, "client id")
    client_type_col = _find_column(columns, "client type")
    account_type_col = _find_column(columns, "account type")
    order_book_col = _find_column(columns, "order book")

    if not all([symbol_col, side_col, executed_size_col, average_price_col]) or not any(
        [execution_id_col, client_order_id_col, trade_report_id_col]
    ) or not any([transact_time_col, trade_date_col]):
        raise ValueError(_format_missing_columns_message())

    if transact_time_col is not None:
        trade_date, trade_time = _parse_datetime_column(_text_series(dataframe, transact_time_col))
    else:
        trade_date, trade_time = _parse_date_and_time_columns(
            _text_series(dataframe, trade_date_col),
            _text_series(dataframe, trade_time_col),
        )

    imported = pd.DataFrame(
        {
            "execution_id": _first_non_empty(
                dataframe,
                [execution_id_col, client_order_id_col, trade_report_id_col],
            ).map(lambda value: value.upper() if pd.notna(value) else pd.NA),
            "trade_date": trade_date,
            "trade_time": trade_time,
            "contract": _text_series(dataframe, symbol_col).map(normalize_contract_code),
            "side_lp": _text_series(dataframe, side_col).map(
                lambda value: value.upper() if pd.notna(value) else pd.NA
            ),
            "quantity_lots": _numeric_series(dataframe, executed_size_col),
            "price_points": _numeric_series(dataframe, average_price_col),
            "counterparty": _first_non_empty(dataframe, [contra_firm_col, client_id_col]),
            "counterparty_type": _first_non_empty(dataframe, [client_type_col, account_type_col, order_book_col]),
            "status": "CONFIRME",
        }
    )

    used_columns = {
        column
        for column in [
            symbol_col,
            side_col,
            executed_size_col,
            average_price_col,
            execution_id_col,
            client_order_id_col,
            trade_report_id_col,
            transact_time_col,
            trade_date_col,
            trade_time_col,
            contra_firm_col,
            client_id_col,
            client_type_col,
            account_type_col,
            order_book_col,
        ]
        if column is not None
    }
    ignored_columns = [column for column in dataframe.columns if column not in used_columns]
    return imported[TRANSACTION_COLUMNS], "Export broker", ignored_columns
