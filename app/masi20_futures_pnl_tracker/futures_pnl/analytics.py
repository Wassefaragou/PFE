import numpy as np
import pandas as pd


MARGIN_CALL_COLUMNS = [
    "previous_price_date",
    "previous_mtm_price",
    "daily_price_delta_points",
    "previous_open_position",
    "new_position_today",
    "new_position_cmp",
    "daily_mtm_mad",
    "daily_latent_mad",
    "margin_call_mad",
]


def _safe_weighted_average(dataframe: pd.DataFrame) -> float:
    if dataframe.empty:
        return 0.0
    denominator = dataframe["quantity_lots"].sum()
    if denominator == 0:
        return 0.0
    return float((dataframe["quantity_lots"] * dataframe["price_points"]).sum() / denominator)


def _peak_abs_position_from_trades(dataframe: pd.DataFrame) -> float:
    if dataframe.empty:
        return 0.0

    ordered = dataframe.copy()
    if "chrono_key" in ordered.columns:
        ordered = ordered.sort_values("chrono_key")

    signed_qty = np.where(ordered["side_lp"].eq("BUY"), ordered["quantity_lots"], -ordered["quantity_lots"])
    cumulative_position = pd.Series(signed_qty, dtype="float64").fillna(0.0).cumsum()
    if cumulative_position.empty:
        return 0.0
    return float(cumulative_position.abs().max())


def _side_label_from_position(position: float) -> str:
    if position > 0:
        return "LONG"
    if position < 0:
        return "SHORT"
    return "FLAT"


def _replication_side_label_from_position(position: float) -> str:
    if position > 0:
        return "SHORT"
    if position < 0:
        return "LONG"
    return "FLAT"


def _position_sign(position: float) -> int:
    if position > 0:
        return 1
    if position < 0:
        return -1
    return 0


def _coerce_float(value: object, fallback: float = np.nan) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return fallback
    return float(numeric)


def _with_empty_margin_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    output = dataframe.copy()
    for column in MARGIN_CALL_COLUMNS:
        dtype = "object" if column == "previous_price_date" else "float64"
        output[column] = pd.Series(dtype=dtype)
    return output


def _open_notional_totals_by_future_side(dataframe: pd.DataFrame) -> dict:
    if dataframe.empty or "notional_mad" not in dataframe.columns:
        return {
            "open_notional_futures_long": 0.0,
            "open_notional_futures_short": 0.0,
        }

    open_view = dataframe.copy()
    if "abs_position" in open_view.columns:
        open_view = open_view.loc[open_view["abs_position"].fillna(0.0) > 0].copy()

    if open_view.empty or "side_label" not in open_view.columns:
        return {
            "open_notional_futures_long": 0.0,
            "open_notional_futures_short": 0.0,
        }

    return {
        "open_notional_futures_long": float(
            open_view.loc[open_view["side_label"] == "LONG", "notional_mad"].sum()
        ),
        "open_notional_futures_short": float(
            open_view.loc[open_view["side_label"] == "SHORT", "notional_mad"].sum()
        ),
    }


def compute_contract_metrics(
    contracts_df: pd.DataFrame, transactions_df: pd.DataFrame, settings: dict
) -> pd.DataFrame:
    contracts = contracts_df.copy()
    if contracts.empty:
        return pd.DataFrame(
            columns=[
                "contract_code",
                "underlying_name",
                "net_position",
                "side_label",
                "replication_side_label",
                "direction",
                "abs_position",
                "wap_buys",
                "wap_sells",
                "entry_wap",
                "delta_points",
                "pnl_unrealized_mad",
                "matched_qty",
                "pnl_realized_mad",
                "pnl_accounting_mad",
                "commissions_mad",
                "pnl_management_mad",
                "notional_mad",
                "peak_abs_position",
            ]
        )

    valid_contracts = contracts.loc[contracts["is_valid"].fillna(False)].drop_duplicates("contract_code").copy()
    valid_trades = transactions_df.loc[transactions_df["is_official_for_calc"].fillna(False)].copy()
    fee_components = pd.to_numeric(
        pd.Series(
            [settings.get(key) for key in ("commission_bvc_rt", "commission_broker_rt", "commission_sgmat_rt")]
        ),
        errors="coerce",
    ).fillna(0.0)
    round_trip_fee_per_lot = float(fee_components.sum())

    metrics: list[dict] = []
    for contract in valid_contracts.itertuples():
        contract_trades = valid_trades.loc[valid_trades["contract"] == contract.contract_code].copy()
        buys = contract_trades.loc[contract_trades["side_lp"] == "BUY"]
        sells = contract_trades.loc[contract_trades["side_lp"] == "SELL"]
        total_buys_lots = float(buys["quantity_lots"].sum())
        total_sells_lots = float(sells["quantity_lots"].sum())
        net_position = total_buys_lots - total_sells_lots
        side_label = "LONG" if net_position > 0 else "SHORT" if net_position < 0 else "FLAT"
        replication_side_label = _replication_side_label_from_position(net_position)
        direction = 1.0 if net_position >= 0 else -1.0
        abs_position = abs(net_position)
        wap_buys = _safe_weighted_average(buys)
        wap_sells = _safe_weighted_average(sells)
        entry_wap = wap_buys if net_position > 0 else wap_sells if net_position < 0 else 0.0
        delta_points = float(contract.mtm_price - entry_wap) if pd.notna(contract.mtm_price) else 0.0
        tick_value = float(contract.effective_tick_value)
        matched_qty = min(total_buys_lots, total_sells_lots)
        pnl_unrealized_mad = (
            0.0
            if abs_position == 0 or pd.isna(contract.mtm_price)
            else float(delta_points * direction * abs_position * tick_value)
        )
        pnl_realized_mad = (
            0.0 if matched_qty == 0 else float((wap_sells - wap_buys) * matched_qty * tick_value)
        )
        pnl_accounting_mad = pnl_unrealized_mad + pnl_realized_mad
        commissions_mad = float(matched_qty * round_trip_fee_per_lot)
        pnl_management_mad = pnl_accounting_mad - commissions_mad
        open_contracts = abs_position
        notional_mad = float(entry_wap * tick_value * open_contracts) if open_contracts > 0 else 0.0
        peak_abs_position = _peak_abs_position_from_trades(contract_trades)

        metrics.append(
            {
                "contract_code": contract.contract_code,
                "underlying_name": contract.underlying_name,
                "expiry_date": contract.expiry_date,
                "days_to_expiry": contract.days_to_expiry,
                "expiry_alert": contract.expiry_alert,
                "settlement_price_points": contract.settlement_price_points,
                "mtm_price": contract.mtm_price,
                "mtm_source": contract.mtm_source,
                "effective_tick_value": tick_value,
                "effective_position_limit_per_contract": float(contract.effective_position_limit_per_contract)
                if pd.notna(contract.effective_position_limit_per_contract)
                else np.nan,
                "total_buys_lots": total_buys_lots,
                "total_sells_lots": total_sells_lots,
                "official_net_position": net_position,
                "net_position": net_position,
                "side_label": side_label,
                "replication_side_label": replication_side_label,
                "direction": direction,
                "abs_position": abs_position,
                "wap_buys": wap_buys,
                "wap_sells": wap_sells,
                "entry_wap": entry_wap,
                "delta_points": delta_points,
                "pnl_unrealized_mad": pnl_unrealized_mad,
                "matched_qty": matched_qty,
                "pnl_realized_mad": pnl_realized_mad,
                "pnl_accounting_mad": pnl_accounting_mad,
                "round_trip_fee_per_lot": float(round_trip_fee_per_lot),
                "commissions_mad": commissions_mad,
                "pnl_management_mad": pnl_management_mad,
                "notional_mad": notional_mad,
                "peak_abs_position": peak_abs_position,
                "position_limit_breach": bool(
                    pd.notna(contract.effective_position_limit_per_contract)
                    and abs_position > float(contract.effective_position_limit_per_contract)
                ),
            }
        )

    metrics_df = pd.DataFrame(metrics)
    if metrics_df.empty:
        return metrics_df
    return metrics_df.sort_values("contract_code").reset_index(drop=True)


def compute_global_metrics(contract_metrics_df: pd.DataFrame) -> dict:
    totals = {
        "total_unrealized_pnl": float(contract_metrics_df.get("pnl_unrealized_mad", pd.Series(dtype=float)).sum()),
        "total_realized_pnl": float(contract_metrics_df.get("pnl_realized_mad", pd.Series(dtype=float)).sum()),
        "total_accounting_pnl": float(contract_metrics_df.get("pnl_accounting_mad", pd.Series(dtype=float)).sum()),
        "total_commissions": float(contract_metrics_df.get("commissions_mad", pd.Series(dtype=float)).sum()),
        "total_management_pnl": float(contract_metrics_df.get("pnl_management_mad", pd.Series(dtype=float)).sum()),
        "total_notional": float(contract_metrics_df.get("notional_mad", pd.Series(dtype=float)).sum()),
    }
    totals.update(_open_notional_totals_by_future_side(contract_metrics_df))
    totals["global_exposure"] = (
        totals["open_notional_futures_long"] + totals["open_notional_futures_short"]
    )
    return totals


def build_cmp_portfolio_view(contract_metrics_df: pd.DataFrame, cmp_summary_df: pd.DataFrame) -> pd.DataFrame:
    output_columns = [
        "contract_code",
        "underlying_name",
        "expiry_date",
        "days_to_expiry",
        "expiry_alert",
        "mtm_price",
        "mtm_source",
        "effective_tick_value",
        "effective_position_limit_per_contract",
        "cmp_final_position",
        "cmp_final_cost",
        "cmp_realized_total",
        "cmp_unrealized",
        "cmp_total",
        "total_buys_lots",
        "total_sells_lots",
        "official_net_position",
        "net_position",
        "side_label",
        "replication_side_label",
        "direction",
        "abs_position",
        "entry_wap",
        "delta_points",
        "pnl_unrealized_mad",
        "pnl_realized_mad",
        "pnl_accounting_mad",
        "commissions_mad",
        "pnl_management_mad",
        "notional_mad",
        "position_limit_breach",
    ]
    if contract_metrics_df.empty:
        return pd.DataFrame(columns=output_columns)

    base = contract_metrics_df[
        [
            "contract_code",
            "underlying_name",
            "expiry_date",
            "days_to_expiry",
            "expiry_alert",
            "mtm_price",
            "mtm_source",
            "effective_tick_value",
            "effective_position_limit_per_contract",
            "total_buys_lots",
            "total_sells_lots",
            "commissions_mad",
        ]
    ].copy()

    cmp_summary = (
        cmp_summary_df[
            [
                "contract_code",
                "cmp_final_position",
                "cmp_final_cost",
                "cmp_realized_total",
                "cmp_unrealized",
                "cmp_total",
            ]
        ].copy()
        if not cmp_summary_df.empty
        else pd.DataFrame(
            columns=[
                "contract_code",
                "cmp_final_position",
                "cmp_final_cost",
                "cmp_realized_total",
                "cmp_unrealized",
                "cmp_total",
            ]
        )
    )

    portfolio = pd.merge(base, cmp_summary, on="contract_code", how="left")
    numeric_fill_columns = [
        "cmp_final_position",
        "cmp_final_cost",
        "cmp_realized_total",
        "cmp_unrealized",
        "cmp_total",
    ]
    for column in numeric_fill_columns:
        portfolio[column] = pd.to_numeric(portfolio[column], errors="coerce").fillna(0.0)

    portfolio["abs_position"] = portfolio["cmp_final_position"].abs()
    portfolio["side_label"] = portfolio["cmp_final_position"].map(_side_label_from_position)
    portfolio["replication_side_label"] = portfolio["cmp_final_position"].map(_replication_side_label_from_position)
    portfolio["official_net_position"] = portfolio["cmp_final_position"]
    portfolio["net_position"] = portfolio["cmp_final_position"]
    portfolio["direction"] = np.where(portfolio["cmp_final_position"].ge(0), 1.0, -1.0)
    portfolio["entry_wap"] = portfolio["cmp_final_cost"]
    portfolio["delta_points"] = np.where(
        portfolio["mtm_price"].notna(),
        portfolio["mtm_price"] - portfolio["cmp_final_cost"],
        0.0,
    )
    portfolio["pnl_unrealized_mad"] = portfolio["cmp_unrealized"]
    portfolio["pnl_realized_mad"] = portfolio["cmp_realized_total"]
    portfolio["pnl_accounting_mad"] = portfolio["cmp_total"]
    portfolio["commissions_mad"] = pd.to_numeric(portfolio["commissions_mad"], errors="coerce").fillna(0.0)
    portfolio["pnl_management_mad"] = portfolio["pnl_accounting_mad"] - portfolio["commissions_mad"]
    portfolio["notional_mad"] = portfolio["abs_position"] * portfolio["cmp_final_cost"] * portfolio["effective_tick_value"]
    portfolio["position_limit_breach"] = (
        portfolio["effective_position_limit_per_contract"].notna()
        & portfolio["abs_position"].gt(portfolio["effective_position_limit_per_contract"])
    )

    return portfolio[output_columns].sort_values("contract_code").reset_index(drop=True)


def _portfolio_numeric_map(
    portfolio_df: pd.DataFrame | None,
    value_column: str,
    fallback_column: str | None = None,
) -> dict[str, float]:
    if portfolio_df is None or portfolio_df.empty or "contract_code" not in portfolio_df.columns:
        return {}

    portfolio = portfolio_df.copy()
    if value_column not in portfolio.columns:
        if fallback_column is None or fallback_column not in portfolio.columns:
            return {}
        value_column = fallback_column

    portfolio = portfolio.dropna(subset=["contract_code"]).drop_duplicates("contract_code", keep="last")
    return dict(
        zip(
            portfolio["contract_code"],
            pd.to_numeric(portfolio[value_column], errors="coerce").fillna(0.0),
        )
    )


def _incremental_position_and_cmp(
    current_position: float,
    current_cmp: float,
    previous_position: float,
    previous_cmp: float,
) -> tuple[float, float]:
    if current_position == 0:
        return 0.0, np.nan

    current_sign = _position_sign(current_position)
    previous_sign = _position_sign(previous_position)
    if previous_sign == 0 or previous_sign != current_sign:
        return current_position, current_cmp

    if abs(current_position) <= abs(previous_position):
        return 0.0, np.nan

    new_position = current_position - previous_position
    if pd.isna(current_cmp):
        return new_position, np.nan
    if pd.isna(previous_cmp) or previous_cmp == 0:
        return new_position, current_cmp

    new_abs_position = abs(new_position)
    if new_abs_position == 0:
        return 0.0, np.nan

    derived_cmp = (
        abs(current_position) * float(current_cmp)
        - abs(previous_position) * float(previous_cmp)
    ) / new_abs_position
    if not np.isfinite(derived_cmp) or derived_cmp <= 0:
        derived_cmp = current_cmp
    return new_position, float(derived_cmp)


def enrich_daily_margin_calls(
    contract_metrics_df: pd.DataFrame,
    previous_portfolio_df: pd.DataFrame | None,
    previous_price_map: dict[str, float],
    previous_price_date: str | None,
) -> pd.DataFrame:
    if contract_metrics_df.empty:
        return _with_empty_margin_columns(contract_metrics_df)

    metrics = contract_metrics_df.copy()
    previous_positions = _portfolio_numeric_map(
        previous_portfolio_df,
        "cmp_final_position",
        fallback_column="net_position",
    )
    previous_cmps = _portfolio_numeric_map(
        previous_portfolio_df,
        "cmp_final_cost",
        fallback_column="entry_wap",
    )

    records: list[dict] = []
    for row in metrics.itertuples():
        contract_code = row.contract_code
        current_price = _coerce_float(getattr(row, "mtm_price", np.nan))
        current_position = _coerce_float(getattr(row, "cmp_final_position", 0.0), fallback=0.0)
        current_cmp = _coerce_float(getattr(row, "cmp_final_cost", np.nan))
        tick_value = _coerce_float(getattr(row, "effective_tick_value", 0.0), fallback=0.0)

        previous_price = previous_price_map.get(contract_code, np.nan)
        raw_previous_position = float(previous_positions.get(contract_code, 0.0))
        previous_cmp = float(previous_cmps.get(contract_code, np.nan))

        has_current_price = pd.notna(current_price)
        has_previous_reference = pd.notna(previous_price) and raw_previous_position != 0
        previous_position_for_margin = raw_previous_position if has_previous_reference else 0.0
        previous_cmp_for_margin = previous_cmp if has_previous_reference else np.nan

        price_delta = (
            float(current_price) - float(previous_price)
            if has_current_price and pd.notna(previous_price)
            else np.nan
        )
        daily_mtm = (
            float(price_delta * previous_position_for_margin * tick_value)
            if has_current_price and pd.notna(price_delta) and previous_position_for_margin != 0
            else 0.0
        )

        new_position, new_cmp = _incremental_position_and_cmp(
            current_position,
            current_cmp,
            previous_position_for_margin,
            previous_cmp_for_margin,
        )
        daily_latent = (
            float((float(current_price) - float(new_cmp)) * new_position * tick_value)
            if has_current_price and new_position != 0 and pd.notna(new_cmp)
            else 0.0
        )
        margin_call = daily_mtm + daily_latent

        records.append(
            {
                "contract_code": contract_code,
                "previous_price_date": previous_price_date,
                "previous_mtm_price": previous_price,
                "daily_price_delta_points": price_delta,
                "previous_open_position": raw_previous_position,
                "new_position_today": new_position,
                "new_position_cmp": new_cmp,
                "daily_mtm_mad": daily_mtm,
                "daily_latent_mad": daily_latent,
                "margin_call_mad": margin_call,
            }
        )

    margin_df = pd.DataFrame(records)
    return pd.merge(metrics, margin_df, on="contract_code", how="left")


def build_dashboard_alerts(
    contracts_df: pd.DataFrame,
    contract_issues_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    transaction_issues_df: pd.DataFrame,
    contract_metrics_df: pd.DataFrame,
) -> list[dict]:
    alerts: list[dict] = []

    if not contract_metrics_df.empty and {"contract_code", "abs_position", "mtm_price"}.issubset(contract_metrics_df.columns):
        open_without_mtm = contract_metrics_df.loc[
            (contract_metrics_df["abs_position"] > 0) & contract_metrics_df["mtm_price"].isna()
        ]
        for record in open_without_mtm.itertuples():
            alerts.append(
                {
                    "severity": "warning",
                    "category": "open_position_without_mtm",
                    "message": f"{record.contract_code}: position ouverte sans MtM renseigne.",
                }
            )

    if not contract_metrics_df.empty and {
        "contract_code",
        "expiry_alert",
        "days_to_expiry",
        "expiry_date",
    }.issubset(contract_metrics_df.columns):
        expiry_watch = contract_metrics_df.loc[
            contract_metrics_df["expiry_alert"].isin(["Near expiry", "Expired"])
        ].copy()
        expiry_watch = expiry_watch.sort_values(["days_to_expiry", "contract_code"], na_position="last")
        for record in expiry_watch.itertuples():
            days_to_expiry = pd.to_numeric(record.days_to_expiry, errors="coerce")
            expiry_date = record.expiry_date if pd.notna(record.expiry_date) else "-"
            if pd.notna(days_to_expiry) and days_to_expiry < 0:
                alerts.append(
                    {
                        "severity": "error",
                        "category": "contract_expired",
                        "message": f"{record.contract_code}: echeance depassee depuis {abs(int(days_to_expiry))} jours ({expiry_date}).",
                    }
                )
            else:
                days_text = int(days_to_expiry) if pd.notna(days_to_expiry) else "?"
                alerts.append(
                    {
                        "severity": "warning",
                        "category": "contract_near_expiry",
                        "message": f"{record.contract_code}: echeance proche dans {days_text} jours ({expiry_date}).",
                    }
                )

    if not contract_metrics_df.empty and {"position_limit_breach", "contract_code", "abs_position"}.issubset(contract_metrics_df.columns):
        breached_limits = contract_metrics_df.loc[contract_metrics_df["position_limit_breach"].fillna(False)]
        for record in breached_limits.itertuples():
            alerts.append(
                {
                    "severity": "warning",
                    "category": "position_limit_breach",
                    "message": f"{record.contract_code}: limite de position depassee ({record.abs_position} lots).",
                }
            )

    duplicate_execution_ids = transaction_issues_df.loc[
        transaction_issues_df["code"] == "duplicate_execution_id", "entity"
    ].dropna()
    for execution_id in duplicate_execution_ids.unique().tolist():
        alerts.append(
            {
                "severity": "error",
                "category": "duplicate_execution_id",
                "message": f"{execution_id}: execution_id en doublon.",
            }
        )

    invalid_rows = sorted(transaction_issues_df.loc[transaction_issues_df["severity"] == "error", "row"].unique().tolist())
    if invalid_rows:
        alerts.append(
            {
                "severity": "error",
                "category": "invalid_transaction",
                "message": f"Transactions invalides detectees sur les lignes {invalid_rows}.",
            }
        )

    missing_contracts = transaction_issues_df.loc[
        transaction_issues_df["code"] == "unknown_contract", "row"
    ].tolist()
    if missing_contracts:
        alerts.append(
            {
                "severity": "error",
                "category": "transaction_contract_missing_from_reference",
                "message": f"Contrat absent du referentiel pour les transactions lignes {missing_contracts}.",
            }
        )

    invalid_contract_rows = sorted(contract_issues_df.loc[contract_issues_df["severity"] == "error", "row"].unique().tolist())
    if invalid_contract_rows:
        alerts.append(
            {
                "severity": "error",
                "category": "invalid_contract",
                "message": f"Contrats invalides detectes sur les lignes {invalid_contract_rows}.",
            }
        )

    return alerts


def compute_confirmed_positions(transactions_df: pd.DataFrame, contract_metrics_df: pd.DataFrame) -> pd.DataFrame:
    if transactions_df.empty or contract_metrics_df.empty:
        return pd.DataFrame(columns=["contract_code", "official_net_position", "confirmed_net_position", "delta_vs_all"])

    valid_tx = transactions_df.loc[transactions_df["is_official_for_calc"].fillna(False)].copy()
    confirmed_tx = valid_tx.loc[valid_tx["is_confirmed"].fillna(False)]

    records = []
    if not confirmed_tx.empty:
        grouped = confirmed_tx.groupby("contract")
        for contract, group in grouped:
            buys = float(group.loc[group["side_lp"] == "BUY", "quantity_lots"].sum())
            sells = float(group.loc[group["side_lp"] == "SELL", "quantity_lots"].sum())
            records.append({
                "contract_code": contract,
                "confirmed_net_position": buys - sells
            })

    confirmed_pos = pd.DataFrame(records)
    if confirmed_pos.empty:
        confirmed_pos = pd.DataFrame(columns=["contract_code", "confirmed_net_position"])

    merged = pd.merge(
        contract_metrics_df[["contract_code", "official_net_position"]],
        confirmed_pos,
        on="contract_code",
        how="left"
    )
    merged["confirmed_net_position"] = merged["confirmed_net_position"].fillna(0.0)
    merged["delta_vs_all"] = merged["confirmed_net_position"] - merged["official_net_position"]

    return merged


def compute_cmp_sequential(transactions_df: pd.DataFrame, contract_metrics_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_cols = [
        "execution_id", "contract_code", "trade_date", "side_lp", "quantity_lots", "price_points",
        "signed_qty", "pos_before", "cmp_before", "closed_qty", "trade_realized_pnl", "pos_after", "cmp_after"
    ]
    summary_cols = [
        "contract_code", "cmp_final_position", "cmp_final_cost", "cmp_realized_total", "cmp_unrealized",
        "cmp_total"
    ]

    if transactions_df.empty or contract_metrics_df.empty:
        return pd.DataFrame(columns=detail_cols), pd.DataFrame(columns=summary_cols)

    valid_tx = transactions_df.loc[transactions_df["is_official_for_calc"].fillna(False)].copy()
    if valid_tx.empty:
        summary_empty = contract_metrics_df[["contract_code"]].copy()
        for col in ["cmp_final_position", "cmp_final_cost", "cmp_realized_total", "cmp_unrealized", "cmp_total"]:
            summary_empty[col] = 0.0
        return pd.DataFrame(columns=detail_cols), summary_empty[summary_cols]

    valid_tx = valid_tx.sort_values(["chrono_key"]).reset_index(drop=True)

    tick_values = dict(zip(contract_metrics_df["contract_code"], contract_metrics_df["effective_tick_value"]))
    mtm_prices = dict(zip(contract_metrics_df["contract_code"], contract_metrics_df["mtm_price"]))

    detail_records = []
    summary_records = []

    grouped = valid_tx.groupby("contract")

    for contract, group in grouped:
        tick_value = tick_values.get(contract, 0.0)
        mtm_price = mtm_prices.get(contract, np.nan)

        pos = 0.0
        cmp_val = 0.0
        realized_pnl_total = 0.0

        for record in group.itertuples():
            qty = float(record.quantity_lots)
            price = float(record.price_points)
            side = record.side_lp
            signed_qty = qty if side == "BUY" else -qty

            pos_before = pos
            cmp_before = cmp_val

            closed_qty = 0.0
            trade_realized_pnl = 0.0

            if pos_before == 0:
                pos = signed_qty
                cmp_val = price
            elif np.sign(pos_before) == np.sign(signed_qty):
                abs_pos = abs(pos_before)
                cmp_val = ((abs_pos * cmp_before) + (qty * price)) / (abs_pos + qty)
                pos = pos_before + signed_qty
            else:
                closing_qty = min(abs(pos_before), qty)
                closed_qty = closing_qty
                if side == "SELL":
                    trade_points = price - cmp_before
                else:
                    trade_points = cmp_before - price

                trade_realized_pnl = float(closing_qty * trade_points * tick_value)
                realized_pnl_total += trade_realized_pnl

                remaining_qty = qty - closing_qty
                pos = pos_before + signed_qty
                if remaining_qty == 0:
                    # A pure reduction keeps the prior CMP on the residual position.
                    cmp_val = 0.0 if pos == 0 else cmp_before
                else:
                    cmp_val = price

            detail_records.append({
                "execution_id": record.execution_id,
                "contract_code": contract,
                "trade_date": record.trade_date,
                "side_lp": side,
                "quantity_lots": qty,
                "price_points": price,
                "signed_qty": signed_qty,
                "pos_before": pos_before,
                "cmp_before": cmp_before,
                "closed_qty": closed_qty,
                "trade_realized_pnl": trade_realized_pnl,
                "pos_after": pos,
                "cmp_after": cmp_val,
            })

        cmp_unrealized = 0.0
        if pos != 0 and pd.notna(mtm_price):
            direction = 1.0 if pos > 0 else -1.0
            cmp_unrealized = float((mtm_price - cmp_val) * direction * abs(pos) * tick_value)

        cmp_total = realized_pnl_total + cmp_unrealized

        summary_records.append({
            "contract_code": contract,
            "cmp_final_position": pos,
            "cmp_final_cost": cmp_val,
            "cmp_realized_total": realized_pnl_total,
            "cmp_unrealized": cmp_unrealized,
            "cmp_total": cmp_total,
        })

    detail_df = pd.DataFrame(detail_records) if detail_records else pd.DataFrame(columns=detail_cols)
    summary_df = pd.DataFrame(summary_records) if summary_records else pd.DataFrame(columns=summary_cols)

    if not contract_metrics_df.empty:
        all_contracts = contract_metrics_df[["contract_code"]].copy()
        if not summary_df.empty:
            summary_df = pd.merge(all_contracts, summary_df, on="contract_code", how="left")

            for col in ["cmp_final_position", "cmp_final_cost", "cmp_realized_total", "cmp_unrealized", "cmp_total"]:
                summary_df[col] = summary_df[col].fillna(0.0)
        else:
            summary_df = all_contracts
            for col in ["cmp_final_position", "cmp_final_cost", "cmp_realized_total", "cmp_unrealized", "cmp_total"]:
                summary_df[col] = 0.0
            
        summary_df = summary_df[summary_cols]

    return detail_df, summary_df
