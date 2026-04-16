import numpy as np
import pandas as pd

from .config import CMP_TOLERANCE


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


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else np.nan


def _side_label_from_position(position: float) -> str:
    if position > 0:
        return "LONG"
    if position < 0:
        return "SHORT"
    return "FLAT"


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
                "signed_notional_mad",
                "margin_mad",
                "peak_abs_position",
                "capital_engaged_mad",
                "leverage",
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
        notional_mad = float(entry_wap * tick_value * abs_position) if abs_position > 0 else 0.0
        signed_notional_mad = float(entry_wap * tick_value * net_position) if net_position != 0 else 0.0
        margin_mad = float(abs_position * contract.effective_initial_margin_per_lot)
        peak_abs_position = _peak_abs_position_from_trades(contract_trades)
        capital_engaged_mad = float(peak_abs_position * contract.effective_initial_margin_per_lot)
        leverage = float(notional_mad / margin_mad) if margin_mad else 0.0

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
                "effective_initial_margin_per_lot": float(contract.effective_initial_margin_per_lot),
                "effective_position_limit_per_contract": float(contract.effective_position_limit_per_contract)
                if pd.notna(contract.effective_position_limit_per_contract)
                else np.nan,
                "total_buys_lots": total_buys_lots,
                "total_sells_lots": total_sells_lots,
                "official_net_position": net_position,
                "net_position": net_position,
                "side_label": side_label,
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
                "signed_notional_mad": signed_notional_mad,
                "margin_mad": margin_mad,
                "peak_abs_position": peak_abs_position,
                "capital_engaged_mad": capital_engaged_mad,
                "leverage": leverage,
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
        "total_net_notional": float(contract_metrics_df.get("signed_notional_mad", pd.Series(dtype=float)).sum()),
        "total_margin": float(contract_metrics_df.get("margin_mad", pd.Series(dtype=float)).sum()),
    }
    total_margin = totals["total_margin"]
    capital_total_engaged = float(
        contract_metrics_df.get("capital_engaged_mad", pd.Series(dtype=float)).sum()
    )
    if not capital_total_engaged:
        capital_total_engaged = total_margin
    totals.update(
        {
            "global_leverage": float(totals["total_notional"] / total_margin) if total_margin else 0.0,
            "roi_on_margin": _safe_ratio(totals["total_management_pnl"], total_margin),
            "capital_total_engaged": capital_total_engaged,
            "roi_on_capital_engaged": _safe_ratio(totals["total_management_pnl"], capital_total_engaged),
        }
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
        "effective_initial_margin_per_lot",
        "effective_position_limit_per_contract",
        "cmp_final_position",
        "cmp_final_cost",
        "cmp_realized_total",
        "cmp_unrealized",
        "cmp_total",
        "difference_vs_wap",
        "within_tolerance",
        "side_label",
        "abs_position",
        "notional_mad",
        "signed_notional_mad",
        "margin_mad",
        "leverage",
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
            "effective_initial_margin_per_lot",
            "effective_position_limit_per_contract",
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
                "difference_vs_wap",
                "within_tolerance",
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
                "difference_vs_wap",
                "within_tolerance",
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
        "difference_vs_wap",
    ]
    for column in numeric_fill_columns:
        portfolio[column] = pd.to_numeric(portfolio[column], errors="coerce").fillna(0.0)
    portfolio["within_tolerance"] = (
        portfolio["within_tolerance"]
        .astype("boolean")
        .fillna(True)
        .astype(bool)
    )

    portfolio["abs_position"] = portfolio["cmp_final_position"].abs()
    portfolio["side_label"] = portfolio["cmp_final_position"].map(_side_label_from_position)
    portfolio["notional_mad"] = (
        portfolio["abs_position"] * portfolio["cmp_final_cost"] * portfolio["effective_tick_value"]
    )
    portfolio["signed_notional_mad"] = (
        portfolio["cmp_final_position"] * portfolio["cmp_final_cost"] * portfolio["effective_tick_value"]
    )
    portfolio["margin_mad"] = portfolio["abs_position"] * portfolio["effective_initial_margin_per_lot"]
    portfolio["leverage"] = np.where(
        portfolio["margin_mad"].ne(0),
        portfolio["notional_mad"] / portfolio["margin_mad"],
        0.0,
    )
    portfolio["position_limit_breach"] = (
        portfolio["effective_position_limit_per_contract"].notna()
        & portfolio["abs_position"].gt(portfolio["effective_position_limit_per_contract"])
    )

    return portfolio[output_columns].sort_values("contract_code").reset_index(drop=True)


def compute_cmp_global_metrics(cmp_portfolio_df: pd.DataFrame) -> dict:
    totals = {
        "total_cmp_unrealized": float(cmp_portfolio_df.get("cmp_unrealized", pd.Series(dtype=float)).sum()),
        "total_cmp_realized": float(cmp_portfolio_df.get("cmp_realized_total", pd.Series(dtype=float)).sum()),
        "total_cmp_pnl": float(cmp_portfolio_df.get("cmp_total", pd.Series(dtype=float)).sum()),
        "total_notional": float(cmp_portfolio_df.get("notional_mad", pd.Series(dtype=float)).sum()),
        "total_net_notional": float(cmp_portfolio_df.get("signed_notional_mad", pd.Series(dtype=float)).sum()),
        "total_margin": float(cmp_portfolio_df.get("margin_mad", pd.Series(dtype=float)).sum()),
    }
    total_margin = totals["total_margin"]
    totals.update(
        {
            "global_leverage": float(totals["total_notional"] / total_margin) if total_margin else 0.0,
            "roi_on_margin": _safe_ratio(totals["total_cmp_pnl"], total_margin),
        }
    )
    return totals


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
        "cmp_total", "wap_accounting_total", "difference_vs_wap", "within_tolerance"
    ]

    if transactions_df.empty or contract_metrics_df.empty:
        return pd.DataFrame(columns=detail_cols), pd.DataFrame(columns=summary_cols)

    valid_tx = transactions_df.loc[transactions_df["is_official_for_calc"].fillna(False)].copy()
    if valid_tx.empty:
        summary_empty = contract_metrics_df[["contract_code", "pnl_accounting_mad"]].copy()
        summary_empty.rename(columns={"pnl_accounting_mad": "wap_accounting_total"}, inplace=True)
        for col in ["cmp_final_position", "cmp_final_cost", "cmp_realized_total", "cmp_unrealized", "cmp_total", "difference_vs_wap"]:
            summary_empty[col] = 0.0
        summary_empty["within_tolerance"] = True
        return pd.DataFrame(columns=detail_cols), summary_empty[summary_cols]

    valid_tx = valid_tx.sort_values(["chrono_key"]).reset_index(drop=True)

    tick_values = dict(zip(contract_metrics_df["contract_code"], contract_metrics_df["effective_tick_value"]))
    mtm_prices = dict(zip(contract_metrics_df["contract_code"], contract_metrics_df["mtm_price"]))
    wap_accounting = dict(zip(contract_metrics_df["contract_code"], contract_metrics_df["pnl_accounting_mad"]))

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
        wap_acc = wap_accounting.get(contract, 0.0)
        diff = cmp_total - wap_acc
        within_tol = abs(diff) <= CMP_TOLERANCE

        summary_records.append({
            "contract_code": contract,
            "cmp_final_position": pos,
            "cmp_final_cost": cmp_val,
            "cmp_realized_total": realized_pnl_total,
            "cmp_unrealized": cmp_unrealized,
            "cmp_total": cmp_total,
            "wap_accounting_total": wap_acc,
            "difference_vs_wap": diff,
            "within_tolerance": within_tol
        })

    detail_df = pd.DataFrame(detail_records) if detail_records else pd.DataFrame(columns=detail_cols)
    summary_df = pd.DataFrame(summary_records) if summary_records else pd.DataFrame(columns=summary_cols)

    if not contract_metrics_df.empty:
        all_contracts = contract_metrics_df[["contract_code"]].copy()
        if not summary_df.empty:
            summary_df = pd.merge(all_contracts, summary_df, on="contract_code", how="left")
            summary_df["wap_accounting_total"] = summary_df["contract_code"].map(wap_accounting).fillna(0.0)
            
            for col in ["cmp_final_position", "cmp_final_cost", "cmp_realized_total", "cmp_unrealized", "cmp_total"]:
                summary_df[col] = summary_df[col].fillna(0.0)
            
            summary_df["difference_vs_wap"] = summary_df["cmp_total"] - summary_df["wap_accounting_total"]
            summary_df["within_tolerance"] = summary_df["difference_vs_wap"].abs() <= CMP_TOLERANCE
        else:
            summary_df = all_contracts
            summary_df["wap_accounting_total"] = summary_df["contract_code"].map(wap_accounting).fillna(0.0)
            for col in ["cmp_final_position", "cmp_final_cost", "cmp_realized_total", "cmp_unrealized", "cmp_total"]:
                summary_df[col] = 0.0
            summary_df["difference_vs_wap"] = summary_df["cmp_total"] - summary_df["wap_accounting_total"]
            summary_df["within_tolerance"] = summary_df["difference_vs_wap"].abs() <= CMP_TOLERANCE
            
        summary_df = summary_df[summary_cols]

    return detail_df, summary_df
