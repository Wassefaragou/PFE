# MASI20 Futures PnL Tracker

Application Streamlit complete pour gerer un referentiel de contrats futures sur indice, enregistrer les transactions et suivre le P&L officiel en WAP ainsi qu'un controle alternatif en CMP sequentiel.

## Arborescence

```text
masi20_futures_pnl_tracker/
|-- masi20_futures_pnl_tracker_app.py
|-- requirements.txt
|-- README.md
|-- style.css
|-- futures_pnl/
|   |-- __init__.py
|   |-- analytics.py
|   |-- config.py
|   |-- pricing.py
|   |-- storage.py
|   |-- ui.py
|   `-- validators.py
|-- pages/
|   |-- dashboard_page.py
|   |-- parametres_page.py
|   |-- referentiel_contrats_page.py
|   |-- transactions_page.py
|   |-- position_par_contrat_page.py
|   |-- pnl_global_page.py
|   `-- cmp_sequentiel_page.py
`-- storage/
    `-- .gitkeep
```

## Installation

```bash
cd masi20_futures_pnl_tracker
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run masi20_futures_pnl_tracker_app.py
```

## Donnees

- Le dossier `storage/` est le stockage local de travail de l'application.
- Au premier lancement, l'application cree un stockage vide.
- Depuis la barre laterale, vous pouvez reinitialiser le stockage local.
- L'application ne livre aucun fichier metier precharge: vous construisez librement votre propre referentiel et vos propres transactions.
- La marge initiale est maintenant renseignee contrat par contrat dans le referentiel.

## Formules implementees

### Prix theorique

- `days_to_expiry = expiry_date - valuation_date`
- `theoretical_price = spot_index * exp((risk_free_rate - dividend_yield) * days_to_expiry / 360)`
- `mtm_price = settlement_price_points` si `is_active_lp = True` et settlement renseigne
- sinon `mtm_price = theoretical_price`

### Moteur officiel WAP agrege

- `total_buys_lots = somme BUY`
- `total_sells_lots = somme SELL`
- `net_position = total_buys_lots - total_sells_lots`
- `direction = +1 si net_position >= 0 sinon -1`
- `abs_position = abs(net_position)`
- `wap_buys = sum(qty * price) / sum(qty) sur BUY`
- `wap_sells = sum(qty * price) / sum(qty) sur SELL`
- `entry_wap = wap_buys si LONG, wap_sells si SHORT, 0 sinon`
- `delta_points = mtm_price - entry_wap`
- `pnl_unrealized_mad = delta_points * direction * abs_position * tick_value`
- `matched_qty = min(total_buys_lots, total_sells_lots)`
- `pnl_realized_mad = (wap_sells - wap_buys) * matched_qty * tick_value`
- `pnl_accounting_mad = pnl_unrealized_mad + pnl_realized_mad`
- `round_trip_fee_per_lot = commission_bvc_rt + commission_broker_rt + commission_sgmat_rt`
- `commissions_mad = matched_qty * round_trip_fee_per_lot`
- `pnl_management_mad = pnl_accounting_mad - commissions_mad`
- `notional_mad = entry_wap * tick_value * abs_position`  -> exposition ouverte en valeur absolue
- `signed_notional_mad = entry_wap * tick_value * net_position`  -> exposition nette signee
- `margin_mad = abs_position * initial_margin_per_lot`
- `leverage = notional_mad / margin_mad`
- `base_points = mtm_price - spot_index`
- `mispricing_points = mtm_price - theoretical_price`

### Vue globale

- `total_notional = somme des expositions ouvertes en valeur absolue`
- `total_net_notional = somme des expositions nettes signees`
- `global_leverage = total_notional / total_margin`
- `roi_on_margin = total_management_pnl / total_margin`
- dans l'etat actuel du code, `capital_total_engaged = total_margin`
- `roi_on_capital_engaged = total_management_pnl / capital_total_engaged`

### Moteur alternatif CMP sequentiel

Pour chaque trade trie par `chrono_key` :

- `signed_qty = +qty` si BUY, `-qty` si SELL
- `closed_qty = 0` si pas de cloture
- `trade_realized_pnl = sign(pos_before) * (trade_price - cmp_before) * closed_qty * tick_value`
- `cmp_after` suit les regles de moyenne glissante, reduction ou retournement
- `cmp_total = cmp_realized_total + cmp_unrealized`
- controle de coherence avec le moteur WAP officiel a tolerance numerique faible

## Fonctions metier principales

- `compute_theoretical_prices(...)`
- `validate_contracts(...)`
- `validate_transactions(...)`
- `compute_contract_metrics(...)`
- `compute_global_metrics(...)`
- `compute_cmp_sequential(...)`
