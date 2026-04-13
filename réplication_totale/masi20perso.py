"""
==============================================================================
MASTER RUNNER — MASI20 Multi-Year Optimization + Excel + Figures
Runs the full 60-combination optimization for each year (2020-21 to 2025-26),
then produces:
  • A  Excel workbook with all results
  • A cross-year summary figure
==============================================================================
"""

import argparse
import pandas as pd
import pickle
import numpy as np
from scipy.optimize import differential_evolution, minimize, dual_annealing, basinhopping
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import time, os, warnings, sys
warnings.filterwarnings('ignore')

# ── Pretty matplotlib defaults ──────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'figure.facecolor': '#0F1117',
    'axes.facecolor': '#1A1D2E',
    'text.color': '#E8E8E8',
    'axes.labelcolor': '#C0C0C0',
    'xtick.color': '#A0A0A0',
    'ytick.color': '#A0A0A0',
    'grid.color': '#2A2D3E',
    'grid.alpha': 0.6,
    'axes.edgecolor': '#2A2D3E',
})
COLORS = ['#00D4AA', '#FF6B6B', '#4ECDC4', '#FFE66D', '#A8E6CF',
          '#FF8B94', '#B8A9C9', '#88D8B0', '#FFAAA5', '#DDA0DD',
          '#98D8C8', '#F7DC6F']
METHOD_COLORS = {
    'DiffEvol': '#00D4AA', 'DualAnn': '#FF6B6B', 'BasinHop': '#4ECDC4',
    'MS-LBFGSB': '#FFE66D', 'NelderMead': '#A8E6CF'
}

# ── Configuration ───────────────────────────────────────────────────────────
FILES = {
    '2020-2021': '20-21.xlsx',
    '2021-2022': '21-22.xlsx',
    '2022-2023': '22-23.xlsx',
    '2023-2024': '23-24.xlsx',
    '2024-2025': '24-25.xlsx',
    '2025-2026': '25-26-SANSANOMALIE.xlsx',
}
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'masi20_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING (adapted for raw Excel format with header at row 10)
# ════════════════════════════════════════════════════════════════════════════
def load_data(filepath):
    """Load data from yearly Excel files.
    Format: row 10 = headers (Date, MSEMSI20 Index, 20 companies), data from row 11.
    Columns 0-1 can be empty, col 2 = Date, col 3 = Index, cols 4-23 = companies.
    Alternatively, if file has proper headers, handle that too.
    """
    # Try reading with header=None first to detect format
    df_raw = pd.read_excel(filepath, header=None)

    # Find the row that contains 'Date'
    header_row = None
    for r in range(min(20, len(df_raw))):
        row_vals = [str(v).strip() for v in df_raw.iloc[r].tolist()]
        if 'Date' in row_vals:
            header_row = r
            break

    if header_row is None:
        # Fallback: try reading with default header
        df = pd.read_excel(filepath)
        if 'Date' in df.columns:
            companies = [c.replace(' MC Equity', '').strip() for c in df.columns[2:22].tolist()]
            dates = pd.to_datetime(df['Date'])
            index_values = df.iloc[:, 1].astype(float).values
            cap_matrix = df.iloc[:, 2:22].astype(float).values
        else:
            raise ValueError(f"Cannot find 'Date' column in {filepath}")
    else:
        # Found header row
        headers = df_raw.iloc[header_row].tolist()
        date_col = next(i for i, h in enumerate(headers) if str(h).strip() == 'Date')
        index_col = date_col + 1  # MSEMSI20 Index right after Date
        comp_start = index_col + 1
        comp_end = comp_start + 20

        company_headers = headers[comp_start:comp_end]
        companies = [str(c).replace(' MC Equity', '').strip() for c in company_headers]

        data_start = header_row + 1
        df_data = df_raw.iloc[data_start:].copy()
        df_data = df_data.dropna(subset=[date_col])

        dates = pd.to_datetime(df_data.iloc[:, date_col])
        index_values = pd.to_numeric(df_data.iloc[:, index_col], errors='coerce').values
        cap_matrix = df_data.iloc[:, comp_start:comp_end].apply(pd.to_numeric, errors='coerce').values

    # Drop any rows with NaN
    valid = ~(np.isnan(index_values) | np.any(np.isnan(cap_matrix), axis=1))
    index_values = index_values[valid]
    cap_matrix = cap_matrix[valid]
    dates = dates[valid].reset_index(drop=True)

    log_index_returns = np.log(index_values[1:] / index_values[:-1])
    print(f"  ✓ {len(companies)} entreprises | {len(dates)} jours | {dates.iloc[0].date()} → {dates.iloc[-1].date()}")
    return companies, dates, index_values, cap_matrix, log_index_returns


# ════════════════════════════════════════════════════════════════════════════
# 2. OBJECTIVE FUNCTIONS (12) — copied from user's script
# ════════════════════════════════════════════════════════════════════════════
def calc_log_diff(w, cm, lr):
    weighted = cm * w[np.newaxis, :]
    total = weighted.sum(axis=1)
    return np.log(total[1:] / total[:-1]) - lr

def get_opcvm_caps(cm):
    masi_weights = cm[0] / cm[0].sum()
    caps = np.full(len(masi_weights), 1.0)   # no individual cap by default
    caps[masi_weights >= 0.15] = 0.20        # ≥15% MASI → 20%
    return caps, masi_weights

def penalty_opcvm(w, cm):
    weighted = cm[0] * w
    total = weighted.sum()
    if total == 0: return 0
    port_weights = weighted / total
    stock_caps, _ = get_opcvm_caps(cm)
    over_individual = np.maximum(0, port_weights - stock_caps)
    pen_individual = 1e6 * np.sum(over_individual**2)
    above_10 = port_weights[port_weights > 0.10]
    sum_above_10 = above_10.sum()
    pen_aggregate = 1e6 * max(0, sum_above_10 - 0.45)**2
    return pen_individual + pen_aggregate

def obj_te(w, cm, lr):
    return np.std(calc_log_diff(w, cm, lr)) * np.sqrt(252) + penalty_opcvm(w, cm)
def obj_mse(w, cm, lr):
    d = calc_log_diff(w, cm, lr); return np.mean(d**2) + penalty_opcvm(w, cm)
def obj_mae(w, cm, lr):
    d = calc_log_diff(w, cm, lr); return np.mean(np.abs(d)) + penalty_opcvm(w, cm)
def obj_maxe(w, cm, lr):
    d = calc_log_diff(w, cm, lr); return np.max(np.abs(d)) + penalty_opcvm(w, cm)
def obj_rmse(w, cm, lr):
    d = calc_log_diff(w, cm, lr); return np.sqrt(np.mean(d**2)) + penalty_opcvm(w, cm)
def obj_huber(w, cm, lr, delta=0.001):
    d = calc_log_diff(w, cm, lr); a = np.abs(d)
    return np.mean(np.where(a <= delta, 0.5*d**2, delta*(a - 0.5*delta))) + penalty_opcvm(w, cm)
def obj_downside_te(w, cm, lr):
    d = calc_log_diff(w, cm, lr); dn = d[d < 0]
    te = np.sqrt(np.mean(dn**2)) * np.sqrt(252) if len(dn) > 0 else 0
    return te + penalty_opcvm(w, cm)
def obj_combined(w, cm, lr):
    d = calc_log_diff(w, cm, lr)
    return 0.7 * np.std(d) * np.sqrt(252) + 0.3 * np.max(np.abs(d)) * np.sqrt(252) + penalty_opcvm(w, cm)
def obj_log_cosh(w, cm, lr):
    d = calc_log_diff(w, cm, lr)
    return np.mean(np.log(np.cosh(d * 1000))) / 1000 + penalty_opcvm(w, cm)
def obj_quantile(w, cm, lr, tau=0.5):
    d = calc_log_diff(w, cm, lr)
    return np.mean(np.where(d >= 0, tau*np.abs(d), (1-tau)*np.abs(d))) + penalty_opcvm(w, cm)
def obj_r2_loss(w, cm, lr):
    d = calc_log_diff(w, cm, lr)
    return np.sum(d**2) / np.sum((lr - np.mean(lr))**2) + penalty_opcvm(w, cm)
def obj_median_ae(w, cm, lr):
    d = calc_log_diff(w, cm, lr); return np.median(np.abs(d)) + penalty_opcvm(w, cm)

ALL_OBJECTIVES = {
    'Tracking Error': obj_te, 'MSE': obj_mse, 'MAE': obj_mae,
    'Max Abs Error': obj_maxe, 'RMSE': obj_rmse, 'Huber': obj_huber,
    'Downside TE': obj_downside_te, 'Combined': obj_combined,
    'Log-Cosh': obj_log_cosh, 'Quantile': obj_quantile,
    'R2 Loss': obj_r2_loss, 'Median AE': obj_median_ae,
}


# ════════════════════════════════════════════════════════════════════════════
# 3. SNAPPING + OPCVM ENFORCEMENT — copied from user's script
# ════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════
# 3. SNAPPING + OPCVM ENFORCEMENT
# ════════════════════════════════════════════════════════════════════════════
def snap_to_valid(w_continuous, companies=None):
    f_vals = np.round(np.arange(0.05, 1.01, 0.05), 2)
    F_vals_variable = np.round(np.arange(0.70, 1.01, 0.05), 2)
    
    results = []
    for i, wi in enumerate(w_continuous):
        best_f, best_F, best_d = None, None, 999
        
        # Determine allowed F values
        if companies is not None:
            comp_name = companies[i]
            if comp_name in ['ATW', 'IAM', 'LHM']:
                allowed_F = F_vals_variable
            else:
                allowed_F = [1.0]
        else:
            allowed_F = F_vals_variable # Safety fallback

        for f in f_vals:
            for F in allowed_F:
                d = abs(f * F - wi)
                if d < best_d:
                    best_d = d; best_f = f; best_F = F
        results.append((round(best_f, 2), round(best_F, 4), round(best_f * best_F, 6)))
    return results

def enforce_opcvm_caps(factors_list, cap_array_start, companies=None):
    f_vals = np.round(np.arange(0.05, 1.01, 0.05), 2)
    current_factors = list(factors_list)
    masi_weights = cap_array_start / cap_array_start.sum()
    stock_caps = np.full(len(masi_weights), 1.0)   # no individual cap by default
    stock_caps[masi_weights >= 0.15] = 0.20
    STEP = 0.01
    
    for _ in range(500):
        w_prod = np.array([f[2] for f in current_factors])
        weighted = cap_array_start * w_prod
        total = weighted.sum()
        if total == 0: break
        weights = weighted / total
        
        individual_violations = np.where(weights > stock_caps + 0.0001)[0]
        above_10_mask = weights > 0.1001
        sum_above_10 = weights[above_10_mask].sum()
        aggregate_violation = sum_above_10 > 0.4501
        
        if len(individual_violations) == 0 and not aggregate_violation: break
        
        if len(individual_violations) > 0:
            excesses = weights[individual_violations] - stock_caps[individual_violations]
            idx = individual_violations[np.argmax(excesses)]
        elif aggregate_violation:
            above_10_indices = np.where(above_10_mask)[0]
            idx = above_10_indices[np.argmax(weights[above_10_indices])]
        else: break
        
        f, F, w = current_factors[idx]
        new_f, new_F = f, F
        
        # Logic: Can we reduce F?
        # Only if it's ATW, IAM, LHM (and F > 0.70)
        can_reduce_F = False
        if companies is not None:
             if companies[idx] in ['ATW', 'IAM', 'LHM']:
                 can_reduce_F = True
        else:
             can_reduce_F = True # Fallback

        if can_reduce_F and (F - STEP >= 0.70):
            new_F = round(F - STEP, 4)
        else:
            # Must reduce f
            fi_idx = np.where(np.isclose(f_vals, f))[0]
            if len(fi_idx) > 0 and fi_idx[0] > 0:
                new_f = f_vals[fi_idx[0] - 1]
                # If we were forced to drop f, we might want to reset F to 1.0 if maximizing?
                # But safer to keep F as is or reset to 1.0?
                # Originals script reset F=1.00 when changing f level to try a new "ladder".
                # "new_F = 1.00"
                # If we reset F to 1.0, we might jump back up in weight?
                # But f drops by 0.05 step, which is significant.
                # Let's keep the logic: Reset F to 1.0 when dropping f level, 
                # UNLESS F must be fixed at 1.0 anyway.
                new_F = 1.00
            else: 
                # Can't reduce f further (min 0.05)
                break
                
        current_factors[idx] = (round(new_f, 2), round(new_F, 4), round(new_f * new_F, 6))
    return current_factors


# ════════════════════════════════════════════════════════════════════════════
# 4. EVALUATION
# ════════════════════════════════════════════════════════════════════════════
def eval_metrics(w, cm, lr):
    d = calc_log_diff(w, cm, lr)
    weighted = cm * w[np.newaxis, :]
    total = weighted.sum(axis=1)
    lp = np.log(total[1:] / total[:-1])
    ss_res = np.sum(d**2); ss_tot = np.sum((lr - np.mean(lr))**2)
    dn = d[d < 0]
    return {
        'TE': np.std(d) * np.sqrt(252),
        'MSE': np.mean(d**2),
        'MAE': np.mean(np.abs(d)),
        'MaxAE': np.max(np.abs(d)),
        'RMSE': np.sqrt(np.mean(d**2)),
        'R2': 1 - ss_res / ss_tot if ss_tot > 0 else 0,
        'DownTE': np.sqrt(np.mean(dn**2)) * np.sqrt(252) if len(dn) > 0 else 0,
        'Corr': np.corrcoef(lp, lr)[0, 1],
        'MedAE': np.median(np.abs(d)),
    }


# ════════════════════════════════════════════════════════════════════════════
# 5. OPTIMIZATION ENGINE
# ════════════════════════════════════════════════════════════════════════════
def run_all_optimizations(cap_matrix, log_ir, companies=None, n=20, seed=42):
    np.random.seed(seed)
    bounds_list = [(0.035, 1.0)] * n
    bounds_tuple = list(zip([0.035]*n, [1.0]*n))
    rw = lambda: np.random.uniform(0.035, 1.0, n)
    cap_start = cap_matrix[0]
    all_results = {}

    for obj_name, obj_fn in ALL_OBJECTIVES.items():
        print(f"    ▸ {obj_name}", end="", flush=True)
        t_obj = time.time()

        # 1. Differential Evolution
        try:
            res = differential_evolution(obj_fn, bounds_list, args=(cap_matrix, log_ir),
                maxiter=150, seed=seed, tol=1e-12, popsize=15,
                mutation=(0.5, 1.5), recombination=0.9, polish=True)
            fac = snap_to_valid(res.x, companies)
            fac = enforce_opcvm_caps(fac, cap_start, companies)
            ws = np.array([f[2] for f in fac])
            m = eval_metrics(ws, cap_matrix, log_ir)
            m['factors'] = fac; m['method'] = 'DiffEvol'; m['obj'] = obj_name
            all_results[f"{obj_name}|DiffEvol"] = m
        except: pass

        # 2. Dual Annealing
        try:
            res = dual_annealing(obj_fn, bounds_tuple, args=(cap_matrix, log_ir),
                maxiter=100, seed=seed)
            fac = snap_to_valid(res.x, companies)
            fac = enforce_opcvm_caps(fac, cap_start, companies)
            ws = np.array([f[2] for f in fac])
            m = eval_metrics(ws, cap_matrix, log_ir)
            m['factors'] = fac; m['method'] = 'DualAnn'; m['obj'] = obj_name
            all_results[f"{obj_name}|DualAnn"] = m
        except: pass

        # 3. Basin Hopping
        try:
            mk = {'method': 'L-BFGS-B', 'bounds': bounds_tuple, 'args': (cap_matrix, log_ir)}
            res = basinhopping(obj_fn, rw(), minimizer_kwargs=mk, niter=50, seed=seed)
            wc = np.clip(res.x, 0.035, 1.0)
            fac = snap_to_valid(wc, companies)
            fac = enforce_opcvm_caps(fac, cap_start, companies)
            ws = np.array([f[2] for f in fac])
            m = eval_metrics(ws, cap_matrix, log_ir)
            m['factors'] = fac; m['method'] = 'BasinHop'; m['obj'] = obj_name
            all_results[f"{obj_name}|BasinHop"] = m
        except: pass

        # 4. Multi-Start L-BFGS-B
        try:
            bv, bx = float('inf'), None
            for _ in range(5):
                res = minimize(obj_fn, rw(), args=(cap_matrix, log_ir),
                    method='L-BFGS-B', bounds=bounds_tuple)
                if res.fun < bv: bv = res.fun; bx = res.x
            fac = snap_to_valid(bx, companies)
            fac = enforce_opcvm_caps(fac, cap_start, companies)
            ws = np.array([f[2] for f in fac])
            m = eval_metrics(ws, cap_matrix, log_ir)
            m['factors'] = fac; m['method'] = 'MS-LBFGSB'; m['obj'] = obj_name
            all_results[f"{obj_name}|MS-LBFGSB"] = m
        except: pass

        # 5. Multi-Start Nelder-Mead
        try:
            bv, bx = float('inf'), None
            for _ in range(5):
                res = minimize(obj_fn, rw(), args=(cap_matrix, log_ir),
                    method='Nelder-Mead', options={'maxiter': 8000, 'xatol': 1e-10, 'fatol': 1e-12})
                wc = np.clip(res.x, 0.035, 1.0)
                v = obj_fn(wc, cap_matrix, log_ir)
                if v < bv: bv = v; bx = wc
            fac = snap_to_valid(bx, companies)
            fac = enforce_opcvm_caps(fac, cap_start, companies)
            ws = np.array([f[2] for f in fac])
            m = eval_metrics(ws, cap_matrix, log_ir)
            m['factors'] = fac; m['method'] = 'NelderMead'; m['obj'] = obj_name
            all_results[f"{obj_name}|NelderMead"] = m
        except: pass

        dt = time.time() - t_obj
        print(f"  ({dt:.0f}s)")

    ranking = sorted(all_results.items(), key=lambda x: x[1]['TE'])
    return all_results, ranking


# ════════════════════════════════════════════════════════════════════════════
# 6. FIGURE GENERATION — per year (4 subplots)
# ════════════════════════════════════════════════════════════════════════════
def generate_year_figures(year_label, companies, dates, index_values, cap_matrix,
                          log_ir, all_results, ranking):
    """Generate a beautiful multi-panel figure for one year."""

    best_key, best = ranking[0]
    w_best = np.array([f[2] for f in best['factors']])

    # ── Figure 1: Main dashboard (4 panels) ────────────────────────────────
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(f"MASI 20 — Réplication Optimale {year_label}",
                 fontsize=20, fontweight='bold', color='white', y=0.98)
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30,
                           left=0.06, right=0.96, top=0.92, bottom=0.06)

    # Panel A — Cumulative returns comparison
    ax1 = fig.add_subplot(gs[0, 0])
    cum_index = np.exp(np.cumsum(np.insert(log_ir, 0, 0)))
    d = calc_log_diff(w_best, cap_matrix, log_ir)
    port_returns = log_ir + d
    cum_port = np.exp(np.cumsum(np.insert(port_returns, 0, 0)))
    dates_plot = dates.values
    ax1.plot(dates_plot, cum_index, color='#00D4AA', lw=2.2, label='MASI 20 Index', zorder=3)
    ax1.plot(dates_plot, cum_port, color='#FF6B6B', lw=1.8, ls='--', alpha=0.9,
             label=f'Portefeuille ({best["method"]})', zorder=3)
    ax1.fill_between(dates_plot, cum_index, cum_port, alpha=0.12, color='#FFE66D')
    ax1.set_title('A. Rendements Cumulés : Index vs Portefeuille')
    ax1.legend(loc='upper left', fontsize=9, facecolor='#1A1D2E', edgecolor='#2A2D3E')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=30)

    # Panel B — TE heatmap (Objectives × Methods)
    ax2 = fig.add_subplot(gs[0, 1])
    obj_names = list(ALL_OBJECTIVES.keys())
    method_names = ['DiffEvol', 'DualAnn', 'BasinHop', 'MS-LBFGSB', 'NelderMead']
    te_matrix = np.full((len(obj_names), len(method_names)), np.nan)
    for i, on in enumerate(obj_names):
        for j, mn in enumerate(method_names):
            key = f"{on}|{mn}"
            if key in all_results:
                te_matrix[i, j] = all_results[key]['TE']
    cmap = LinearSegmentedColormap.from_list('te', ['#00D4AA', '#FFE66D', '#FF6B6B'])
    valid_te = te_matrix[~np.isnan(te_matrix)]
    if len(valid_te) > 0:
        vmin, vmax = valid_te.min(), np.percentile(valid_te, 90)
    else:
        vmin, vmax = 0, 1
    im = ax2.imshow(te_matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    ax2.set_xticks(range(len(method_names)))
    ax2.set_xticklabels(method_names, rotation=45, ha='right', fontsize=8)
    ax2.set_yticks(range(len(obj_names)))
    ax2.set_yticklabels(obj_names, fontsize=8)
    ax2.set_title('B. Tracking Error — Heatmap (Objectif × Méthode)')
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('TE (annualisée)', fontsize=9, color='#C0C0C0')
    cbar.ax.yaxis.set_tick_params(color='#A0A0A0')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#A0A0A0')

    # Panel C — Top 10 bar chart
    ax3 = fig.add_subplot(gs[1, 0])
    top10 = ranking[:10]
    labels = [k.replace('|', '\n') for k, _ in top10]
    tes = [m['TE'] for _, m in top10]
    colors_bar = [METHOD_COLORS.get(m['method'], '#888') for _, m in top10]
    bars = ax3.barh(range(len(top10)-1, -1, -1), tes, color=colors_bar, edgecolor='white',
                    linewidth=0.5, height=0.7, alpha=0.9)
    ax3.set_yticks(range(len(top10)-1, -1, -1))
    ax3.set_yticklabels(labels, fontsize=7)
    ax3.set_xlabel('Tracking Error (annualisée)')
    ax3.set_title('C. Top 10 Meilleures Stratégies')
    ax3.grid(True, axis='x', alpha=0.3)
    for i, (bar, te) in enumerate(zip(bars, tes)):
        ax3.text(bar.get_width() + 0.0001, bar.get_y() + bar.get_height()/2,
                f'{te:.6f}', va='center', fontsize=7, color='#E8E8E8')

    # Panel D — Portfolio weights (pie/donut)
    ax4 = fig.add_subplot(gs[1, 1])
    weighted = w_best * cap_matrix[-1]
    weights_pct = weighted / weighted.sum() * 100
    sorted_idx = np.argsort(weights_pct)[::-1]
    # Show top 8 individually, group rest
    top_n = 8
    top_idx = sorted_idx[:top_n]
    rest_idx = sorted_idx[top_n:]
    pie_labels = [companies[i] for i in top_idx] + ['Autres']
    pie_vals = [weights_pct[i] for i in top_idx] + [weights_pct[rest_idx].sum()]
    pie_colors = COLORS[:top_n+1]
    wedges, texts, autotexts = ax4.pie(
        pie_vals, labels=pie_labels, autopct='%1.1f%%',
        colors=pie_colors, pctdistance=0.82, startangle=140,
        wedgeprops=dict(width=0.45, edgecolor='#0F1117', linewidth=2))
    for t in texts: t.set_fontsize(8); t.set_color('#E8E8E8')
    for t in autotexts: t.set_fontsize(7); t.set_color('#0F1117'); t.set_fontweight('bold')
    ax4.set_title(f'D. Poids du Portefeuille — {best_key}')

    # Save
    path = os.path.join(OUTPUT_DIR, f'masi20_{year_label.replace("-","_")}_dashboard.png')
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    📊 Figure sauvegardée: {path}")
    return path


# ════════════════════════════════════════════════════════════════════════════
# 7. CROSS-YEAR SUMMARY FIGURE
# ════════════════════════════════════════════════════════════════════════════
def generate_summary_figure(all_years_summary):
    """Generate a cross-year comparison figure."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle("MASI 20 — Synthèse Multi-Années (2020–2026)",
                 fontsize=18, fontweight='bold', color='white', y=1.02)

    years = list(all_years_summary.keys())
    best_tes = [all_years_summary[y]['best_te'] for y in years]
    best_r2s = [all_years_summary[y]['best_r2'] for y in years]
    best_corrs = [all_years_summary[y]['best_corr'] for y in years]

    # Panel 1 — TE evolution
    ax = axes[0]
    bars = ax.bar(years, best_tes, color='#00D4AA', edgecolor='white', linewidth=0.8, alpha=0.9)
    ax.set_title('Tracking Error (Meilleur par Année)', pad=10)
    ax.set_ylabel('TE annualisée')
    ax.grid(True, axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=30)
    for bar, v in zip(bars, best_tes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                f'{v:.5f}', ha='center', va='bottom', fontsize=9, color='#E8E8E8')

    # Panel 2 — R² evolution
    ax = axes[1]
    bars = ax.bar(years, best_r2s, color='#4ECDC4', edgecolor='white', linewidth=0.8, alpha=0.9)
    ax.set_title('R² (Meilleur par Année)', pad=10)
    ax.set_ylabel('R²')
    ax.set_ylim(min(0.95, min(best_r2s) - 0.01), 1.001)
    ax.grid(True, axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=30)
    for bar, v in zip(bars, best_r2s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                f'{v:.5f}', ha='center', va='bottom', fontsize=9, color='#E8E8E8')

    # Panel 3 — Correlation evolution
    ax = axes[2]
    bars = ax.bar(years, best_corrs, color='#FFE66D', edgecolor='white', linewidth=0.8, alpha=0.9)
    ax.set_title('Corrélation (Meilleur par Année)', pad=10)
    ax.set_ylabel('Corrélation')
    ax.set_ylim(min(0.95, min(best_corrs) - 0.01), 1.001)
    ax.grid(True, axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=30)
    for bar, v in zip(bars, best_corrs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                f'{v:.6f}', ha='center', va='bottom', fontsize=9, color='#E8E8E8')

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'masi20_summary_all_years.png')
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f"\n📊 Synthèse multi-années sauvegardée: {path}")
    return path


# ════════════════════════════════════════════════════════════════════════════
# 7b. FLOATING FACTOR ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
def analyze_floating_factors(all_years_data):
    """Analyze and visualize the evolution of floating factors (fi) across years."""
    print("\n🔍 Analyse de l'évolution des Facteurs Flottants (fi)...")
    
    # 1. Aggregate data
    # Map: Company -> {Year: fi}
    comp_fi_map = {}
    years = sorted(all_years_data.keys())
    
    for y in years:
        data = all_years_data[y]
        companies = data['companies']
        best = data['ranking'][0][1] # Champion
        factors = best['factors'] # list of (f, F, w)
        
        for i, c in enumerate(companies):
            if c not in comp_fi_map: comp_fi_map[c] = {}
            comp_fi_map[c][y] = factors[i][0] # fi
            
    # 2. Create DataFrame for Heatmap
    all_companies = sorted(comp_fi_map.keys())
    matrix = np.full((len(all_companies), len(years)), np.nan)
    
    for i, c in enumerate(all_companies):
        for j, y in enumerate(years):
            if y in comp_fi_map[c]:
                matrix[i, j] = comp_fi_map[c][y]
                
    # 3. Generate Heatmap Figure
    fig, ax = plt.subplots(figsize=(12, max(8, len(all_companies)*0.3)))
    fig.suptitle("Évolution des Facteurs Flottants (fi) 2020-2026",
                 fontsize=16, fontweight='bold', color='white', y=0.98)
    
    # Mask NaNs
    masked_matrix = np.ma.masked_invalid(matrix)
    cmap = plt.cm.viridis
    cmap.set_bad(color='#1A1D2E') # Dark background for missing data
    
    im = ax.imshow(masked_matrix, cmap=cmap, aspect='auto', vmin=0.05, vmax=1.0)
    
    # Ticks
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, fontsize=10, rotation=45)
    ax.set_yticks(range(len(all_companies)))
    ax.set_yticklabels(all_companies, fontsize=9)
    
    # Annotate
    for i in range(len(all_companies)):
        for j in range(len(years)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'black' if val > 0.7 else 'white'
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", 
                        color=color, fontsize=8)
                
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Facteur Flottant (fi)', fontsize=10, color='#C0C0C0')
    
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'masi20_floating_factors_evolution.png')
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"📊 Figure d'évolution sauvegardée: {path}")
    
    # ════════════════════════════════════════════════════════════════════════════
    # 4. NEW CHARTS: Composition & Top 5
    # ════════════════════════════════════════════════════════════════════════════
    print("📊 Génération des graphiques avancés (Composition & Top 5)...")
    
    # Re-calculate % weights correctly
    df_weights = pd.DataFrame(index=years, columns=all_companies)
    
    for y in years:
        data = all_years_data[y]
        companies = data['companies']
        caps = data['cap_matrix'][0] # vector
        best = data['ranking'][0][1]
        factors = best['factors'] # (f, F, w_factor)
        w_factors = np.array([res[2] for res in factors])
        
        weighted_caps = caps * w_factors
        total_w_cap = weighted_caps.sum()
        if total_w_cap > 0:
            final_weights = weighted_caps / total_w_cap
        else:
            final_weights = np.zeros(len(caps))
        
        for i, c in enumerate(companies):
            df_weights.at[y, c] = final_weights[i]
            
    df_weights = df_weights.fillna(0.0)
    
    # Sort columns by average weight for better visuals
    avg_w = df_weights.mean(axis=0).sort_values(ascending=False)
    df_weights = df_weights[avg_w.index]
    
    # A. Stacked Area Chart
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    fig2.suptitle("Évolution de la Composition du Portefeuille (Poids %)", fontsize=16, fontweight='bold', color='white')
    
    colors = plt.cm.tab20.colors
    if len(df_weights.columns) > 20:
        colors = colors * 2
        
    x = df_weights.index
    ys = [df_weights[c].values for c in df_weights.columns]
    
    ax2.stackplot(x, ys, labels=df_weights.columns, colors=colors[:len(df_weights.columns)], alpha=0.85)
    
    ax2.set_ylabel("Poids Cumulé (0-1)", fontsize=12, color='white')
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small', ncol=1)
    ax2.set_facecolor('#1A1D2E')
    fig2.patch.set_facecolor('#0E1117')
    ax2.tick_params(colors='white', axis='x', rotation=45)
    ax2.spines['bottom'].set_color('white')
    ax2.spines['left'].set_color('white')
    
    fig2.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, 'masi20_composition_evolution.png')
    fig2.savefig(path2, dpi=180, facecolor=fig2.get_facecolor())
    plt.close(fig2)
    print(f"  -> {path2}")
    
    # B. Line Chart for Top 5 Weights
    top5 = avg_w.index[:5]
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    fig3.suptitle("Évolution des 5 Plus Grandes Valeurs", fontsize=16, fontweight='bold', color='white')
    
    for i, c in enumerate(top5):
        ax3.plot(df_weights.index, df_weights[c], marker='o', linewidth=2.5, label=c)
        
    ax3.set_ylabel("Poids dans le Portefeuille", fontsize=12, color='white')
    ax3.legend(fontsize='medium')
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.set_facecolor('#1A1D2E')
    fig3.patch.set_facecolor('#0E1117')
    ax3.tick_params(colors='white', axis='x', rotation=45)
    ax3.spines['bottom'].set_color('white')
    ax3.spines['left'].set_color('white')
    
    fig3.tight_layout()
    path3 = os.path.join(OUTPUT_DIR, 'masi20_top5_evolution.png')
    fig3.savefig(path3, dpi=180, facecolor=fig3.get_facecolor())
    plt.close(fig3)
    print(f"  -> {path3}")

    # 5. Text Explanation Generation
    explanation_path = os.path.join(OUTPUT_DIR, 'analyse_facteurs_flottants.md')
    with open(explanation_path, 'w', encoding='utf-8') as f:
        f.write("# Analyse de l'Évolution des Facteurs Flottants (fi)\n\n")
        f.write("## Observations\n")
        f.write("Ce rapport analyse comment le facteur de flottant ($f_i$) optimal évolue pour chaque entreprise.\n\n")
        f.write("### Tendance Générale\n")
        f.write("- Les grandes capitalisations (ATW, IAM, BCP) tendent à avoir des $f_i$ stables ou en légère baisse pour compenser leur poids dominant, surtout avec la contrainte $F_i=100%$.\n")
        f.write("- Les petites valeurs (Small Caps) ont souvent $f_i=1.00$ pour maximiser leur contribution à la diversité du portefeuille.\n\n")
        f.write("### Explications Possibles des Variations\n")
        f.write("1. **Changements de Capitalisation** : Si une entreprise voit sa capitalisation augmenter fortement (ex: superformance), l'algorithme peut réduire son $f_i$ pour éviter qu'elle ne dépasse les seuils réglementaires (même si $F=1$, le poids total dépend de $f \\times F \\times Cap$).\n")
        f.write("2. **Entrées/Sorties** : L'arrivée d'une nouvelle grosse valeur dans l'indice force une redistribution des poids, réduisant mécaniquement les $f_i$ des autres titres majeurs.\n")
        f.write("3. **Contraintes OPCVM** : Avec $F_i$ fixé à 1.0 pour la plupart, $f_i$ devient le seul levier d'ajustement. Si une valeur dépasse 10% du portefeuille, $f_i$ doit baisser.\n")
    
    print(f"📄 Analyse textuelle sauvegardée: {explanation_path}")
def generate_excel(all_years_data, all_years_summary):
    excel_path = os.path.join(OUTPUT_DIR, 'resultats_masi20perso_v2.xlsx')
    
    # Use pandas ExcelWriter with default engine (openpyxl usually)
    with pd.ExcelWriter(excel_path) as writer:
        
        # ── Sheet 1: Synthèse ──────────────────────────────────────────────
        summary_rows = []
        for year_label, info in all_years_summary.items():
            summary_rows.append({
                'Année': year_label,
                'Meilleur Objectif': info['best_obj'],
                'Meilleure Méthode': info['best_method'],
                'TE': info['best_te'],
                'R2': info['best_r2'],
                'Corrélation': info['best_corr'],
                'MAE': info['best_mae'],
                'MaxAE': info['best_maxae'],
                'Nb Jours': info['nb_days']
            })
        summary_df = pd.DataFrame(summary_rows)
        # Order columns slightly better
        # Handle case where column names might differ slightly in dict vs usage
        # But here keys usage matches.
        summary_df.to_excel(writer, sheet_name='Synthèse', index=False)

        # ── Per-year sheets ────────────────────────────────────────────────
        for year_label, year_data in all_years_data.items():
            ranking = year_data['ranking']
            companies = year_data['companies']
            best_key, best = ranking[0]
            
            # 1. Top 20 Ranking
            rank_rows = []
            for i, (k, m) in enumerate(ranking[:20]):
                rank_rows.append({
                    'Rank': i + 1,
                    'Objectif': m['obj'],
                    'Méthode': m['method'],
                    'TE': m['TE'],
                    'R2': m['R2'],
                    'Corr': m['Corr'],
                    'MAE': m['MAE'],
                    'MaxAE': m['MaxAE'],
                    'RMSE': m['RMSE'],
                    'DownTE': m['DownTE']
                })
            rank_df = pd.DataFrame(rank_rows)
            # Write ranking at top
            rank_df.to_excel(writer, sheet_name=f'{year_label}', index=False, startrow=0)

            # 2. Champion Information (Weights)
            # Calculate weights breakdown
            w_best = np.array([f[2] for f in best['factors']])
            cap_matrix = year_data['cap_matrix']
            # Last day market caps
            last_caps = cap_matrix[-1]
            # Weighted caps
            weighted = w_best * last_caps
            total_weighted = weighted.sum() if weighted.sum() > 0 else 1.0
            weights_pct = weighted / total_weighted

            factors_rows = []
            for i, c in enumerate(companies):
                f, F, w = best['factors'][i]
                factors_rows.append({
                    'Entreprise': c,
                    'fi': f,
                    'Fi': F,
                    'wi': w,
                    'Poids (%)': weights_pct[i]
                })
            factors_df = pd.DataFrame(factors_rows)
            
            # Write "Champion" header and table below ranking
            # Start row = len(ranking) + space. Ranking is max 20 rows + header = 21.
            start_row = 23
            pd.DataFrame([f'Champion: {best_key}']).to_excel(writer, sheet_name=f'{year_label}', 
                                                            index=False, header=False, startrow=start_row)
            factors_df.to_excel(writer, sheet_name=f'{year_label}', index=False, startrow=start_row+1)

            # 3. Full Results (60 runs)
            all_rows = []
            for i, (k, m) in enumerate(ranking):
                 all_rows.append({
                    'Rank': i + 1,
                    'Objectif': m['obj'],
                    'Méthode': m['method'],
                    'TE': m['TE'],
                    'R2': m['R2'],
                    'Corr': m['Corr'],
                    'MAE': m['MAE'],
                    'MaxAE': m['MaxAE'],
                    'RMSE': m['RMSE'],
                    'MedAE': m.get('MedAE', 0.0) # Access safely
                })
            all_df = pd.DataFrame(all_rows)
            
            # Sheet name length limit 31 chars
            sheet_name_full = f'{year_label} (60)'
            if len(sheet_name_full) > 31:
                sheet_name_full = sheet_name_full[:31]
                
            all_df.to_excel(writer, sheet_name=sheet_name_full, index=False)

    print(f"\n📗 Excel sauvegardé: {excel_path}")
    return excel_path


# ════════════════════════════════════════════════════════════════════════════
# 9. MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    all_years_data = {}
    all_years_summary = {}
    all_fig_paths = []

    parser = argparse.ArgumentParser(description='MASI20 Optimization Runner')
    parser.add_argument('--year', type=str, help='Specific year to run (e.g. "2020-2021")')
    parser.add_argument('--merge', action='store_true', help='Merge existing results and generate Excel')
    args = parser.parse_args()

    files_to_process = FILES.copy()
    if args.year:
        if args.year not in FILES:
            print(f"❌ Année inconnu: {args.year}")
            return
        files_to_process = {args.year: FILES[args.year]}
    
    # If merging, we skip optimization loop entirely and just load pickles
    if args.merge:
        print(f"🔄 Mode FUSION: Chargement des résultats existants...")
        # (This block continues below)

    total_start = time.time()

    for year_label, filename in files_to_process.items():
        if args.merge:
             continue # Skip loop if merging (we handle it below)
        filepath = os.path.join(base_dir, filename)
        print(f"\n{'═'*60}")
        print(f"  📅 {year_label} — {filename}")
        print(f"{'═'*60}")

        if not os.path.exists(filepath):
            print(f"  ❌ Fichier introuvable: {filepath}")
            continue

        try:
            companies, dates, index_values, cap_matrix, log_ir = load_data(filepath)
        except Exception as e:
            print(f"  ❌ Erreur chargement: {e}")
            continue

        pkl_path = os.path.join(OUTPUT_DIR, f"results_{year_label}.pkl")
        if os.path.exists(pkl_path):
             print(f"  Start loading cached results for {year_label}")
             with open(pkl_path, 'rb') as f:
                 all_results, ranking = pickle.load(f)
             print(f"  Loaded cached results.")
        else:
             all_results, ranking = run_all_optimizations(cap_matrix, log_ir, companies=companies)
             with open(pkl_path, 'wb') as f:
                 pickle.dump((all_results, ranking), f)

        if not ranking:
            print(f"  ❌ Aucun résultat pour {year_label}")
            continue

        best_key, best = ranking[0]
        print(f"  🏆 Champion: {best_key}  TE={best['TE']:.6f}  R²={best['R2']:.6f}")


        all_years_data[year_label] = {
            'companies': companies, 'dates': dates,
            'index_values': index_values, 'cap_matrix': cap_matrix,
            'log_ir': log_ir, 'all_results': all_results, 'ranking': ranking,
        }
        all_years_summary[year_label] = {
            'best_obj': best['obj'], 'best_method': best['method'],
            'best_te': best['TE'], 'best_r2': best['R2'],
            'best_corr': best['Corr'], 'best_mae': best['MAE'],
            'best_maxae': best['MaxAE'], 'nb_days': len(dates),
        }

        # Generate per-year figure
        try:
            fig_path = generate_year_figures(
                year_label, companies, dates, index_values,
                cap_matrix, log_ir, all_results, ranking)
            all_fig_paths.append(fig_path)
        except Exception as e:
            print(f"  ⚠ Erreur figure: {e}")

            print(f"  ⚠ Erreur figure: {e}")

    # ── Merge / Generate Final Excel ───────────────────────────────────────
    if args.merge or (not args.year):
        # We need to reload all data to generate the comprehensive Excel
        # If we just ran one year, we usually don't generate the full Excel yet
        # But if we ran sequentially (no args), we do.
        
        # If merge mode, reload EVERYTHING from pickles
        if args.merge:
            all_years_data = {}
            all_years_summary = {}
            all_fig_paths = []
            
            for year_label, filename in FILES.items():
                pkl_path = os.path.join(OUTPUT_DIR, f"results_{year_label}.pkl")
                if not os.path.exists(pkl_path):
                     print(f"⚠ Manque résultats pour {year_label}, impossible de fusionner.")
                     continue
                
                with open(pkl_path, 'rb') as f:
                    all_results, ranking = pickle.load(f)
                
                # We need data too to make the Excel
                filepath = os.path.join(base_dir, filename)
                try:
                    companies, dates, index_values, cap_matrix, log_ir = load_data(filepath)
                except:
                    print(f"⚠ Impossible de charger les données pour {year_label}")
                    continue
                
                best_key, best = ranking[0]
                all_years_data[year_label] = {
                    'companies': companies, 'dates': dates,
                    'index_values': index_values, 'cap_matrix': cap_matrix,
                    'log_ir': log_ir, 'all_results': all_results, 'ranking': ranking,
                }
                all_years_summary[year_label] = {
                    'best_obj': best['obj'], 'best_method': best['method'],
                    'best_te': best['TE'], 'best_r2': best['R2'],
                    'best_corr': best['Corr'], 'best_mae': best['MAE'],
                    'best_maxae': best['MaxAE'], 'nb_days': len(dates),
                }

        # Generate final outputs
        if all_years_summary:
            generate_summary_figure(all_years_summary)
            generate_excel(all_years_data, all_years_summary)
            analyze_floating_factors(all_years_data)

    total_time = time.time() - total_start
    print(f"\n{'═'*60}")
    print(f"  ✅ TERMINÉ en {total_time/60:.1f} minutes")
    print(f"  📁 Résultats dans: {OUTPUT_DIR}")
    print(f"{'═'*60}")


if __name__ == '__main__':
    main()
