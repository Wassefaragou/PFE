# -*- coding: utf-8 -*-

import time, warnings, os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LassoCV
from sklearn.covariance import LedoitWolf

warnings.filterwarnings('ignore')

# ── Paramètres par défaut ──
TRAIN_DAYS = 63
TEST_DAYS  = 15
REBAL_DAYS = 15

LASSO_CV_FOLDS = 5
LASSO_MAX_ITER = 5000
LASSO_TOL      = 1e-4
LASSO_SEED     = 42

DE_N_RESTARTS    = 5
DE_MAX_ITER      = 300
DE_STRATEGY      = 'best1bin'
DE_MUTATION      = (0.4, 1.2)
DE_RECOMBINATION = 0.8
DE_TOL           = 1e-9
DE_SEED_BASE     = 42

TRANSACTION_COST_BPS = 10


def normalize_ticker(value):
    ticker = str(value).strip()
    if not ticker or ticker.lower() == 'nan':
        return ''

    ticker = ' '.join(ticker.split()).upper()
    for suffix in (' MC EQUITY', ' EQUITY'):
        if ticker.endswith(suffix):
            ticker = ticker[:-len(suffix)]
            break
    return ticker.strip()


def safe_corr(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) < 2 or len(y) < 2:
        return 0.0
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0

    corr = np.corrcoef(x, y)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0


def safe_beta(x, y, default=1.0):
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) < 2 or len(y) < 2:
        return default

    var_y = np.var(y)
    if not np.isfinite(var_y) or var_y <= 1e-12:
        return default

    beta = np.cov(x, y)[0, 1] / var_y
    return float(beta) if np.isfinite(beta) else default


def project_capped_simplex(v, upper, tol=1e-12, max_iter=100):
    v = np.asarray(v, dtype=float)
    n = len(v)

    if upper is None:
        total = v.sum()
        if total <= 0:
            return np.ones(n) / n
        return v / total

    if upper <= 0 or n * upper < 1.0 - tol:
        raise ValueError(f"Plafond de poids infaisable: K={n}, max_weight={upper:.6f}")

    lo = np.min(v) - upper
    hi = np.max(v)

    for _ in range(max_iter):
        tau = 0.5 * (lo + hi)
        w = np.clip(v - tau, 0.0, upper)
        s = w.sum()

        if abs(s - 1.0) <= tol:
            return w
        if s > 1.0:
            lo = tau
        else:
            hi = tau

    w = np.clip(v - 0.5 * (lo + hi), 0.0, upper)
    residual = 1.0 - w.sum()
    if abs(residual) > 1e-10:
        if residual > 0:
            free_idx = np.where(w < upper - 1e-12)[0]
        else:
            free_idx = np.where(w > 1e-12)[0]

        if len(free_idx) > 0:
            w[free_idx] += residual / len(free_idx)
            w = np.clip(w, 0.0, upper)

    return w


def params_to_weights(params, max_weight=None):
    arr = np.asarray(params, dtype=float)
    is_vector = arr.ndim == 1
    arr2d = arr.reshape(1, -1) if is_vector else arr

    shifted = arr2d - np.max(arr2d, axis=1, keepdims=True)
    exp_p = np.exp(shifted)
    weights = exp_p / np.sum(exp_p, axis=1, keepdims=True)

    if max_weight is not None:
        weights = np.vstack([project_capped_simplex(row, max_weight) for row in weights])

    return weights[0] if is_vector else weights


def optimize_weights_capped(X_sel, y_train, target_beta=None, max_weight=None):
    K = X_sel.shape[1]
    if max_weight is None:
        raise ValueError("max_weight est requis pour l'optimisation plafonnee.")
    if K * max_weight < 1.0 - 1e-12:
        raise ValueError(f"Plafond de poids infaisable: K={K}, max_weight={max_weight:.6f}")

    t0 = time.time()
    var_y = np.var(y_train, ddof=1) if len(y_train) > 1 else 0.0
    bounds = [(0.0, max_weight)] * K
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

    def obj(w):
        port_ret = X_sel @ w
        std_diff = np.std(port_ret - y_train)
        penalty = 0.0
        if target_beta is not None and var_y > 0:
            beta = safe_beta(port_ret, y_train, default=0.0)
            penalty += 10.0 * (beta - target_beta) ** 2
        return std_diff + penalty

    starts = [project_capped_simplex(np.ones(K) / K, max_weight)]
    rng = np.random.default_rng(DE_SEED_BASE)
    for _ in range(max(DE_N_RESTARTS, 5) - 1):
        starts.append(project_capped_simplex(rng.random(K), max_weight))

    best_w = starts[0]
    best_fun = obj(best_w)
    best_res = None

    for idx, x0 in enumerate(starts):
        res = minimize(
            obj,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max(DE_MAX_ITER, 200), 'ftol': DE_TOL, 'disp': False},
        )

        candidate_w = res.x if getattr(res, 'x', None) is not None else x0
        if candidate_w is None or len(candidate_w) != K:
            candidate_w = x0

        candidate_w = np.clip(candidate_w, 0.0, max_weight)
        total = candidate_w.sum()
        if total <= 0:
            candidate_w = x0
        else:
            candidate_w = project_capped_simplex(candidate_w / total, max_weight)

        candidate_fun = obj(candidate_w)
        feasible = (
            abs(candidate_w.sum() - 1.0) <= 1e-8
            and np.all(candidate_w >= -1e-10)
            and np.all(candidate_w <= max_weight + 1e-8)
        )

        if feasible and candidate_fun < best_fun:
            best_fun = candidate_fun
            best_w = candidate_w
            best_res = (idx, res)

    best_restart = best_res[0] if best_res is not None else 0
    best_result = best_res[1] if best_res is not None else None

    de_info = {
        'n_restarts': len(starts),
        'popsize': len(starts),
        'maxiter': max(DE_MAX_ITER, 200),
        'strategy': 'SLSQP capped simplex',
        'mutation': '',
        'recombination': 0.0,
        'tolerance': DE_TOL,
        'bounds': f'(0.0, {max_weight:.6f}) + sum(w)=1',
        'polish': True,
        'elapsed': time.time() - t0,
        'obj_value': best_fun,
        'best_restart': best_restart,
        'seed': DE_SEED_BASE,
        'converged': bool(best_result.success) if best_result is not None else True,
        'message': best_result.message if best_result is not None else 'Projected feasible start',
        'n_iterations': int(best_result.nit) if best_result is not None and hasattr(best_result, 'nit') else 0,
        'n_fev': int(best_result.nfev) if best_result is not None and hasattr(best_result, 'nfev') else 0,
        'convergence_curve': [],
    }
    return best_w, de_info


def prepare_data(df):
    """
    Prend un DataFrame brut (1ère col = date, 2ème col = indice, reste = actions).
    Retourne un dict compatible avec la logique rolling.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    idx_name = df.columns[1] if len(df.columns) > 1 else "Indice"
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
    df = df.dropna(subset=[df.columns[0]])
    df = df.set_index(df.columns[0]).sort_index()
    df = df[~df.index.duplicated(keep='first')]
    df = df.ffill().bfill().fillna(0.0)

    idx_vals = df.iloc[:, 0].astype(float).values
    cap_mat  = df.iloc[:, 1:].astype(float).values
    dates    = df.index
    tickers  = df.columns[1:].tolist()

    with np.errstate(divide='ignore', invalid='ignore'):
        lr_idx = np.zeros(len(idx_vals))
        lr_idx[1:] = np.log(idx_vals[1:] / idx_vals[:-1])
        lr_idx = np.nan_to_num(lr_idx, nan=0.0, posinf=0.0, neginf=0.0)
        lr_stk = np.zeros_like(cap_mat)
        lr_stk[1:] = np.log(cap_mat[1:] / cap_mat[:-1])
        lr_stk = np.clip(np.nan_to_num(lr_stk, nan=0.0, posinf=0.0, neginf=0.0), -0.5, 0.5)

    comps = [normalize_ticker(t) for t in tickers]
    return {
        'companies': comps, 'tickers': tickers, 'dates': dates,
        'index_values': idx_vals, 'stock_values': cap_mat,
        'log_returns_index': lr_idx, 'log_returns_stocks': lr_stk,
        'index_name': idx_name,
    }


def select_lasso_cv(X_train, y_train, K):
    t0 = time.time()
    lasso_cv = LassoCV(
        cv=LASSO_CV_FOLDS, positive=True, fit_intercept=False,
        max_iter=LASSO_MAX_ITER, tol=LASSO_TOL,
        random_state=LASSO_SEED, n_alphas=100,
    )
    lasso_cv.fit(X_train, y_train)
    alpha_cv = lasso_cv.alpha_
    scores = lasso_cv.coef_.copy()
    sel_idx = np.argsort(np.abs(scores))[-K:]

    ranks = np.zeros(len(scores))
    for rank, idx in enumerate(np.argsort(np.abs(scores))[::-1]):
        ranks[idx] = rank + 1

    lasso_info = {
        'alpha_method': 'LassoCV (5-fold CV)',
        'alpha_value': alpha_cv,
        'cv_folds': LASSO_CV_FOLDS,
        'max_iter': LASSO_MAX_ITER,
        'tolerance': LASSO_TOL,
        'seed': LASSO_SEED,
        'n_alphas': 100,
        'elapsed': time.time() - t0,
        'n_nonzero': int(np.sum(np.abs(scores) > 1e-8)),
    }
    return sel_idx, scores, ranks, lasso_info


def select_meta_score(X_train, y_train, stock_vals_train_end, K):
    n_stocks = X_train.shape[1]
    scores = np.zeros(n_stocks)
    
    corr_list = []
    r2_list = []
    beta_list = []
    
    var_y = np.var(y_train)
    
    for i in range(n_stocks):
        x_i = X_train[:, i]
        if np.std(x_i) == 0 or np.std(y_train) == 0:
            corr = 0.0
            beta = 0.0
            r2 = 0.0
        else:
            corr = safe_corr(x_i, y_train)
            beta = safe_beta(x_i, y_train, default=0.0) if var_y > 0 else 0.0
            r2 = corr ** 2
        
        corr_list.append(corr)
        r2_list.append(r2)
        beta_list.append(beta)
        
    corr_arr = np.array(corr_list)
    r2_arr = np.array(r2_list)
    beta_arr = np.array(beta_list)
    
    # "Les termes avec * doivent être normalisés entre 0 et 1." (w_i^*)
    w_raw = stock_vals_train_end / (np.sum(stock_vals_train_end) + 1e-8)
    if np.max(w_raw) == np.min(w_raw):
        w_norm = np.zeros_like(w_raw)
    else:
        w_norm = (w_raw - np.min(w_raw)) / (np.max(w_raw) - np.min(w_raw))
        
    for i in range(n_stocks):
        scores[i] = 0.4 * corr_arr[i] + 0.3 * r2_arr[i] + 0.2 * w_norm[i] + 0.1 * (1 - abs(beta_arr[i] - 1))
        
    sel_idx = np.argsort(scores)[-K:]
    
    ranks = np.zeros(n_stocks)
    for rank, idx in enumerate(np.argsort(scores)[::-1]):
        ranks[idx] = rank + 1
        
    score_info = {
        'method': 'Meta Score',
    }
    return sel_idx, scores, ranks, score_info

def select_beta(X_train, y_train, K):
    n_stocks = X_train.shape[1]
    var_y = np.var(y_train)
    beta_list = []
    for i in range(n_stocks):
        x_i = X_train[:, i]
        if np.std(x_i) == 0 or var_y == 0:
            beta = 0.0
        else:
            beta = safe_beta(x_i, y_train, default=0.0)
        beta_list.append(beta)
    
    beta_arr = np.array(beta_list)
    # Selectionner les betas les plus proches de 1
    scores = -np.abs(beta_arr - 1)
    sel_idx = np.argsort(scores)[-K:]
    
    ranks = np.zeros(n_stocks)
    for rank, idx in enumerate(np.argsort(scores)[::-1]):
        ranks[idx] = rank + 1
        
    return sel_idx, beta_arr, ranks, {'method': 'Sélection Beta (proche de 1)'}

def load_factors(required=False):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'F_f.xlsx')

    try:
        if not os.path.exists(path):
            if required:
                raise FileNotFoundError(f"Fichier de facteurs introuvable: {path}")
            return {}

        df_f = pd.read_excel(path)
        if 'Valeur' not in df_f.columns:
            raise ValueError("Le fichier F_f.xlsx doit contenir une colonne 'Valeur'.")

        factors = {}
        for _, row in df_f.iterrows():
            ticker_key = normalize_ticker(row['Valeur'])
            if not ticker_key:
                continue
            factors[ticker_key] = {
                'flottant': float(row.get('Flottant', 1.0) if pd.notna(row.get('Flottant', 1.0)) else 1.0),
                'plafonnement': float(row.get('Plafonnement', 1.0) if pd.notna(row.get('Plafonnement', 1.0)) else 1.0)
            }

        if required and not factors:
            raise ValueError(f"Aucune valeur exploitable trouvée dans {path}.")
        return factors
    except Exception:
        if required:
            raise
        return {}

def select_corr_cap(X_train, y_train, stock_vals_train_window, tickers, K):
    n_stocks = X_train.shape[1]
    corr_list = []
    for i in range(n_stocks):
        x_i = X_train[:, i]
        if np.std(x_i) == 0 or np.std(y_train) == 0:
            corr = 0.0
        else:
            corr = safe_corr(x_i, y_train)
        corr_list.append(corr)
    
    corr_arr = np.array(corr_list)
    
    # New logic: Cap = Flottant * Plafonnement * Moyenne_Periode
    factors_dict = load_factors(required=True)
    avg_prices = np.mean(stock_vals_train_window, axis=0) # Moyenne dans la période
    
    adj_caps = np.zeros(n_stocks)
    for i in range(n_stocks):
        t = normalize_ticker(tickers[i])
        f_info = factors_dict.get(t, {'flottant': 1.0, 'plafonnement': 1.0})
        adj_caps[i] = f_info['flottant'] * f_info['plafonnement'] * avg_prices[i]
    
    # Normalisation par la somme (somme de même)
    sum_adj = np.sum(adj_caps) + 1e-8
    cap_rel = adj_caps / sum_adj
    
    # Normalisation pour combiner (0 à 1)
    # On garde la normalisation min-max pour que Corr et Cap aient le même poids relatif
    corr_norm = (corr_arr - corr_arr.min()) / (corr_arr.max() - corr_arr.min() + 1e-8)
    cap_norm = (cap_rel - cap_rel.min()) / (cap_rel.max() - cap_rel.min() + 1e-8)
    
    scores = corr_norm * cap_norm
    sel_idx = np.argsort(scores)[-K:]
    
    ranks = np.zeros(n_stocks)
    for rank, idx in enumerate(np.argsort(scores)[::-1]):
        ranks[idx] = rank + 1
        
    return sel_idx, scores, ranks, {'method': 'Sélection Corr * Adjusted Cap'}

def select_corr_float(X_train, y_train, tickers, K):
    n_stocks = X_train.shape[1]
    corr_list = []
    for i in range(n_stocks):
        x_i = X_train[:, i]
        if np.std(x_i) == 0 or np.std(y_train) == 0:
            corr = 0.0
        else:
            corr = safe_corr(x_i, y_train)
        corr_list.append(corr)
    
    corr_arr = np.array(corr_list)
    
    factors_dict = load_factors(required=True)
    float_arr = np.array([factors_dict.get(normalize_ticker(t), {'flottant': 1.0})['flottant'] for t in tickers])
    
    # Normalisation pour combiner (0 à 1)
    corr_norm = (corr_arr - corr_arr.min()) / (corr_arr.max() - corr_arr.min() + 1e-8)
    float_norm = (float_arr - float_arr.min()) / (float_arr.max() - float_arr.min() + 1e-8)
    
    scores = corr_norm * float_norm
    sel_idx = np.argsort(scores)[-K:]
    
    ranks = np.zeros(n_stocks)
    for rank, idx in enumerate(np.argsort(scores)[::-1]):
        ranks[idx] = rank + 1
        
    return sel_idx, scores, ranks, {'method': 'Sélection Corr * Flottant'}

def select_lw_forward(X_train, y_train, K):
    """
    Sélection forward greedy basée sur la tracking error variance estimée via Ledoit-Wolf.
    """
    t0 = time.time()
    n_stocks = X_train.shape[1]
    
    # 1. Préparation des données (X + y)
    Z = np.column_stack([X_train, y_train])
    
    # 2. Estimation Ledoit-Wolf
    lw = LedoitWolf().fit(Z)
    Sigma = lw.covariance_
    shrinkage = lw.shrinkage_
    
    Sigma_xx = Sigma[:-1, :-1]
    sigma_xy = Sigma[:-1, -1]
    sigma_yy = Sigma[-1, -1]
    
    # 3. Sélection Greedy Forward
    selected = []
    remaining = list(range(n_stocks))
    
    for _ in range(K):
        best_obj = np.inf
        best_idx = -1
        
        for i in remaining:
            S = selected + [i]
            # Extraction sous-blocs
            S_idx = np.ix_(S, S)
            Sigma_SS = Sigma_xx[S_idx]
            sigma_SI = sigma_xy[S]
            
            # Résoudre Sigma_SS * w = sigma_SI (Approximation linéaire des poids)
            # Ajout d'une petite régularisation pour la stabilité numérique
            try:
                reg = 1e-8 * np.eye(len(S))
                w = np.linalg.solve(Sigma_SS + reg, sigma_SI)
                w = np.clip(w, 0, None)
                if w.sum() > 0:
                    w = w / w.sum()
                else:
                    w = np.ones(len(S)) / len(S)
            except:
                w = np.ones(len(S)) / len(S)
                
            # Calcul J = w' Sigma_SS w - 2 w' sigma_SI + sigma_yy
            obj = w.T @ Sigma_SS @ w - 2.0 * (w.T @ sigma_SI) + sigma_yy
            
            if obj < best_obj:
                best_obj = obj
                best_idx = i
        
        if best_idx != -1:
            selected.append(best_idx)
            remaining.remove(best_idx)
        else:
            break
            
    sel_idx = np.array(selected)
    
    # Rangs et Scores artificiels pour l'affichage
    scores = np.zeros(n_stocks)
    for i, idx in enumerate(selected):
        scores[idx] = K - i # Score décroissant selon l'ordre d'entrée
        
    ranks = np.zeros(n_stocks)
    for rank, idx in enumerate(selected):
        ranks[idx] = rank + 1
        
    lw_info = {
        'method': 'Ledoit-Wolf Forward Greedy',
        'shrinkage': shrinkage,
        'elapsed': time.time() - t0,
        'sigma_yy': sigma_yy
    }
    
    return sel_idx, scores, ranks, lw_info


def optimize_weights_de_robust(X_sel, y_train, target_beta=None, max_weight=None):
    K = X_sel.shape[1]
    if max_weight is not None:
        return optimize_weights_capped(X_sel, y_train, target_beta=target_beta, max_weight=max_weight)

    t0_global = time.time()
    best_w     = np.ones(K) / K
    best_fun   = np.inf
    best_info  = {}
    convergence_best = []
    
    var_y = np.var(y_train, ddof=1) if len(y_train) > 1 else 0.0

    bnd_min, bnd_max = -6.0, 6.0
    mut_min, mut_max = DE_MUTATION if isinstance(DE_MUTATION, (tuple, list)) else (0.5, 1.0)
    pop_size = K * 10

    for restart in range(DE_N_RESTARTS):
        seed_i = DE_SEED_BASE + restart
        np.random.seed(seed_i)
        
        pop = np.random.uniform(bnd_min, bnd_max, (pop_size, K))
        
        def eval_pop(p):
            w = params_to_weights(p, max_weight=max_weight)
            port_ret = w @ X_sel.T
            diff = port_ret - y_train
            std_diff = np.std(diff, axis=1)
            
            penalty = 0.0
            if target_beta is not None and var_y > 0:
                port_ret_centered = port_ret - np.mean(port_ret, axis=1, keepdims=True)
                y_train_centered = y_train - np.mean(y_train)
                covs = np.sum(port_ret_centered * y_train_centered, axis=1) / (len(y_train) - 1)
                betas = covs / var_y
                penalty += 10.0 * (betas - target_beta)**2
                
            return std_diff + penalty
            
        fitness = eval_pop(pop)
        best_idx = np.argmin(fitness)
        best_cost = fitness[best_idx]
        
        convergence_curve = []
        
        for it in range(DE_MAX_ITER):
            r1 = np.random.randint(0, pop_size, pop_size)
            r2 = np.random.randint(0, pop_size, pop_size)
            
            F = np.random.uniform(mut_min, mut_max, (pop_size, 1))
            mutant = pop[best_idx] + F * (pop[r1] - pop[r2])
            
            cross_points = np.random.rand(pop_size, K) < DE_RECOMBINATION
            force_idx = np.random.randint(0, K, pop_size)
            cross_points[np.arange(pop_size), force_idx] = True
            
            trial = np.where(cross_points, mutant, pop)
            trial = np.clip(trial, bnd_min, bnd_max)
            
            trial_fitness = eval_pop(trial)
            
            improved = trial_fitness < fitness
            pop[improved] = trial[improved]
            fitness[improved] = trial_fitness[improved]
            
            curr_best_idx = np.argmin(fitness)
            if fitness[curr_best_idx] < best_cost:
                best_cost = fitness[curr_best_idx]
                best_idx = curr_best_idx
                
            convergence_curve.append(best_cost)
            
        best_x = pop[best_idx]
        
        # Polish step (facultatif mais identique au polish de Scipy)
        def single_obj(x):
            w = params_to_weights(x, max_weight=max_weight)
            port_ret = X_sel @ w
            std_diff = np.std(port_ret - y_train)
            penalty = 0.0
            if target_beta is not None and var_y > 0:
                cov = np.cov(port_ret, y_train)[0, 1]
                beta = cov / var_y
                penalty += 10.0 * (beta - target_beta)**2
            return std_diff + penalty
            
        res_min = minimize(single_obj, best_x, bounds=[(bnd_min, bnd_max)]*K, method='L-BFGS-B', tol=DE_TOL)
        if res_min.success and res_min.fun < best_cost:
            best_x = res_min.x
            best_cost = res_min.fun
            if convergence_curve:
                convergence_curve[-1] = best_cost

        if best_cost < best_fun:
            best_fun = best_cost
            best_w = params_to_weights(best_x, max_weight=max_weight)
            convergence_best = list(convergence_curve)
            best_info = {
                'best_restart': restart, 'seed': seed_i,
                'converged': True, 'message': 'Custom Vectorized DE converged',
                'n_iterations': DE_MAX_ITER, 'n_fev': DE_MAX_ITER * pop_size,
            }

    elapsed = time.time() - t0_global
    de_info = {
        'n_restarts': DE_N_RESTARTS, 'popsize': pop_size,
        'maxiter': DE_MAX_ITER, 'strategy': 'Custom Vectorized Numpy DE',
        'mutation': str(DE_MUTATION), 'recombination': DE_RECOMBINATION,
        'tolerance': DE_TOL, 'bounds': '(-6.0, 6.0)',
        'polish': True, 'elapsed': elapsed,
        'obj_value': best_fun, **best_info,
        'convergence_curve': convergence_best,
    }
    return best_w, de_info


def compute_rebal_schedule(data, train_days=TRAIN_DAYS, test_days=TEST_DAYS, rebal_days=REBAL_DAYS):
    """Compute the full rebalancing schedule with train/test details.
    Returns a list of dicts with info about each rebalancing window."""
    total_len = len(data['dates'])
    schedule = []
    cursor = train_days
    rebal_num = 0

    while cursor < total_len:
        train_start = cursor - train_days
        train_end = cursor
        test_start = cursor
        test_end = min(cursor + test_days, total_len)
        if test_end <= test_start:
            break
        # Skip if not enough data for a full test window
        if (test_end - test_start) < test_days:
            break
        rebal_num += 1
        schedule.append({
            'rebal_num': rebal_num,
            'train_start_idx': train_start,
            'train_end_idx': train_end,
            'test_start_idx': test_start,
            'test_end_idx': test_end,
            'train_start_date': data['dates'][train_start].strftime('%Y-%m-%d'),
            'train_end_date': data['dates'][train_end - 1].strftime('%Y-%m-%d'),
            'test_start_date': data['dates'][test_start].strftime('%Y-%m-%d'),
            'test_end_date': data['dates'][test_end - 1].strftime('%Y-%m-%d'),
            'train_days': train_end - train_start,
            'test_days': test_end - test_start,
        })
        cursor += rebal_days

    # Compute unused data points
    if schedule:
        last = schedule[-1]
        used_end = last['test_end_idx']  # last data point used
        unused_end = total_len - used_end  # points after last test window
    else:
        used_end = 0
        unused_end = total_len

    return {
        'schedule': schedule,
        'total_data_points': total_len,
        'used_data_points': used_end,
        'unused_data_points': unused_end,
        'n_rebals': len(schedule),
    }

def greedy_round_l2(w_target, prices, V, max_iter=10000, eps=1e-8, penalize_cash=True):
    """
    Convertit des poids cibles en quantités entières d'actions via une heuristique Greedy
    basée sur la minimisation de l'erreur L2 des poids réalisés.
    """
    n_assets = len(w_target)
    n = np.zeros(n_assets, dtype=int)
    
    # Étape 1 : Initialisation par arrondi inférieur
    for i in range(n_assets):
        if prices[i] > 0:
            s_star = (w_target[i] * V) / prices[i]
            n[i] = int(np.floor(s_star))
            
    invested = np.sum(n * prices)
    cash = V - invested
    
    def calc_J(n_test, cash_test):
        w_real = (n_test * prices) / V
        w_cash = cash_test / V
        J = np.sum((w_real - w_target)**2)
        if penalize_cash:
            J += w_cash**2
        return J

    J_current = calc_J(n, cash)
    
    # Étape 2 : Boucle greedy
    iteration = 0
    while iteration < max_iter:
        best_i = -1
        best_J = J_current
        
        # Trouver les titres "achetables"
        affordable = np.where(prices <= cash)[0]
        if len(affordable) == 0:
            break
            
        # Tester l'achat d'une action pour chaque titre achetable
        for i in affordable:
            n_test = n.copy()
            n_test[i] += 1
            cash_test = cash - prices[i]
            
            J_test = calc_J(n_test, cash_test)
            
            if J_test < best_J - eps:
                best_J = J_test
                best_i = i
            elif abs(J_test - best_J) <= eps and best_i != -1: # Tie break
                # Choisir le plus sous-pondéré
                w_real_test = (n[i] * prices[i]) / V
                w_real_best = (n[best_i] * prices[best_i]) / V
                diff_test = w_target[i] - w_real_test
                diff_best = w_target[best_i] - w_real_best
                
                if diff_test > diff_best:
                    best_i = i
                    best_J = J_test
                elif diff_test == diff_best:
                    # Moins cher
                    if prices[i] < prices[best_i]:
                        best_i = i
                        best_J = J_test
        
        if best_i == -1:
             break # Aucune amélioration possible
             
        # Appliquer le meilleur choix
        n[best_i] += 1
        cash -= prices[best_i]
        J_current = best_J
        iteration += 1

    w_real = (n * prices) / V
    return n, cash, w_real, J_current



def run_rolling(data, K=None, selected_indices=None, selection_method='lasso', weight_method='de', manual_weights=None, progress_callback=None, max_rebals=None, selected_rebals=None, target_beta=None, train_days=TRAIN_DAYS, test_days=TEST_DAYS, rebal_days=REBAL_DAYS, max_weight=None):
    """Mode backtest complet avec rolling window."""
    np.random.seed(42)
    total_len = len(data['dates'])
    lr_stocks = data['log_returns_stocks']
    lr_index  = data['log_returns_index']

    rebal_records, weight_records, oos_records = [], [], []
    hyper_records, convergence_records = [], []
    all_port_returns, all_idx_returns = [], []

    prev_weights = None
    prev_sel_idx = None
    rebal_count  = 0
    cursor = train_days

    # Count total rebalancings for progress
    total_rebals = 0
    c = train_days
    while c < total_len:
        ts = c; te = min(c + test_days, total_len)
        if te <= ts:
            break
        # Skip incomplete test windows
        if (te - ts) < test_days:
            break
        total_rebals += 1
        c += rebal_days

    # Limit if max_rebals is specified
    if max_rebals is not None and max_rebals < total_rebals:
        total_rebals = max_rebals

    # If selected_rebals provided, adjust total for progress
    if selected_rebals is not None:
        selected_set = set(selected_rebals)
        total_rebals = len(selected_set)
    else:
        selected_set = None

    rebal_global_num = 0  # tracks the global rebal number (1-based)
    executed_count = 0    # tracks how many we actually ran

    while cursor < total_len:
        train_start = cursor - train_days
        train_end   = cursor
        test_start  = cursor
        test_end    = min(cursor + test_days, total_len)
        if test_end <= test_start:
            break
        # Skip if not enough data for a full test window
        if (test_end - test_start) < test_days:
            break

        rebal_global_num += 1

        # Stop if we've reached the max number of rebalancings (legacy)
        if max_rebals is not None and selected_set is None and executed_count >= max_rebals:
            break

        # Skip if not in selected set
        if selected_set is not None and rebal_global_num not in selected_set:
            cursor += rebal_days
            continue

        X_train = lr_stocks[train_start:train_end]
        y_train = lr_index[train_start:train_end]
        X_test  = lr_stocks[test_start:test_end]
        y_test  = lr_index[test_start:test_end]

        executed_count += 1
        rebal_date = data['dates'][test_start]

        if progress_callback:
            progress_callback(executed_count, total_rebals, rebal_date)

        if selection_method == 'manual':
            sel_idx = np.array(selected_indices)
            scores = np.zeros(lr_stocks.shape[1])
            ranks = np.zeros(lr_stocks.shape[1])
            lasso_info = {'elapsed': 0.0}
        elif selection_method == 'lasso':
            sel_idx, scores, ranks, lasso_info = select_lasso_cv(X_train, y_train, K)
        elif selection_method == 'score':
            stock_vals_end = data['stock_values'][train_end - 1]
            sel_idx, scores, ranks, lasso_info = select_meta_score(X_train, y_train, stock_vals_end, K)
            lasso_info['elapsed'] = 0.0
        elif selection_method == 'beta':
            sel_idx, scores, ranks, lasso_info = select_beta(X_train, y_train, K)
            lasso_info['elapsed'] = 0.0
        elif selection_method == 'corr_cap':
            stock_vals_window = data['stock_values'][train_start:train_end]
            sel_idx, scores, ranks, lasso_info = select_corr_cap(X_train, y_train, stock_vals_window, data['tickers'], K)
            lasso_info['elapsed'] = 0.0
        elif selection_method == 'corr_float':
            sel_idx, scores, ranks, lasso_info = select_corr_float(X_train, y_train, data['tickers'], K)
            lasso_info['elapsed'] = 0.0
        elif selection_method == 'lw':
            sel_idx, scores, ranks, lasso_info = select_lw_forward(X_train, y_train, K)

        X_sel_train = X_train[:, sel_idx]
        X_sel_test  = X_test[:, sel_idx]
        if weight_method == 'manual' and selection_method == 'manual':
            weights = manual_weights
            de_info = {'elapsed': 0.0, 'obj_value': 0.0, 'n_restarts': 0, 'popsize': 0, 'maxiter': 0, 'strategy': '', 'mutation': '', 'recombination': 0, 'tolerance': 0, 'bounds': '', 'polish': False}
        else:
            weights, de_info = optimize_weights_de_robust(X_sel_train, y_train, target_beta=target_beta, max_weight=max_weight)
            
        port_ret = X_sel_test @ weights
        idx_ret  = y_test

        diff_oos  = port_ret - idx_ret
        te_window = np.std(diff_oos) if len(diff_oos) > 1 else 0.0
        cum_port  = np.sum(port_ret)
        cum_idx   = np.sum(idx_ret)

        all_port_returns.extend(port_ret.tolist())
        all_idx_returns.extend(idx_ret.tolist())

        selected = [data['companies'][i] for i in sel_idx]

        rebal_records.append({
            'Rebal #': executed_count,
            'Date Rebal': rebal_date.strftime('%Y-%m-%d'),
            'Train Début': data['dates'][train_start].strftime('%Y-%m-%d'),
            'Train Fin': data['dates'][train_end - 1].strftime('%Y-%m-%d'),
            'Test Début': data['dates'][test_start].strftime('%Y-%m-%d'),
            'Test Fin': data['dates'][test_end - 1].strftime('%Y-%m-%d'),
            'Jours Train': train_end - train_start,
            'Jours Test': test_end - test_start,
            'Titres': ', '.join(selected),
        })

        for i, idx in enumerate(sel_idx):
            weight_records.append({
                'Rebal #': executed_count,
                'Date Rebal': rebal_date.strftime('%Y-%m-%d'),
                'Titre': data['companies'][idx],
                'Poids DE (%)': weights[i] * 100,
                'Score': scores[idx],
                'Rang': int(ranks[idx]),
            })

        oos_records.append({
            'Rebal #': executed_count,
            'Date Rebal': rebal_date.strftime('%Y-%m-%d'),
            'Rend. Port DE (%)': cum_port * 100,
            'Rend. Indice (%)': cum_idx * 100,
            'Écart DE-Idx (%)': (cum_port - cum_idx) * 100,
            'TE Fenêtre (bps)': te_window * 10000,
        })

        hyper_records.append({
            'Rebal #': executed_count,
            'Date Rebal': rebal_date.strftime('%Y-%m-%d'),
            'LASSO α': lasso_info.get('alpha_value', ''),
            'LASSO Méthode': lasso_info.get('alpha_method', lasso_info.get('method', '')),
            'LASSO CV': lasso_info.get('cv_folds', ''),
            'LASSO MaxIter': lasso_info.get('max_iter', ''),
            'LASSO Tol': lasso_info.get('tolerance', ''),
            'LASSO Seed': lasso_info.get('seed', ''),
            'LASSO NonZero': lasso_info.get('n_nonzero', ''),
            'LASSO Temps (s)': round(lasso_info.get('elapsed', 0.0), 3),
            'DE Restarts': de_info['n_restarts'],
            'DE Best Restart #': de_info.get('best_restart', ''),
            'DE Seed': de_info.get('seed', ''),
            'DE Pop': de_info['popsize'],
            'DE MaxIter': de_info['maxiter'],
            'DE Stratégie': de_info['strategy'],
            'DE Mutation': de_info['mutation'],
            'DE Recombinaison': de_info['recombination'],
            'DE Tol': de_info['tolerance'],
            'DE Bornes': de_info['bounds'],
            'DE Polish': de_info['polish'],
            'DE Convergé': de_info.get('converged', ''),
            'DE Obj Finale': de_info['obj_value'],
            'DE Itérations': de_info.get('n_iterations', ''),
            'DE Évals Fn': de_info.get('n_fev', ''),
            'DE Temps (s)': round(de_info['elapsed'], 3),
            'Temps Total (s)': round(lasso_info['elapsed'] + de_info['elapsed'], 3),
        })

        for step, val in enumerate(de_info.get('convergence_curve', [])):
            convergence_records.append({
                'Rebal #': executed_count, 'Date Rebal': rebal_date.strftime('%Y-%m-%d'),
                'Iteration': step + 1, 'Obj Value': val,
            })

        prev_weights = weights.copy()
        prev_sel_idx = sel_idx.copy()
        cursor += rebal_days

    all_port = np.array(all_port_returns)
    all_idx  = np.array(all_idx_returns)
    all_diff = all_port - all_idx

    global_te    = np.std(all_diff) if len(all_diff) > 1 else 0.0
    te_list = [r['TE Fenêtre (bps)'] for r in oos_records]

    # Beta = Cov(port, idx) / Var(idx)
    if len(all_port) > 1 and np.var(all_idx) > 0:
        beta = safe_beta(all_port, all_idx, default=1.0)
    else:
        beta = 1.0

    summary = {
        'K': len(sel_idx) if 'sel_idx' in locals() else K,
        'TE Moyen (bps)': np.mean(te_list),
        'TE Médian (bps)': np.median(te_list),
        'TE Max (bps)': np.max(te_list),
        'TE Min (bps)': np.min(te_list),
        'TE Écart-type (bps)': np.std(te_list),
        'TE Global (bps)': global_te * 10000,
        'Rend. Cumulé Port (%)': np.sum(all_port) * 100,
        'Rend. Cumulé Indice (%)': np.sum(all_idx) * 100,
        'Écart Cumulé Final (%)': (np.sum(all_port) - np.sum(all_idx)) * 100,
        'Nb Rebalancements': executed_count,
        'Corrélation OOS': safe_corr(all_port, all_idx),
        'Beta': beta,
    }

    return {
        'summary': summary,
        'rebal_records': rebal_records,
        'weight_records': weight_records,
        'oos_records': oos_records,
        'hyper_records': hyper_records,
        'convergence_records': convergence_records,
        'all_port_returns': all_port,
        'all_idx_returns': all_idx,
    }


def run_simple_replication(data, K=None, selected_indices=None, selection_method='lasso', weight_method='de', manual_weights=None, progress_callback=None, target_beta=None, train_days=TRAIN_DAYS, max_weight=None):
    """Mode réplication simple — un seul portefeuille, pas de test OOS."""
    np.random.seed(42)
    total_len = len(data['dates'])
    lr_stocks = data['log_returns_stocks']
    lr_index  = data['log_returns_index']

    # Use specified train window
    train_end   = total_len
    train_start = max(0, total_len - train_days)

    X_train = lr_stocks[train_start:train_end]
    y_train = lr_index[train_start:train_end]

    if progress_callback:
        progress_callback(1, 2, "Sélection des titres...")

    if selection_method == 'manual':
        sel_idx = np.array(selected_indices)
        scores = np.zeros(lr_stocks.shape[1])
        ranks = np.zeros(lr_stocks.shape[1])
        lasso_info = {'elapsed': 0.0}
    elif selection_method == 'lasso':
        sel_idx, scores, ranks, lasso_info = select_lasso_cv(X_train, y_train, K)
    elif selection_method == 'score':
        stock_vals_end = data['stock_values'][train_end - 1]
        sel_idx, scores, ranks, lasso_info = select_meta_score(X_train, y_train, stock_vals_end, K)
        lasso_info['elapsed'] = 0.0
    elif selection_method == 'beta':
        sel_idx, scores, ranks, lasso_info = select_beta(X_train, y_train, K)
        lasso_info['elapsed'] = 0.0
    elif selection_method == 'corr_cap':
        stock_vals_window = data['stock_values'][train_start:train_end]
        sel_idx, scores, ranks, lasso_info = select_corr_cap(X_train, y_train, stock_vals_window, data['tickers'], K)
        lasso_info['elapsed'] = 0.0
    elif selection_method == 'corr_float':
        sel_idx, scores, ranks, lasso_info = select_corr_float(X_train, y_train, data['tickers'], K)
        lasso_info['elapsed'] = 0.0
    elif selection_method == 'lw':
        sel_idx, scores, ranks, lasso_info = select_lw_forward(X_train, y_train, K)

    if progress_callback:
        progress_callback(2, 2, "Calcul des poids...")

    X_sel_train = X_train[:, sel_idx]
    
    if weight_method == 'manual' and selection_method == 'manual':
        weights = manual_weights
        de_info = {'elapsed': 0.0, 'obj_value': 0.0}
    else:
        weights, de_info = optimize_weights_de_robust(X_sel_train, y_train, target_beta=target_beta, max_weight=max_weight)

    # Quality metrics on training window
    port_ret_train = X_sel_train @ weights
    diff_train = port_ret_train - y_train
    te_train = np.std(diff_train) * 10000
    corr_train = safe_corr(port_ret_train, y_train)

    selected = [data['companies'][i] for i in sel_idx]

    weight_records = []
    for i, idx in enumerate(sel_idx):
        weight_records.append({
            'Titre': data['companies'][idx],
            'Poids DE (%)': weights[i] * 100,
            'Score': scores[idx],
            'Rang': int(ranks[idx]),
        })

    return {
        'selected': selected,
        'weights': weights,
        'weight_records': weight_records,
        'lasso_info': lasso_info,
        'de_info': de_info,
        'te_train_bps': te_train,
        'corr_train': corr_train,
        'train_start': data['dates'][train_start].strftime('%Y-%m-%d'),
        'train_end': data['dates'][train_end - 1].strftime('%Y-%m-%d'),
        'train_days': train_end - train_start,
        'port_ret_train': port_ret_train,
        'idx_ret_train': y_train,
    }
