# -*- coding: utf-8 -*-
import hashlib
import io
import json
import math
import os
import re
import socket
import ssl
import time
import traceback
import unicodedata
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from masi20_index_replication_engine import (
    prepare_data, run_rolling, run_simple_replication, compute_rebal_schedule,
    TRAIN_DAYS, TEST_DAYS, REBAL_DAYS, greedy_round_l2, load_factors,
    normalize_ticker, filter_replication_universe
)

APP_TITLE = "MASI20 Index Replication"
NAVIGATION_STATE_KEY = 'selected_app'
APP_ICON_URL = "https://www.google.com/s2/favicons?domain=attijariwafabank.com&sz=128"
PRICE_SOURCE_MANUAL = "Saisie manuelle"
PRICE_SOURCE_FILE = "Fichier prix"
PRICE_SOURCE_API = "Fetch API BVC"
MASI20_INDEX_CODE = "MSI20"
MASI20_INDEX_PAGE_URL = f"https://www.casablanca-bourse.com/fr/live-market/indices/{MASI20_INDEX_CODE}"
BVC_PROXY_BASE_URL = "https://www.casablanca-bourse.com/api/proxy"
EXPECTED_MASI20_TITLES = 20
BVC_MARKET_ROWS_STATE_KEY = "replication_bvc_market_rows"
BVC_MARKET_FETCHED_AT_STATE_KEY = "replication_bvc_market_fetched_at"
MASI20_TICKER_ALIASES = {
    "ADH": "DOUJA PROM ADDOHA",
    "ADI": "ALLIANCES",
    "AKT": "AKDITAL",
    "ATW": "ATTIJARIWAFA BANK",
    "BCP": "BCP",
    "BOA": "BANK OF AFRICA",
    "CFG": "CFG BANK",
    "CDM": "CDM",
    "CMA": "CIMENTS DU MAROC",
    "CSR": "COSUMAR",
    "CMG": "CMGP GROUP",
    "IAM": "ITISSALAT AL-MAGHRIB",
    "LBV": "LABEL VIE",
    "LHM": "LAFARGEHOLCIM MAROC",
    "SID": "SONASID",
    "MSA": "SODEP-Marsa Maroc",
    "JET": "JET CONTRACTORS",
    "RDS": "RESIDENCES DAR SAADA",
    "TGC": "TGCC S.A",
    "TQM": "TAQA MOROCCO",
}
MASI20_LONG_TO_SHORT = {
    normalize_ticker(long_name): short_code
    for short_code, long_name in MASI20_TICKER_ALIASES.items()
}


def infer_selection_method_label(method_info):
    info = (method_info or "").lower()
    if not info or 'manual' in info:
        return None
    if 'beta' in info:
        return "Beta"
    if 'exhaustive' in info:
        return "Exhaustive DE"
    if 'corr' in info and 'cap' in info:
        return "Corr*Cap"
    if 'corr' in info and 'float' in info:
        return "Corr*Float"
    if 'ledoit' in info or 'lw' in info:
        return "LW"
    return "Lasso"


def format_weight_records(df, method_label):
    display_df = df.copy()
    rename_map = {'Poids DE (%)': 'Poids (%)'}

    if 'Rebal #' in display_df.columns:
        rename_map['Rebal #'] = 'Rebal'
    if 'Date Rebal' in display_df.columns:
        rename_map['Date Rebal'] = 'Date'

    if method_label:
        rename_map['Score'] = f'Score {method_label}'
        rename_map['Rang'] = f'Rang {method_label}'
    else:
        drop_cols = [col for col in ('Score', 'Rang') if col in display_df.columns]
        if drop_cols:
            display_df = display_df.drop(columns=drop_cols)

    return display_df.rename(columns=rename_map)


def run_with_elapsed(func, *args, **kwargs):
    t0 = time.time()
    result = func(*args, **kwargs)
    return result, time.time() - t0


def decode_text_file(raw_bytes):
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode("utf-8", errors="replace")


def normalize_bvc_key(value):
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", ascii_text.lower()).strip("_")


def clean_bvc_title(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+MC\s+Equity$", "", text, flags=re.IGNORECASE)
    if not text or text.lower() in {"nan", "na", "none", "-"}:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text


def fetch_remote_payload(url, accept, timeout=15.0):
    request = Request(
        url,
        headers={
            "Accept": accept,
            "Referer": MASI20_INDEX_PAGE_URL,
            "User-Agent": "MASI20-Index-Replication/1.0",
        },
    )
    max_attempts = 2
    for attempt in range(max_attempts):
        try:
            with urlopen(request, timeout=timeout, context=ssl._create_unverified_context()) as response:
                return response.read()
        except HTTPError as exc:
            raise RuntimeError(f"Erreur Casablanca Bourse ({exc.code}) sur {url}.") from exc
        except URLError as exc:
            reason = getattr(exc, "reason", None)
            is_timeout = isinstance(reason, (TimeoutError, socket.timeout))
            if is_timeout and attempt + 1 < max_attempts:
                continue
            if is_timeout:
                raise RuntimeError(
                    "La Bourse de Casablanca a mis trop de temps a repondre. Reessayez dans quelques secondes."
                ) from exc
            raise RuntimeError(f"Connexion Casablanca Bourse impossible pour {url}.") from exc
        except (TimeoutError, socket.timeout) as exc:
            if attempt + 1 < max_attempts:
                continue
            raise RuntimeError(
                "La Bourse de Casablanca a mis trop de temps a repondre. Reessayez dans quelques secondes."
            ) from exc


def fetch_remote_json(url, timeout=15.0):
    raw_payload = fetch_remote_payload(
        url,
        accept="application/json, application/vnd.api+json, text/plain, */*",
        timeout=timeout,
    )
    try:
        return json.loads(raw_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Reponse JSON invalide pour {url}.") from exc


def fetch_remote_text(url, timeout=15.0):
    raw_payload = fetch_remote_payload(
        url,
        accept="text/html, application/json;q=0.9, */*;q=0.8",
        timeout=timeout,
    )
    return decode_text_file(raw_payload)


def proxy_bvc_api_url(url):
    return (
        url.replace("https://api.casablanca-bourse.com", BVC_PROXY_BASE_URL)
        .replace("http://api.casablanca-bourse.com", BVC_PROXY_BASE_URL)
    )


def extract_next_build_id(page_html):
    build_match = re.search(r'"buildId":"([^"]+)"', page_html)
    if build_match:
        return build_match.group(1)

    manifest_match = re.search(r"/_next/static/([^/]+)/_buildManifest\\.js", page_html)
    if manifest_match:
        return manifest_match.group(1)

    raise RuntimeError("Build Next.js introuvable sur la page MSI20.")


def build_market_watch_rows(page_payload):
    included = {
        item.get("id"): item
        for item in page_payload.get("included", [])
        if item.get("type") == "instrument"
    }

    rows = []
    for item in page_payload.get("data", []):
        attributes = item.get("attributes") or {}
        symbol_info = ((item.get("relationships") or {}).get("symbol") or {}).get("data") or {}
        instrument_id = symbol_info.get("id")
        instrument_attributes = (included.get(instrument_id) or {}).get("attributes") or {}
        ticker = clean_bvc_title(
            instrument_attributes.get("libelleFR")
            or instrument_attributes.get("libelleEN")
            or instrument_id
        )
        if not ticker:
            continue

        try:
            price_value = float(attributes["coursCourant"])
            cap_value = float(attributes["capitalisation"])
        except (KeyError, TypeError, ValueError):
            continue

        rows.append(
            {
                "Ticker": ticker,
                "Cours": price_value,
                "Capitalisation": cap_value,
            }
        )

    return rows


def fetch_masi20_market_snapshot(timeout=20.0):
    page_html = fetch_remote_text(MASI20_INDEX_PAGE_URL, timeout=timeout)
    build_id = extract_next_build_id(page_html)
    next_data_url = (
        f"https://www.casablanca-bourse.com/_next/data/{build_id}/fr/live-market/indices/{MASI20_INDEX_CODE}.json"
    )
    next_payload = fetch_remote_json(next_data_url, timeout=timeout)

    try:
        paragraphs = next_payload["pageProps"]["node"]["field_vactory_paragraphs"]
        composition_paragraph = next(
            paragraph
            for paragraph in paragraphs
            if (paragraph.get("field_vactory_component") or {}).get("widget_id")
            == "bourse_data_listing:index-composition"
        )
        widget_data = json.loads(composition_paragraph["field_vactory_component"]["widget_data"])
        first_page_payload = widget_data["components"][0]["collection"]["data"]
    except (KeyError, StopIteration, json.JSONDecodeError) as exc:
        raise RuntimeError("Structure inattendue pour la composition MSI20.") from exc

    rows = build_market_watch_rows(first_page_payload)
    next_page_url = (((first_page_payload.get("links") or {}).get("next") or {}).get("href"))
    visited_urls = set()

    while next_page_url and next_page_url not in visited_urls:
        visited_urls.add(next_page_url)
        proxy_url = proxy_bvc_api_url(next_page_url)
        next_page_payload = fetch_remote_json(proxy_url, timeout=timeout)
        rows.extend(build_market_watch_rows(next_page_payload))
        next_page_url = (((next_page_payload.get("links") or {}).get("next") or {}).get("href"))

    deduplicated_rows = []
    seen_titles = set()
    for row in rows:
        ticker = str(row["Ticker"])
        if ticker in seen_titles:
            continue
        seen_titles.add(ticker)
        deduplicated_rows.append(row)

    if len(deduplicated_rows) != EXPECTED_MASI20_TITLES:
        raise RuntimeError(
            f"La composition fetchee contient {len(deduplicated_rows)} titres au lieu de {EXPECTED_MASI20_TITLES}."
        )

    fetched_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    return deduplicated_rows, fetched_at


def format_fetch_timestamp(value):
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return "-"
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def build_bvc_price_lookup(rows):
    lookup = {}
    for row in rows:
        ticker = clean_bvc_title(row.get("Ticker"))
        try:
            price = float(row.get("Cours"))
        except (TypeError, ValueError):
            continue

        normalized_title = normalize_ticker(ticker)
        candidate_keys = {normalized_title}
        short_code = MASI20_LONG_TO_SHORT.get(normalized_title)
        if short_code:
            candidate_keys.add(short_code)

        normalized_key = normalize_bvc_key(ticker)
        for alias_code, alias_name in MASI20_TICKER_ALIASES.items():
            if normalize_bvc_key(alias_name) == normalized_key:
                candidate_keys.add(alias_code)
                candidate_keys.add(normalize_ticker(alias_name))
                break

        for key in candidate_keys:
            if key:
                lookup[key] = price
    return lookup


def run():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON_URL,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if st.button("\u2b05\ufe0f Retour a l'accueil", key='back_to_home_index_replication'):
        st.session_state[NAVIGATION_STATE_KEY] = None
        st.rerun()


    # ══════════════════════════════════════════════════════════════
    # PREMIUM CSS
    # ══════════════════════════════════════════════════════════════
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] nav[aria-label="Page navigation"],
        section[data-testid="stSidebar"] div[data-testid="stSidebarNav"],
        section[data-testid="stSidebar"] ul[data-testid="stSidebarNavItems"],
        section[data-testid="stSidebar"] [data-testid="stSidebarNavSeparator"],
        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


    # ══════════════════════════════════════════════════════════════
    # PLOTLY CHART THEME
    # ══════════════════════════════════════════════════════════════
    CHART_LAYOUT = dict(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.02)',
        font=dict(family='Inter, sans-serif', color='#94a3b8', size=12),
        title_font=dict(size=16, color='#e2e8f0', family='Inter, sans-serif'),
        legend=dict(
            bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8', size=11),
            orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
        ),
        xaxis=dict(gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.06)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.06)'),
        margin=dict(l=50, r=30, t=60, b=50),
    )

    COLORS = {
        'port': '#a78bfa',   # purple
        'idx':  '#f87171',   # red
        'diff': '#818cf8',   # indigo
        'bar1': '#6c63ff',
        'bar2': '#00d4aa',
    }


    # ══════════════════════════════════════════════════════════════
    # HELPER FUNCTIONS
    # ══════════════════════════════════════════════════════════════

    def generate_preview_data():
        dates = pd.bdate_range('2024-01-02', periods=5, freq='B')
        idx = [10014.90, 10010.75, 10030.18, 10075.88, 10068.85]
        d = {'Date': dates.strftime('%Y-%m-%d'), 'Indice': idx}
        sample_tickers = ['ATW MC Equity', 'IAM MC Equity', 'BCP MC Equity', 'BOA MC Equity']
        for i, t in enumerate(sample_tickers):
            d[t] = [100 + i*15 + j*0.5 for j in range(5)]
        return pd.DataFrame(d)


    def get_uploaded_file_signature(uploaded):
        payload = uploaded.getvalue()
        return uploaded.name, len(payload), hashlib.md5(payload).hexdigest()


    def sync_uploaded_file_state(uploaded):
        prev_signature = st.session_state.get('uploaded_signature')

        if uploaded is None:
            if prev_signature is not None:
                clear_results()
                reset_window_inputs()
                del st.session_state['uploaded_signature']
            return

        curr_signature = get_uploaded_file_signature(uploaded)
        if prev_signature != curr_signature:
            clear_results()
            reset_window_inputs()
            st.session_state['uploaded_signature'] = curr_signature

        uploaded.seek(0)


    @st.cache_data(show_spinner=False)
    def load_uploaded_dataset(file_name, payload):
        buffer = io.BytesIO(payload)
        raw_df = pd.read_csv(buffer) if file_name.lower().endswith('.csv') else pd.read_excel(buffer)
        return raw_df, prepare_data(raw_df)




    def create_export_excel(result, mode):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            if mode == 'backtest':
                pd.DataFrame(result['rebal_records']).to_excel(writer, 'Rebalancements', index=False)
                pd.DataFrame(result['weight_records']).to_excel(writer, 'Poids', index=False)
                pd.DataFrame(result['oos_records']).to_excel(writer, 'Résultats OOS', index=False)
                s = result['summary']
                rows = [(k, round(s[k], 4) if isinstance(s[k], float) else s[k]) for k in s]
                pd.DataFrame(rows, columns=['Indicateur', 'Valeur']).to_excel(writer, 'Résumé', index=False)
                n = len(result['all_port_returns'])
                pd.DataFrame({
                    'Jour #': range(1, n+1), 'Rend. Port DE': result['all_port_returns'],
                    'Rend. Indice': result['all_idx_returns'],
                    'Cumul DE': np.cumsum(result['all_port_returns']),
                    'Cumul Indice': np.cumsum(result['all_idx_returns']),
                }).to_excel(writer, 'Perf Quotidienne', index=False)
            else:
                pd.DataFrame(result['weight_records']).to_excel(writer, 'Portefeuille', index=False)
                info_rows = [
                    ('Période', f"{result['train_start']} → {result['train_end']}"),
                    ('Jours entraînement', result['train_days']),
                    ('TE entraînement (bps)', round(result['te_train_bps'], 2)),
                    ('Corrélation entraînement', round(result['corr_train'], 6)),
                    ('LASSO α', result['lasso_info'].get('alpha_value', '')),
                    ('DE Obj Finale', result['de_info']['obj_value']),
                    ('DE Temps (s)', round(result['de_info']['elapsed'], 2)),
                ]
                pd.DataFrame(info_rows, columns=['Indicateur', 'Valeur']).to_excel(writer, 'Résumé', index=False)
        output.seek(0)
        return output


    def metric_card(label, value, glow='purple'):
        return f'''<div class="glass-metric">
            <div class="glow glow-{glow}"></div>
            <h4>{label}</h4>
            <div class="val">{value}</div>
        </div>'''


    def render_greedy_l2_section(df_w_selected, data, start_date_str, end_date_str, key_prefix=''):
        st.markdown("---")
        st.markdown("### 🧮 Ordres")
        
        c1, c2 = st.columns(2)
        with c1:
            V = st.number_input("Budget (MAD)", min_value=0.0, value=100000.0, step=1000.0, key=f"{key_prefix}_V")
        with c2:
            price_mode = st.radio(
                "Prix",
                [PRICE_SOURCE_MANUAL, PRICE_SOURCE_FILE, PRICE_SOURCE_API],
                horizontal=True,
                key=f"{key_prefix}_pmode",
            )
            
        tickers = df_w_selected['Titre'].tolist()
        weights_target = df_w_selected['Poids DE (%)'].values / 100.0
        
        if price_mode == PRICE_SOURCE_FILE:
            price_file = st.file_uploader("Fichier prix (CSV/XLS/XLSX)", type=['csv', 'xlsx', 'xls'], key=f"{key_prefix}_pfile")
            imported_prices = {}
            if price_file is not None:
                try:
                    price_file.seek(0)
                    if price_file.name.endswith('.csv'):
                        pdf = pd.read_csv(price_file)
                    else:
                        pdf = pd.read_excel(price_file)
                    # lowercase everything for flexible matching
                    pdf.columns = [str(c).lower().strip() for c in pdf.columns]
                    
                    # Try to map ticker and price
                    ticker_cols = [c for c in pdf.columns if 'tick' in c or 'titre' in c or 'action' in c]
                    price_cols = [c for c in pdf.columns if 'price' in c or 'prix' in c or 'cours' in c]
                    
                    if ticker_cols and price_cols:
                        t_col = ticker_cols[0]
                        p_col = price_cols[0]
                        for _, row in pdf.iterrows():
                            ticker_key = normalize_ticker(row[t_col])
                            price_value = pd.to_numeric(row[p_col], errors='coerce')
                            if ticker_key and pd.notna(price_value):
                                imported_prices[ticker_key] = float(price_value)
                        st.success("✅ Prix importés.")
                    else:
                        st.error("❌ Le fichier doit contenir une colonne titre et une colonne prix.")
                except Exception as e:
                    st.error(f"❌ Lecture impossible : {e}")
                    
            editor_data = [{"ticker": t, "price": imported_prices.get(normalize_ticker(t), 0.0)} for t in tickers]
        elif price_mode == PRICE_SOURCE_API:
            st.caption("Récupère les cours actions via la même API Casablanca Bourse que le pricer.")

            if st.button("Fetch prix MASI20", key=f"{key_prefix}_fetch_bvc_prices", width='stretch'):
                try:
                    fetched_rows, fetched_at = fetch_masi20_market_snapshot()
                    st.session_state[BVC_MARKET_ROWS_STATE_KEY] = fetched_rows
                    st.session_state[BVC_MARKET_FETCHED_AT_STATE_KEY] = fetched_at
                    st.success(f"✅ {len(fetched_rows)} titres récupérés depuis Casablanca Bourse.")
                except Exception as exc:
                    st.error(f"❌ Fetch impossible : {exc}")

            fetched_rows = st.session_state.get(BVC_MARKET_ROWS_STATE_KEY, [])
            fetched_at = st.session_state.get(BVC_MARKET_FETCHED_AT_STATE_KEY)
            st.caption(f"Dernier fetch : {format_fetch_timestamp(fetched_at)}")

            api_prices = build_bvc_price_lookup(fetched_rows)
            editor_data = []
            matched_titles = []
            missing_titles = []
            for ticker in tickers:
                price_value = api_prices.get(normalize_ticker(ticker), 0.0)
                editor_data.append({"ticker": ticker, "price": price_value})
                if price_value > 0:
                    matched_titles.append(ticker)
                else:
                    missing_titles.append(ticker)

            if fetched_rows:
                if matched_titles:
                    st.info(
                        f"{len(matched_titles)}/{len(tickers)} titre(s) sélectionné(s) ont été pré-remplis depuis l'API."
                    )
                if missing_titles:
                    st.warning("Prix introuvables dans le fetch pour : " + ", ".join(missing_titles))
            else:
                st.info("Cliquez sur le bouton de fetch pour pré-remplir les prix depuis Casablanca Bourse.")
        else:
            editor_data = [{"ticker": t, "price": 0.0} for t in tickers]
        
        edited_df = st.data_editor(
            pd.DataFrame(editor_data),
            column_config={
                "ticker": st.column_config.TextColumn("Titre", disabled=True),
                "price": st.column_config.NumberColumn("Prix (MAD)", min_value=0.0, format="%.2f")
            },
            width='stretch',
            hide_index=True,
            key=f"{key_prefix}_deditor"
        )
        
        if st.button("Calculer les quantités", key=f"{key_prefix}_btn_calc", type="primary"):
            prices = edited_df['price'].values
            missing = edited_df[edited_df['price'] <= 0]
            
            if not missing.empty:
                st.error(f"❌ Renseignez tous les prix (> 0). Manquants : {', '.join(missing['ticker'])}")
            elif V <= np.min(prices[prices > 0]):
                 st.error("❌ Budget insuffisant pour acheter une action.")
            else:
                w_target_filtered = []
                prices_filtered = []
                tickers_filtered = []
                for t, w, p in zip(tickers, weights_target, prices):
                    w_target_filtered.append(w)
                    prices_filtered.append(p)
                    tickers_filtered.append(t)
                
                w_target_arr = np.array(w_target_filtered)
                prices_arr = np.array(prices_filtered)
                
                n, cash_rem, w_real, final_J = greedy_round_l2(w_target_arr, prices_arr, V, penalize_cash=True)
                
                invested = V - cash_rem
                cash_weight = cash_rem / V
                
                # --- CALCUL DU TRACKING ERROR ---
                new_te_str = "N/A"
                try:
                    dates_str = [d.strftime('%Y-%m-%d') for d in data['dates']]
                    s_idx = dates_str.index(start_date_str)
                    e_idx = dates_str.index(end_date_str) + 1
                    
                    sel_indices = [data['companies'].index(t) for t in tickers_filtered]
                    X_ret = data['log_returns_stocks'][s_idx:e_idx][:, sel_indices]
                    y_ret = data['log_returns_index'][s_idx:e_idx]
                    
                    new_port_ret = X_ret @ w_real
                    diff = new_port_ret - y_ret
                    new_te = np.std(diff) * 10000
                    new_te_str = f"{new_te:.2f} bps"
                except Exception as e:
                    pass
                
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Investi (MAD)", f"{invested:,.0f}")
                c2.metric("Cash (MAD)", f"{cash_rem:,.0f}")
                c3.metric("Cash (%)", f"{cash_weight*100:.2f}%")
                c4.metric("Erreur L2", f"{final_J:.6f}")
                c5.metric("TE après arrondi", new_te_str, help=f"TE calculé du {start_date_str} au {end_date_str} avec les poids réalisés.")
                
                # Affichage de la table
                res_df = pd.DataFrame({
                    "Titre": tickers_filtered,
                    "Poids cible": [f"{w*100:.2f}%" for w in w_target_arr],
                    "Prix (MAD)": prices_arr,
                    "Qté": n,
                    "Montant (MAD)": n * prices_arr,
                    "Poids réel": [f"{wr*100:.2f}%" for wr in w_real],
                    "Écart poids": [f"{(wr-wt)*100:.2f}%" for wr, wt in zip(w_real, w_target_arr)]
                })
                st.dataframe(res_df, width='stretch', hide_index=True)
                




    # ══════════════════════════════════════════════════════════════
    # DISPLAY FUNCTIONS
    # ══════════════════════════════════════════════════════════════

    def display_backtest_results(result, data):
        s = result['summary']

        # KPI row
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(metric_card('TE global', f'{s["TE Global (bps)"]:.1f} <span class="unit">bps</span>', 'purple'), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card('Corr. OOS', f'{s["Corrélation OOS"]:.4f}', 'green'), unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card('Bêta', f'{s["Beta"]:.4f}', 'blue'), unsafe_allow_html=True)
        with c4:
            st.markdown(metric_card('Rebals', f'{s["Nb Rebalancements"]}', 'pink'), unsafe_allow_html=True)

        st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)

        tabs = st.tabs(["📈 Performance", "📊 TE", "📋 Rebals", "⚖️ Poids", "📑 Résumé"])

        with tabs[0]:
            cum_port = np.cumsum(result['all_port_returns'])
            cum_idx  = np.cumsum(result['all_idx_returns'])

            fig = go.Figure()
            fig.add_trace(go.Scatter(y=cum_port, name='Portefeuille DE', line=dict(color=COLORS['port'], width=2.5),
                                     fill='tonexty' if False else None))
            fig.add_trace(go.Scatter(y=cum_idx, name='Indice', line=dict(color=COLORS['idx'], width=2.5, dash='dash')))
            fig.update_layout(**CHART_LAYOUT, height=460, title='Perf. cumulée OOS')
            st.plotly_chart(fig, width='stretch')

            cum_diff = np.cumsum(result['all_port_returns'] - result['all_idx_returns'])
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(y=cum_diff, fill='tozeroy', name='Δ Port – Idx',
                                      line=dict(color=COLORS['diff'], width=2),
                                      fillcolor='rgba(129,140,248,0.1)'))
            fig2.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.15)")
            fig2.update_layout(**CHART_LAYOUT, height=320, title='Écart cumulé')
            st.plotly_chart(fig2, width='stretch')

        with tabs[1]:
            df_oos = pd.DataFrame(result['oos_records'])
            te_vals = df_oos['TE Fenêtre (bps)'].values
            display_oos = df_oos.rename(columns={
                'Rebal #': 'Rebal',
                'Date Rebal': 'Date',
                'Rend. Port DE (%)': 'Perf. port (%)',
                'Rend. Indice (%)': 'Perf. indice (%)',
                'Écart DE-Idx (%)': 'Écart (%)',
                'TE Fenêtre (bps)': 'TE (bps)',
            })
            colors = ['#34d399' if v <= 20 else '#f87171' if v >= 40 else '#fbbf24' for v in te_vals]

            fig3 = go.Figure()
            fig3.add_trace(go.Bar(x=df_oos['Date Rebal'], y=te_vals, marker_color=colors,
                                  marker_line=dict(width=0), name='TE'))
            fig3.add_hline(y=np.mean(te_vals), line_dash="dot", line_color="#a78bfa",
                           annotation_text=f"μ = {np.mean(te_vals):.1f} bps",
                           annotation_font=dict(color='#a78bfa', size=12))
            fig3.update_layout(**CHART_LAYOUT, height=420, title='TE par fenêtre OOS',
                               bargap=0.25)
            st.plotly_chart(fig3, width='stretch')
            st.dataframe(display_oos, width='stretch', height=350)

        with tabs[2]:
            display_rebals = pd.DataFrame(result['rebal_records']).rename(columns={
                'Rebal #': 'Rebal',
                'Date Rebal': 'Date',
                'Train Début': 'Train début',
                'Train Fin': 'Train fin',
                'Test Début': 'Test début',
                'Test Fin': 'Test fin',
                'Jours Train': 'Train (j)',
                'Jours Test': 'Test (j)',
            })
            st.dataframe(display_rebals, width='stretch', height=500)

        with tabs[3]:
            df_w = pd.DataFrame(result['weight_records'])
            rebal_nums = sorted(df_w['Rebal #'].unique())

            if len(rebal_nums) > 1:
                sel_rebal = st.selectbox(
                    '🔍 Rebal',
                    options=rebal_nums,
                    index=len(rebal_nums) - 1,
                    format_func=lambda x: f"Rebal #{x}",
                )
            else:
                sel_rebal = rebal_nums[0] if rebal_nums else 1

            sel_w = df_w[df_w['Rebal #'] == sel_rebal]
            if not sel_w.empty:
                fig4 = go.Figure()
                fig4.add_trace(go.Bar(y=sel_w['Titre'], x=sel_w['Poids DE (%)'], orientation='h',
                                      name='DE', marker_color=COLORS['bar1'],
                                      marker_line=dict(width=0)))
                fig4.update_layout(**CHART_LAYOUT, height=420, barmode='group',
                                   title=f'Allocation rebal #{sel_rebal}')
                st.plotly_chart(fig4, width='stretch')

            method_info = ''
            if 'hyper_records' in result and len(result['hyper_records']) > 0:
                method_info = result['hyper_records'][0].get('LASSO Méthode', '')
            method_label = infer_selection_method_label(method_info)
            display_sel_w = format_weight_records(sel_w, method_label)

            st.dataframe(display_sel_w, width='stretch', height=350)
            
            # Injection du composant Greedy L2
            st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)
            rebal_info = next((r for r in result['rebal_records'] if r['Rebal #'] == sel_rebal), None)
            if rebal_info:
                render_greedy_l2_section(sel_w, data, rebal_info['Test Début'], rebal_info['Test Fin'], key_prefix=f'bt_{sel_rebal}')

        with tabs[4]:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown('#### 📉 TE')
                te_items = [
                    ('TE global (bps)', 'TE Global (bps)'),
                    ('TE moyen (bps)', 'TE Moyen (bps)'),
                    ('TE médian (bps)', 'TE Médian (bps)'),
                    ('TE max (bps)', 'TE Max (bps)'),
                    ('TE min (bps)', 'TE Min (bps)'),
                    ('TE écart-type (bps)', 'TE Écart-type (bps)'),
                ]
                for label, key in te_items:
                    st.metric(label, f"{s[key]:.2f}")
            with col_b:
                st.markdown('#### 💰 Perf. & stats')
                ret_items = [
                    ('Perf. port (%)', 'Rend. Cumulé Port (%)'),
                    ('Perf. indice (%)', 'Rend. Cumulé Indice (%)'),
                    ('Écart final (%)', 'Écart Cumulé Final (%)'),
                    ('Corr. OOS', 'Corrélation OOS'),
                    ('Bêta', 'Beta'),
                ]
                for label, key in ret_items:
                    fmt = f"{s[key]:.6f}" if key in ['Corrélation OOS', 'Beta'] else f"{s[key]:.4f}"
                    st.metric(label, fmt)

        # Export
        st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
        excel_data = create_export_excel(result, 'backtest')
        st.download_button("📥 Télécharger Excel", data=excel_data,
                           file_name="resultats_backtest.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           width='stretch')


    def display_simple_results(result, data):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(metric_card('TE train', f'{result["te_train_bps"]:.1f} <span class="unit">bps</span>', 'purple'), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card('Corrélation', f'{result["corr_train"]:.4f}', 'green'), unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card('Jours', f'{result["train_days"]}', 'blue'), unsafe_allow_html=True)

        st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)
        tabs = st.tabs(["📋 Portefeuille", "📈 Qualité"])

        with tabs[0]:
            df_w = pd.DataFrame(result['weight_records'])
            palette = px.colors.qualitative.Pastel[:len(df_w)]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_w['Titre'], y=df_w['Poids DE (%)'], marker_color=palette,
                                 text=[f'{v:.1f}%' for v in df_w['Poids DE (%)']], textposition='outside',
                                 textfont=dict(color='#94a3b8', size=11), marker_line=dict(width=0)))
            fig.update_layout(**CHART_LAYOUT, height=460, title='Portefeuille optimal')
            st.plotly_chart(fig, width='stretch')
            method_info = result.get('lasso_info', {}).get('method', result.get('lasso_info', {}).get('alpha_method', ''))
            method_label = infer_selection_method_label(method_info)
            display_df_w = format_weight_records(df_w, method_label)

            st.dataframe(display_df_w, width='stretch')
            
            # Injection du composant Greedy L2
            st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)
            render_greedy_l2_section(df_w, data, result['train_start'], result['train_end'], key_prefix='simp')

        with tabs[1]:
            port_ret = result['port_ret_train']
            idx_ret  = result['idx_ret_train']
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(y=np.cumsum(port_ret), name='Portefeuille', line=dict(color=COLORS['port'], width=2.5)))
            fig2.add_trace(go.Scatter(y=np.cumsum(idx_ret), name='Indice', line=dict(color=COLORS['idx'], width=2.5, dash='dash')))
            fig2.update_layout(**CHART_LAYOUT, height=420, title="Réplication sur la fenêtre train")
            st.plotly_chart(fig2, width='stretch')

        st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
        excel_data = create_export_excel(result, 'simple')
        st.download_button("📥 Télécharger Excel", data=excel_data,
                           file_name="resultats_replication_simple.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           width='stretch')


    # ══════════════════════════════════════════════════════════════
    # SIDEBAR
    # ══════════════════════════════════════════════════════════════
    def clear_results():
        for key in ['result', 'data', 'mode', 'elapsed']:
            if key in st.session_state:
                del st.session_state[key]

    def clamp_int(value, min_value, max_value):
        value = int(value)
        return max(min_value, min(value, max_value))

    def clamp_optional_int(value, min_value, max_value):
        if value is None or max_value is None:
            return None
        return clamp_int(value, min_value, max_value)

    def reset_window_inputs():
        for key in ['window_max_days', 'train_days_input', 'test_days_input', 'rebal_days_input']:
            if key in st.session_state:
                del st.session_state[key]

    train_days = TRAIN_DAYS
    test_days = TEST_DAYS
    rebal_days = REBAL_DAYS

    with st.sidebar:
        st.markdown(f"""
        <div class="sidebar-logo" style="display: flex; align-items: center; margin-bottom: 20px;">
            <img src="{APP_ICON_URL}" style="width: 40px; margin-right: 12px; border-radius: 4px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
            <div class="logo-text" style="font-size: 1.1rem; font-weight: 600;">{APP_TITLE}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        mode = st.radio(
            "**Mode**",
            ["Backtest", "Simple"],
            on_change=clear_results
        )

        st.markdown("---")
        st.markdown("##### Fenetres")
        
        window_max_days = st.session_state.get('window_max_days')
        has_data_window = window_max_days is not None
        if has_data_window:
            window_max_days = max(1, int(window_max_days))

        train_default = clamp_optional_int(st.session_state.get('train_days_input'), 1, window_max_days)
        if st.session_state.get('train_days_input') != train_default:
            st.session_state['train_days_input'] = train_default

        train_days = st.number_input(
            "Train (jours)",
            min_value=1 if has_data_window else None,
            max_value=window_max_days if has_data_window else None,
            value=train_default,
            step=1,
            key='train_days_input',
            placeholder="NaN" if not has_data_window else f"1 a {window_max_days}",
            disabled=not has_data_window,
            on_change=clear_results
        )
        
        if "Backtest" in mode:
            remaining_days = None
            if has_data_window and train_days is not None:
                remaining_days = window_max_days - int(train_days)

            oos_ready = remaining_days is not None and remaining_days > 15
            oos_max_days = remaining_days if oos_ready else None

            test_default = clamp_optional_int(st.session_state.get('test_days_input'), 15, oos_max_days)
            if st.session_state.get('test_days_input') != test_default:
                st.session_state['test_days_input'] = test_default

            rebal_default = clamp_optional_int(st.session_state.get('rebal_days_input'), 15, oos_max_days)
            if st.session_state.get('rebal_days_input') != rebal_default:
                st.session_state['rebal_days_input'] = rebal_default

            test_days = st.number_input(
                "Test OOS (jours)",
                min_value=15 if oos_ready else None,
                max_value=oos_max_days if oos_ready else None,
                value=test_default,
                step=1,
                key='test_days_input',
                placeholder="NaN" if not oos_ready else f"15 a {oos_max_days}",
                disabled=not oos_ready,
                on_change=clear_results
            )
            rebal_days = st.number_input(
                "Rebal (jours)",
                min_value=15 if oos_ready else None,
                max_value=oos_max_days if oos_ready else None,
                value=rebal_default,
                step=1,
                key='rebal_days_input',
                placeholder="NaN" if not oos_ready else f"15 a {oos_max_days}",
                disabled=not oos_ready,
                on_change=clear_results
            )


    # ══════════════════════════════════════════════════════════════
    # MAIN CONTENT
    # ══════════════════════════════════════════════════════════════

    # Hero
    st.markdown(f"""
    <div class="hero-wrap">
        <h1 class="hero-title">{APP_TITLE}</h1>
    </div>
    """, unsafe_allow_html=True)

    # ── Section 1: Upload ──
    st.markdown("""
    <div class="section-card">
        <div class="section-label"><span class="num">1</span> DONNÉES</div>
        <div class="section-heading">Importer le fichier</div>
        <p style="color:#94a3b8; font-size:0.9rem;">
            Formats : <strong>CSV</strong>, <strong>XLS</strong>, <strong>XLSX</strong>.
            Les log-rendements sont calculés automatiquement.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("❓ Format du fichier"):
        st.markdown(f"""
        <div style="color: #cbd5e1; font-size: 0.95rem; margin-bottom: 1rem;">
            <strong>Structure minimale :</strong><br><br>
            <ul style="margin-top: 0;">
                <li><strong>Colonne 1</strong> : date</li>
                <li><strong>Colonne 2</strong> : indice</li>
                <li><strong>Colonnes 3+</strong> : titres</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(generate_preview_data(), width='stretch', hide_index=True)

    uploaded = st.file_uploader("Choisir un fichier", type=['csv', 'xlsx', 'xls'],
                                label_visibility='collapsed')
    sync_uploaded_file_state(uploaded)

    if uploaded is not None:
        try:
            raw_df, data = load_uploaded_dataset(uploaded.name, uploaded.getvalue())
            st.markdown('<div class="alert-success">✅ Fichier chargé.</div>', unsafe_allow_html=True)

            with st.expander("👁️ Aperçu brut"):
                st.dataframe(raw_df.head(15), width='stretch')

            if raw_df.shape[1] < 3:
                st.markdown(f'<div class="alert-error">❌ Le fichier doit avoir <strong>au moins 3 colonnes</strong> : Date, Indice et 1 titre. Actuel : <strong>{raw_df.shape[1]}</strong>.</div>', unsafe_allow_html=True)
                st.stop()

            atw_available = 'ATW' in data['companies']
            include_atw = st.checkbox(
                "Inclure Attijari (ATW) dans le portefeuille de replication",
                value=atw_available,
                disabled=not atw_available,
                help="Decochez pour exclure ATW de la selection automatique et manuelle.",
                on_change=clear_results
            )
            if not atw_available:
                st.markdown('<div class="alert-info">ATW est absente du fichier importe.</div>', unsafe_allow_html=True)

            data = filter_replication_universe(
                data,
                excluded_tickers=[] if include_atw else ['ATW']
            )

            if atw_available and not include_atw:
                st.markdown('<div class="alert-info">ATW est exclue de l\'univers de replication pour cette execution.</div>', unsafe_allow_html=True)

            n_days = len(data['dates'])
            n_actions = len(data['companies'])
            idx_name = data.get('index_name')

            if st.session_state.get('window_max_days') != n_days:
                st.session_state['window_max_days'] = n_days
                if st.session_state.get('train_days_input') is not None:
                    st.session_state['train_days_input'] = clamp_int(st.session_state['train_days_input'], 1, n_days)
                if 'test_days_input' in st.session_state:
                    st.session_state['test_days_input'] = None
                if 'rebal_days_input' in st.session_state:
                    st.session_state['rebal_days_input'] = None
                st.rerun()

            if n_actions == 0:
                st.markdown('<div class="alert-error">Aucun titre disponible pour la replication avec ce filtrage.</div>', unsafe_allow_html=True)
                st.stop()

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(metric_card('Jours', str(n_days), 'purple'), unsafe_allow_html=True)
            with c2:
                st.markdown(metric_card('Titres', str(n_actions), 'green'), unsafe_allow_html=True)
            with c3:
                st.markdown(metric_card('Indice', idx_name[:20], 'blue'), unsafe_allow_html=True)
            with c4:
                period = f"{data['dates'][0].strftime('%d/%m/%y')} → {data['dates'][-1].strftime('%d/%m/%y')}"
                st.markdown(f'''<div class="glass-metric">
                    <div class="glow glow-pink"></div>
                    <h4>Période</h4>
                    <div class="val" style="font-size:0.95rem">{period}</div>
                </div>''', unsafe_allow_html=True)

            # ── Section 2: Parameter picking ──
            st.markdown("""
            <div class="section-card" style="margin-top:2rem;">
                <div class="section-label"><span class="num">2</span> PORTEFEUILLE</div>
                <div class="section-heading">Titres et poids</div>
            </div>
            """, unsafe_allow_html=True)
            
            try:
                load_factors(required=True)
                factor_methods_available = True
                factor_methods_error = ""
            except Exception as exc:
                factor_methods_available = False
                factor_methods_error = str(exc)

            col_k, col_sel = st.columns(2)
            with col_k:
                user_k = st.number_input(
                    "Nb titres (K)", 
                    min_value=1, max_value=n_actions, value=min(7, n_actions), step=1,
                    help="Nombre de titres à retenir.",
                    on_change=clear_results
                )
                st.session_state['user_k'] = user_k
                
            with col_sel:
                selection_options = ["Lasso", "Beta", "Exhaustive DE"]
                if factor_methods_available:
                    selection_options.extend(["Corr × Cap", "Corr × Flottant"])
                selection_options.extend(["Ledoit-Wolf", "Manuelle"])

                selection_method_choice = st.selectbox(
                    "Sélection", 
                    selection_options,
                    help="Choisissez la méthode de sélection des titres.",
                    on_change=clear_results
                )
                if selection_method_choice == "Lasso":
                    selection_method = "lasso"
                elif selection_method_choice == "Beta":
                    selection_method = "beta"
                elif selection_method_choice == "Exhaustive DE":
                    selection_method = "exhaustive_de"
                elif selection_method_choice == "Corr × Cap":
                    selection_method = "corr_cap"
                elif selection_method_choice == "Corr × Flottant":
                    selection_method = "corr_float"
                elif selection_method_choice == "Ledoit-Wolf":
                    selection_method = "lw"
                else:
                    selection_method = "manual"

            if not factor_methods_available:
                st.markdown(
                    f'<div class="alert-warning">⚠️ Les méthodes Corr × Cap et Corr × Flottant sont temporairement masquées : {factor_methods_error}</div>',
                    unsafe_allow_html=True,
                )
                    
            if selection_method == "exhaustive_de":
                combos_count = math.comb(n_actions, int(user_k))
                st.markdown(
                    f'<div class="alert-warning">Recherche exhaustive active : {combos_count:,} combinaisons seront analysees sur chaque fenetre, puis le DE affinera les meilleurs candidats.</div>',
                    unsafe_allow_html=True,
                )

            selected_titles = []
            if selection_method == "manual":
                selected_titles = st.multiselect(
                    f"🎯 Choisissez {user_k} titres",
                    options=data['companies'],
                    default=[],
                    help=f"Sélectionnez {user_k} titres.",
                    on_change=clear_results
                )
                if len(selected_titles) != user_k and len(selected_titles) > 0:
                    st.markdown(f'<div class="alert-info">ℹ️ K ajusté à {len(selected_titles)} titre(s).</div>', unsafe_allow_html=True)
                    user_k = len(selected_titles)
                    st.session_state['user_k'] = user_k
            
            weights_valid = True
            weights_array = None
            weight_method = 'de'
            target_beta = None
            max_weight = None
            
            if user_k > 0:
                if user_k == 1 and selection_method == 'manual' and len(selected_titles) == 1:
                    st.markdown('<div class="alert-info">ℹ️ K=1 : poids = 100%.</div>', unsafe_allow_html=True)
                    weight_method = 'manual'
                    weights_array = np.array([1.0])
                elif selection_method == 'manual' and len(selected_titles) == 0:
                    st.markdown('<div class="alert-info">ℹ️ Sélectionnez au moins un titre.</div>', unsafe_allow_html=True)
                    weights_valid = False
                else:
                    if selection_method == 'manual':
                        weight_method_choice = st.radio(
                            "Pondération", 
                            ["Optimisée", "Manuelle"], 
                            horizontal=True,
                            help="Optimisée : poids calculés automatiquement. Manuelle : poids saisis par l'utilisateur.",
                            on_change=clear_results
                        )
                        weight_method = 'de' if "Optimisée" in weight_method_choice else 'manual'
                    else:
                        weight_method = 'de'
                    
                    if weight_method == 'de':
                        st.markdown("##### 📌 Contraintes")
                        
                        c1, c2 = st.columns(2)
                        with c1:
                            constrain_beta = st.checkbox("Bêta cible", value=False, help="Ajoute une cible de bêta au portefeuille.", on_change=clear_results)
                            if constrain_beta:
                                target_beta = st.number_input("Bêta", min_value=-2.0, max_value=5.0, value=1.0, step=0.05, on_change=clear_results)
                            else:
                                target_beta = None
                                
                        with c2:
                            apply_cap = st.checkbox("Poids max", value=False, help="Fixe un poids maximal par titre.", on_change=clear_results)
                            if apply_cap:
                                max_weight_val = st.number_input("Max (%)", min_value=1.0, max_value=100.0, value=20.0, step=1.0, on_change=clear_results)
                                max_weight = max_weight_val / 100.0
                            else:
                                max_weight = None

                        if max_weight is not None and (user_k * max_weight) < 1.0 - 1e-12:
                            st.markdown(
                                f'<div class="alert-error">⚠️ Plafond impossible : avec K={user_k}, un max de {max_weight*100:.1f}% ne permet pas d\'atteindre 100%. '
                                f'Minimum faisable : {100.0 / user_k:.2f}% par titre.</div>',
                                unsafe_allow_html=True
                            )
                            weights_valid = False

                    if weight_method == 'manual':
                        weight_cols = st.columns(min(len(selected_titles), 4))
                        raw_weights = {}
                        for i, title in enumerate(selected_titles):
                            with weight_cols[i % len(weight_cols)]:
                                raw_weights[title] = st.number_input(
                                    f"{title} (%)", min_value=0.0, max_value=100.0,
                                    value=round(100.0 / len(selected_titles), 2),
                                    step=0.5, key=f"w_{title}"
                                )

                        total_weight = sum(raw_weights.values())
                        if abs(total_weight - 100.0) < 0.01:
                            weights_array = np.array([raw_weights[t] / 100.0 for t in selected_titles])
                            weights_valid = True
                        else:
                            st.markdown(f'<div class="alert-error">⚠️ Total poids = <strong>{total_weight:.2f}%</strong> (doit être 100%).</div>', unsafe_allow_html=True)
                            weights_valid = False
            else:
                weights_valid = False
                
            selected_indices = [data['companies'].index(t) for t in selected_titles] if selected_titles else []
            
            if not weights_valid:
                st.stop()

            if train_days is None:
                st.markdown('<div class="alert-info">Renseignez la fenetre Train.</div>', unsafe_allow_html=True)
                st.stop()

            train_days = int(train_days)
            if train_days < 2:
                st.markdown(
                    '<div class="alert-error">Train doit etre au moins egal a 2 jours. Avec 1 seul jour, la selection et la tracking error ne sont pas exploitables.</div>',
                    unsafe_allow_html=True
                )
                st.stop()

            is_backtest = "Backtest" in mode

            if is_backtest:
                remaining_after_train = n_days - train_days
                if remaining_after_train <= 15:
                    st.markdown(
                        f'<div class="alert-error">Train trop grand : avec {train_days} jours de train sur une base de {n_days} jours, il reste {remaining_after_train} jour(s). Le reliquat doit etre strictement superieur a 15 jours.</div>',
                        unsafe_allow_html=True
                    )
                    st.stop()

                if test_days is None or rebal_days is None:
                    st.markdown('<div class="alert-info">Renseignez les fenetres Test OOS et Rebal.</div>', unsafe_allow_html=True)
                    st.stop()

                test_days = int(test_days)
                rebal_days = int(rebal_days)

                if not (15 <= test_days <= remaining_after_train):
                    st.markdown(
                        f'<div class="alert-error">Test OOS doit etre dans l\'intervalle [15, {remaining_after_train}] jours.</div>',
                        unsafe_allow_html=True
                    )
                    st.stop()

                if not (15 <= rebal_days <= remaining_after_train):
                    st.markdown(
                        f'<div class="alert-error">Rebal doit etre dans l\'intervalle [15, {remaining_after_train}] jours.</div>',
                        unsafe_allow_html=True
                    )
                    st.stop()

            min_required = train_days + test_days if is_backtest else 10

            if n_days < min_required:
                st.markdown(f'<div class="alert-error">❌ <strong>Données insuffisantes</strong> : {n_days} jours disponibles, {min_required} requis.</div>', unsafe_allow_html=True)
                st.stop()

            if is_backtest:
                schedule_info = compute_rebal_schedule(data, train_days=train_days, test_days=test_days, rebal_days=rebal_days)
                n_rebals = schedule_info['n_rebals']
                unused_pts = schedule_info['unused_data_points']
                used_pts = schedule_info['used_data_points']

                if n_rebals == 0:
                    st.markdown(f'<div class="alert-error">❌ <strong>Configuration impossible</strong> : aucun rebal possible avec {n_days} jours. Réduisez la fenêtre train ou test.</div>', unsafe_allow_html=True)
                    st.stop()

                st.markdown(f'<div class="alert-success">✅ Base valide : <strong>{n_rebals} rebal(s)</strong> possible(s).</div>', unsafe_allow_html=True)

                # ── Show data usage info ──
                if unused_pts > 0:
                    st.markdown(f"""
                    <div class="alert-info">
                        ℹ️ <strong>Utilisation :</strong><br>
                        • Total : <strong>{n_days}</strong> jours<br>
                        • Train initial : <strong>{train_days}</strong> jours<br>
                        • Test couvert : <strong>{used_pts - train_days}</strong> jours<br>
                        • <strong style="color:#fbbf24">{unused_pts} jour(s) non utilisé(s)</strong><br>
                        <span style="font-size:0.82rem; color:#64748b;">Ces jours restants ne suffisent pas pour une fenêtre de test complète de {test_days} jours.</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="alert-info">
                        ℹ️ <strong>Utilisation :</strong> les <strong>{n_days}</strong> lignes sont utilisées.
                    </div>
                    """, unsafe_allow_html=True)

                # ── Detailed schedule with multi-select ──
                if n_rebals > 1:
                    with st.expander(f"📅 Calendrier des {n_rebals} rebals"):
                        schedule_df = pd.DataFrame([{
                            'Rebal #': s['rebal_num'],
                            'Train Début': s['train_start_date'],
                            'Train Fin': s['train_end_date'],
                            'Jours Train': s['train_days'],
                            'Test Début': s['test_start_date'],
                            'Test Fin': s['test_end_date'],
                            'Jours Test': s['test_days'],
                        } for s in schedule_info['schedule']])
                        st.dataframe(schedule_df.rename(columns={
                            'Rebal #': 'Rebal',
                            'Train Début': 'Train début',
                            'Train Fin': 'Train fin',
                            'Jours Train': 'Train (j)',
                            'Test Début': 'Test début',
                            'Test Fin': 'Test fin',
                            'Jours Test': 'Test (j)',
                        }), width='stretch', hide_index=True)

                    rebal_options = {}
                    for s in schedule_info['schedule']:
                        label = f"#{s['rebal_num']} — {s['test_start_date']} → {s['test_end_date']}  (Train {s['train_days']}j | Test {s['test_days']}j)"
                        rebal_options[s['rebal_num']] = label

                    all_rebal_nums = list(rebal_options.keys())

                    chosen_rebal_list = st.multiselect(
                        "🎯 Rebals à exécuter",
                        options=all_rebal_nums,
                        default=all_rebal_nums,
                        format_func=lambda x: rebal_options[x],
                        help="Choisissez les rebals à lancer."
                    )

                    if not chosen_rebal_list:
                        st.markdown('<div class="alert-error">⚠️ Sélectionnez au moins un rebal.</div>', unsafe_allow_html=True)
                    elif len(chosen_rebal_list) < n_rebals:
                        st.markdown(f"""
                        <div class="alert-info">
                            📊 <strong>{len(chosen_rebal_list)} rebal(s)</strong> sélectionné(s) sur {n_rebals}.
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    chosen_rebal_list = [1]

            else:
                actual_days = min(n_days, train_days)
                d_start = data['dates'][max(0, n_days - train_days)].strftime('%Y-%m-%d')
                d_end = data['dates'][-1].strftime('%Y-%m-%d')
                
                if n_days < train_days:
                    st.markdown(f'<div class="alert-info">⚠️ Base courte : {n_days} jours. Le mode simple utilisera toute la période disponible, du <strong>{d_start}</strong> au <strong>{d_end}</strong>.</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="alert-success">✅ Base valide : mode simple sur les {actual_days} derniers jours, du <strong>{d_start}</strong> au <strong>{d_end}</strong>.</div>', unsafe_allow_html=True)

            # ── Section 3: Run ──
            st.markdown("""
            <div class="section-card">
                <div class="section-label"><span class="num">3</span> EXÉCUTION</div>
                <div class="section-heading">Lancer</div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("🚀 Lancer", type="primary", width='stretch'):
                progress_bar = st.progress(0)
                status_text = st.empty()

                if is_backtest:
                    if not chosen_rebal_list:
                        st.markdown('<div class="alert-error">❌ Aucun rebal sélectionné.</div>', unsafe_allow_html=True)
                        st.stop()
                    def progress_cb(current, total, info):
                        progress_bar.progress(current / total)
                        lbl = info.strftime('%Y-%m-%d') if hasattr(info, 'strftime') else info
                        status_text.markdown(f"⏳ **Rebal {current}/{total}** — {lbl}")
                    result, elapsed = run_with_elapsed(
                        run_rolling,
                        data,
                        K=user_k,
                        selected_indices=selected_indices,
                        selection_method=selection_method,
                        weight_method=weight_method,
                        manual_weights=weights_array,
                        progress_callback=progress_cb,
                        selected_rebals=chosen_rebal_list,
                        target_beta=target_beta,
                        train_days=train_days,
                        test_days=test_days,
                        rebal_days=rebal_days,
                        max_weight=max_weight,
                    )
                else:
                    def progress_cb(current, total, info):
                        progress_bar.progress(current / total)
                        status_text.markdown(f"⏳ **Étape {current}/{total}** — {info}")
                    result, elapsed = run_with_elapsed(
                        run_simple_replication,
                        data,
                        K=user_k,
                        selected_indices=selected_indices,
                        selection_method=selection_method,
                        weight_method=weight_method,
                        manual_weights=weights_array,
                        progress_callback=progress_cb,
                        target_beta=target_beta,
                        train_days=train_days,
                        max_weight=max_weight,
                    )

                progress_bar.progress(1.0)
                status_text.empty()
                st.session_state['result'] = result
                st.session_state['data'] = data
                st.session_state['mode'] = 'backtest' if is_backtest else 'simple'
                st.session_state['elapsed'] = elapsed
                st.rerun()

            # ── Display results ──
            if 'result' in st.session_state:
                result = st.session_state['result']
                elapsed = st.session_state.get('elapsed', 0)

                st.markdown(f"""
                <div class="section-card">
                    <div class="section-label"><span class="num">4</span> RÉSULTATS</div>
                    <div class="section-heading">Terminé en {elapsed:.1f}s</div>
                </div>
                """, unsafe_allow_html=True)

                if st.session_state['mode'] == 'backtest':
                    display_backtest_results(result, data)
                else:
                    display_simple_results(result, data)

        except Exception as e:
            st.markdown(f'<div class="alert-error">❌ Erreur : <code>{e}</code></div>', unsafe_allow_html=True)
            st.code(traceback.format_exc())
    else:
        st.markdown("""
        <div class="alert-info">
            ℹ️ Importez un fichier pour commencer.
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="app-footer">
        <span style="color:#64748b">Réplication d'indice</span>
    </div>
    """, unsafe_allow_html=True)
