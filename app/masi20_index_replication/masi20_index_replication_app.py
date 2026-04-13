# -*- coding: utf-8 -*-
"""
Streamlit App — Réplication MASI 20
UI Premium — Glassmorphism + Micro-animations
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io, time, hashlib
from masi20_index_replication_engine import (
    prepare_data, run_rolling, run_simple_replication, compute_rebal_schedule,
    TRAIN_DAYS, TEST_DAYS, REBAL_DAYS, greedy_round_l2, load_factors, normalize_ticker
)

APP_TITLE = "MASI20 Index Replication"

NAVIGATION_STATE_KEY = 'selected_app'


def run():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="https://www.google.com/s2/favicons?domain=attijariwafabank.com&sz=128",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if st.button("\u2b05\ufe0f Retour a l'accueil", key='back_to_home_index_replication'):
        st.session_state[NAVIGATION_STATE_KEY] = None
        st.rerun()


    # ══════════════════════════════════════════════════════════════
    # PREMIUM CSS
    # ══════════════════════════════════════════════════════════════
    import os
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
                del st.session_state['uploaded_signature']
            return

        curr_signature = get_uploaded_file_signature(uploaded)
        if prev_signature != curr_signature:
            clear_results()
            st.session_state['uploaded_signature'] = curr_signature

        uploaded.seek(0)


    @st.cache_data(show_spinner=False)
    def load_uploaded_dataset(file_name, payload):
        buffer = io.BytesIO(payload)
        raw_df = pd.read_csv(buffer) if file_name.lower().endswith('.csv') else pd.read_excel(buffer)
        return raw_df, prepare_data(raw_df)




    def create_export_excel(result, data, mode):
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
            price_mode = st.radio("Prix", ["Saisie manuelle", "Fichier prix"], horizontal=True, key=f"{key_prefix}_pmode")
            
        tickers = df_w_selected['Titre'].tolist()
        weights_target = df_w_selected['Poids DE (%)'].values / 100.0
        
        if price_mode == "Fichier prix":
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
                    
            # Initialize editor data
            editor_data = []
            for t in tickers:
                p = imported_prices.get(normalize_ticker(t), 0.0)
                editor_data.append({"ticker": t, "price": p})
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

            # Filtration et renommage des colonnes de score selon la méthode
            display_sel_w = sel_w.copy()
            
            # Déterminer le label de la méthode
            # On peut essayer de le deviner via result['hyper_records'] ou passer l'info
            method_label = "Lasso" # par défaut
            if 'hyper_records' in result and len(result['hyper_records']) > 0:
                method_info = result['hyper_records'][0].get('LASSO Méthode', '').lower()
                if 'meta' in method_info or 'score' in method_info: method_label = "Meta"
                elif 'beta' in method_info: method_label = "Beta"
                elif 'corr' in method_info and 'cap' in method_info: method_label = "Corr*Cap"
                elif 'corr' in method_info and 'float' in method_info: method_label = "Corr*Float"
                elif 'ledoit' in method_info or 'lw' in method_info: method_label = "LW"
                elif 'manual' in method_info: method_label = None

            if method_label:
                display_sel_w = display_sel_w.rename(columns={
                    'Rebal #': 'Rebal',
                    'Date Rebal': 'Date',
                    'Poids DE (%)': 'Poids (%)',
                    'Score': f'Score {method_label}',
                    'Rang': f'Rang {method_label}'
                })
            else:
                # Mode manuel, on retire les colonnes de score qui n'ont pas de sens
                if 'Score' in display_sel_w.columns:
                    display_sel_w = display_sel_w.drop(columns=['Score', 'Rang'])
                display_sel_w = display_sel_w.rename(columns={
                    'Rebal #': 'Rebal',
                    'Date Rebal': 'Date',
                    'Poids DE (%)': 'Poids (%)',
                })

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
        excel_data = create_export_excel(result, data, 'backtest')
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
            # Filtration et renommage des colonnes de score selon la méthode
            display_df_w = df_w.copy()
            
            method_info = result.get('lasso_info', {}).get('method', result.get('lasso_info', {}).get('alpha_method', '')).lower()
            method_label = "Lasso"
            if 'meta' in method_info or 'score' in method_info: method_label = "Meta"
            elif 'beta' in method_info: method_label = "Beta"
            elif 'corr' in method_info and 'cap' in method_info: method_label = "Corr*Cap"
            elif 'corr' in method_info and 'float' in method_info: method_label = "Corr*Float"
            elif 'ledoit' in method_info or 'lw' in method_info: method_label = "LW"
            elif not method_info: method_label = None # Manuel

            if method_label:
                display_df_w = display_df_w.rename(columns={
                    'Poids DE (%)': 'Poids (%)',
                    'Score': f'Score {method_label}',
                    'Rang': f'Rang {method_label}'
                })
            elif 'Score' in display_df_w.columns:
                display_df_w = display_df_w.drop(columns=['Score', 'Rang'])
                display_df_w = display_df_w.rename(columns={'Poids DE (%)': 'Poids (%)'})

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
        excel_data = create_export_excel(result, data, 'simple')
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

    with st.sidebar:
        st.markdown(f"""
        <div class="sidebar-logo" style="display: flex; align-items: center; margin-bottom: 20px;">
            <img src="https://www.google.com/s2/favicons?domain=attijariwafabank.com&sz=128" style="width: 40px; margin-right: 12px; border-radius: 4px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
            <div class="logo-text" style="font-size: 1.1rem; font-weight: 600;">{APP_TITLE}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        mode = st.radio(
            "**Mode**",
            ["🔄 Backtest", "📌 Simple"],
            help="Backtest : rolling window avec test OOS.\nSimple : un seul portefeuille.",
            on_change=clear_results
        )

        st.markdown("---")
        st.markdown("##### ⚙️ Fenêtres")
        
        # UI defaults come from engine constants
        train_days = st.slider("Train (jours)", min_value=5, max_value=252, value=TRAIN_DAYS, step=1, on_change=clear_results)
        
        if "Backtest" in mode:
            test_days = st.slider("Test OOS (jours)", min_value=5, max_value=60, value=TEST_DAYS, step=1, on_change=clear_results)
            rebal_days = st.slider("Rebal (jours)", min_value=5, max_value=60, value=REBAL_DAYS, step=1, on_change=clear_results)


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

            n_days = len(data['dates'])
            n_actions = len(data['companies'])
            idx_name = data.get('index_name')

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
                selection_options = ["Lasso", "Meta Score", "Beta"]
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
                elif selection_method_choice == "Meta Score":
                    selection_method = "score"
                elif selection_method_choice == "Beta":
                    selection_method = "beta"
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
                    if selection_method not in ['manual', 'beta', 'corr_cap', 'lw']:
                        weight_method = 'de'
                    elif selection_method in ['beta', 'corr_cap', 'lw']:
                        weight_method = 'de'
                    else:
                        weight_method_choice = st.radio(
                            "Pondération", 
                            ["Optimisée", "Manuelle"], 
                            horizontal=True,
                            help="Optimisée : poids calculés automatiquement. Manuelle : poids saisis par l'utilisateur.",
                            on_change=clear_results
                        )
                        weight_method = 'de' if "Optimisée" in weight_method_choice else 'manual'
                    
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

            is_backtest = "Backtest" in mode
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
                    t0 = time.time()
                    result = run_rolling(data, K=user_k, selected_indices=selected_indices, selection_method=selection_method, weight_method=weight_method, manual_weights=weights_array, progress_callback=progress_cb, selected_rebals=chosen_rebal_list, target_beta=target_beta, train_days=train_days, test_days=test_days, rebal_days=rebal_days, max_weight=max_weight)
                    elapsed = time.time() - t0
                else:
                    def progress_cb(current, total, info):
                        progress_bar.progress(current / total)
                        status_text.markdown(f"⏳ **Étape {current}/{total}** — {info}")
                    t0 = time.time()
                    result = run_simple_replication(data, K=user_k, selected_indices=selected_indices, selection_method=selection_method, weight_method=weight_method, manual_weights=weights_array, progress_callback=progress_cb, target_beta=target_beta, train_days=train_days, max_weight=max_weight)
                    elapsed = time.time() - t0

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
            import traceback
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
