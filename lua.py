import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Configurações para deploy
plt.style.use('default')
sns.set_palette("husl")

# =================== EMPATHY PACKAGE (INTEGRADO) ===================

class EmpathyScorer:
    """Compute empathy scores from tabular-like data.

    Components implemented:
    - global mean alignment
    - principal component (pc1) projection with optional prior
    - local kNN cosine affinity (approximate)
    """

    def __init__(self, weights=None, k=3):
        if weights is None:
            weights = {'global': 0.4, 'pc1': 0.4, 'knn': 0.2}
        self.w = weights
        self.k = k

    @staticmethod
    def _to_matrix(data, feature_order=None):
        keys = list(data.keys()) if feature_order is None else feature_order
        arrs = [np.asarray(data[k], dtype=float) for k in keys]
        n = arrs[0].shape[0]
        mat = np.vstack([a.reshape(n) for a in arrs]).T
        return mat, keys

    @staticmethod
    def _zscore(X):
        mu = X.mean(axis=0, keepdims=True)
        sigma = X.std(axis=0, keepdims=True) + 1e-12
        return (X - mu) / sigma

    def _global_mean(self, Xz):
        mu = Xz.mean(axis=0, keepdims=True)
        mu_norm = np.linalg.norm(mu) + 1e-12
        mu_dir = mu / mu_norm
        return (Xz @ mu_dir.T).ravel()

    def _pc1(self, Xz, init_v=None):
        U, S, Vt = np.linalg.svd(Xz, full_matrices=False)
        v1 = Vt[0]
        if init_v is not None:
            iv = np.asarray(init_v, dtype=float)
            iv = iv / (np.linalg.norm(iv) + 1e-12)
            v1 = 0.7 * v1 + 0.3 * iv
        v1 = v1 / (np.linalg.norm(v1) + 1e-12)
        return Xz @ v1

    def _knn_affinity(self, Xz, k=None):
        if k is None:
            k = self.k
        norms = np.linalg.norm(Xz, axis=1, keepdims=True) + 1e-12
        Xn = Xz / norms
        S = Xn @ Xn.T
        np.fill_diagonal(S, -np.inf)
        k = min(k, max(1, Xz.shape[0] - 1))
        idx = np.argpartition(S, -k, axis=1)[:, -k:]
        topk = S[np.arange(S.shape[0])[:, None], idx]
        return topk.mean(axis=1)

    def calculate_empathy(self, data, feature_order=None, init_v=None):
        X, keys = self._to_matrix(data, feature_order)
        Xz = self._zscore(X)
        g = self._global_mean(Xz)
        p = self._pc1(Xz, init_v=init_v)
        k = self._knn_affinity(Xz, k=self.k)
        combined = self.w['global'] * g + self.w['pc1'] * p + self.w['knn'] * k
        comb_z = (combined - combined.mean()) / (combined.std() + 1e-12)
        return comb_z

class EPINNModel:
    """Empathetic Physics-Informed Neural Network wrapper"""

    def __init__(self, lambda_sat=1e-2, lambda_cons=1e-2, lambda_lat=1e-2):
        self.lambda_sat = float(lambda_sat)
        self.lambda_cons = float(lambda_cons)
        self.lambda_lat = float(lambda_lat)
        self.is_fitted = False
        self.coef_ = None
        self.bias_ = None
        self.training_history = []

    @staticmethod
    def _build_design_matrix(data, empathy_z, feature_order=None):
        X, keys = EmpathyScorer._to_matrix(data, feature_order)
        Xz = EmpathyScorer._zscore(X)
        return np.column_stack([Xz, empathy_z.reshape(-1, 1)])

    def fit(self, data, empathy_z, target=None, feature_order=None, epochs=1000, lr=1e-2):
        n = len(empathy_z)
        X = self._build_design_matrix(data, empathy_z, feature_order)
        
        if target is None:
            if 'sales' in data:
                y = np.asarray(data['sales'], dtype=float).reshape(-1, 1)
            else:
                raise ValueError("No target provided and 'sales' not found in data")
        else:
            y = np.asarray(target, dtype=float).reshape(-1, 1)

        # Inicializar parâmetros
        rng = np.random.RandomState(42)
        d = X.shape[1]
        w = rng.normal(scale=0.1, size=(d, 1))
        b = np.array([[0.0]])

        observed_total = y.sum()
        total_slack = 0.25 * observed_total
        
        self.training_history = []

        # Treinamento com penalidades físicas
        for epoch in range(epochs):
            preds = X @ w + b
            mse = np.mean((preds - y) ** 2)
            
            # Penalidade de saturação
            eps = 1e-2
            if X.shape[1] >= 3:
                X_up = X.copy(); X_dn = X.copy()
                X_up[:, 0] += eps; X_dn[:, 0] -= eps
                p_up = X_up @ w + b; p_dn = X_dn @ w + b
                sec_vis = p_up - 2 * preds + p_dn
                sat_vis = np.mean(np.maximum(sec_vis, 0.0))

                X_up2 = X.copy(); X_dn2 = X.copy()
                X_up2[:, 2] += eps; X_dn2[:, 2] -= eps
                p_up2 = X_up2 @ w + b; p_dn2 = X_dn2 @ w + b
                sec_click = p_up2 - 2 * preds + p_dn2
                sat_click = np.mean(np.maximum(sec_click, 0.0))
            else:
                sat_vis = 0.0; sat_click = 0.0

            sat_pen = float(sat_vis + sat_click)
            total_pred = float(np.sum(preds))
            cons_pen = max(0.0, (total_pred - (observed_total + total_slack)) / (observed_total + 1e-12))
            
            preds_r = preds.flatten()
            if len(preds_r) > 1:
                diffs = preds_r[1:] - preds_r[:-1]
                lat_pen = float(np.mean(diffs ** 2))
            else:
                lat_pen = 0.0

            total_loss = mse + self.lambda_sat * sat_pen + self.lambda_cons * cons_pen + self.lambda_lat * lat_pen
            
            # Salvar histórico
            if epoch % 100 == 0:
                self.training_history.append({
                    'epoch': epoch,
                    'mse': mse,
                    'sat_penalty': sat_pen,
                    'cons_penalty': cons_pen,
                    'lat_penalty': lat_pen,
                    'total_loss': total_loss
                })

            # Gradientes
            grad_w = (2.0 / n) * (X.T @ (preds - y))
            grad_b = (2.0 / n) * np.sum(preds - y)

            if cons_pen > 0:
                grad_w += (1.0 / (observed_total + 1e-12)) * np.sum(X, axis=0).reshape(-1, 1)
                grad_b += (1.0 / (observed_total + 1e-12)) * n

            if len(preds_r) > 1:
                Xdiff = X[1:, :] - X[:-1, :]
                pdiff = preds_r[1:] - preds_r[:-1]
                grad_lat_w = (2.0 / (n - 1)) * (Xdiff.T @ pdiff.reshape(-1, 1))
                grad_w += self.lambda_lat * grad_lat_w

            # Atualização
            w -= lr * grad_w
            b -= lr * grad_b

        self.coef_ = w
        self.bias_ = b
        self.is_fitted = True

    def predict(self, data, empathy_z, feature_order=None):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        X = self._build_design_matrix(data, empathy_z, feature_order)
        preds = X @ self.coef_ + self.bias_
        return preds.ravel()

    def generate_visitor_ids(self, empathy_z, n_classes=24):
        qs = np.quantile(empathy_z, [0.2, 0.4, 0.6, 0.8])
        scores = np.digitize(empathy_z, qs) + 1
        rng = np.random.RandomState(42)
        classes = rng.randint(0, n_classes, size=empathy_z.shape[0])
        ids = [f"{int(s)}&{int(c)}" for s, c in zip(scores, classes)]
        return ids

# =================== SISTEMA PRINCIPAL ===================

class MoonPhaseCalculator:
    @staticmethod
    def get_moon_phase(date_input):
        if isinstance(date_input, str):
            date_input = datetime.strptime(date_input, '%Y-%m-%d').date()
        elif isinstance(date_input, datetime):
            date_input = date_input.date()
        
        known_new_moon = date(2000, 1, 6)
        days_since = (date_input - known_new_moon).days
        cycle_position = (days_since % 29.53) / 29.53
        
        if cycle_position < 0.125:
            return 'Nova'
        elif cycle_position < 0.375:
            return 'Crescente'
        elif cycle_position < 0.625:
            return 'Cheia'
        else:
            return 'Minguante'

class EPINNDiagnosticSystem:
    def __init__(self):
        self.empathy_scorer = EmpathyScorer()
        self.epinn_model = EPINNModel()
        self.moon_calculator = MoonPhaseCalculator()
        self.data_processed = False
        
    def generate_comprehensive_data(self, n_samples=2000):
        """Gera dados sintéticos otimizados para análise E-PINN"""
        np.random.seed(42)
        
        data = []
        for i in range(n_samples):
            random_date = date(2023, 1, 1) + pd.Timedelta(days=np.random.randint(0, 730))
            moon_phase = self.moon_calculator.get_moon_phase(random_date)
            
            # Features para análise E-PINN
            visits = np.random.normal(1000, 150)  # Visitas ao site
            stories = np.random.poisson(3.5)      # Histórias visualizadas
            clicks = np.random.normal(visits * 0.2, 30)  # Clicks
            
            # Features adicionais
            age = np.random.normal(35, 12)
            income = np.random.lognormal(10, 0.5)
            email_engagement = np.random.gamma(2, 2)
            social_score = np.random.beta(2, 5) * 10
            
            # Mapeamento numérico da fase da lua
            moon_phase_num = {'Nova': 0, 'Crescente': 1, 'Cheia': 2, 'Minguante': 3}[moon_phase]
            
            # Influência da empathy na probabilidade de venda
            base_prob = (
                0.05 + 
                (visits - 800) * 0.0001 +
                stories * 0.02 +
                clicks * 0.001 +
                (age - 20) * 0.003 +
                np.log(income) * 0.015 +
                email_engagement * 0.01 +
                social_score * 0.008
            )
            
            # Modifica pela fase da lua
            moon_multiplier = {'Nova': 0.8, 'Crescente': 1.1, 'Cheia': 1.3, 'Minguante': 0.9}[moon_phase]
            base_prob *= moon_multiplier
            
            base_prob = max(0, min(1, base_prob + np.random.normal(0, 0.1)))
            
            # Vendas baseadas na probabilidade
            sales = np.random.poisson(base_prob * 50)
            purchase = 's' if sales > 25 else 'n'
            
            data.append({
                'visits': visits,
                'stories': stories, 
                'clicks': clicks,
                'moon_phase': moon_phase_num,
                'age': age,
                'income': income,
                'email_engagement': email_engagement,
                'social_score': social_score,
                'sales': sales,
                'purchase': purchase,
                'moon_phase_name': moon_phase,
                'base_probability': base_prob
            })
        
        return pd.DataFrame(data)
    
    def run_epinn_analysis(self, df):
        """Executa análise completa E-PINN"""
        # Preparar dados para E-PINN
        epinn_data = {
            'visits': df['visits'].tolist(),
            'stories': df['stories'].tolist(),
            'clicks': df['clicks'].tolist(),
            'moon_phase': df['moon_phase'].tolist(),
            'age': df['age'].tolist(),
            'income': df['income'].tolist(),
            'email_engagement': df['email_engagement'].tolist(),
            'social_score': df['social_score'].tolist(),
            'sales': df['sales'].tolist()
        }
        
        # Calcular empathy scores
        empathy_scores = self.empathy_scorer.calculate_empathy(epinn_data)
        
        # Treinar modelo E-PINN
        self.epinn_model.fit(epinn_data, empathy_scores, epochs=1000)
        
        # Predições
        predictions = self.epinn_model.predict(epinn_data, empathy_scores)
        
        # Gerar visitor IDs
        visitor_ids = self.epinn_model.generate_visitor_ids(empathy_scores)
        
        # Análise de features por variância
        feature_names = ['visits', 'stories', 'clicks', 'moon_phase', 'age', 'income', 'email_engagement', 'social_score']
        feature_matrix = np.column_stack([epinn_data[f] for f in feature_names])
        feature_variances = np.var(feature_matrix, axis=0)
        
        # Importância relativa dos coeficientes E-PINN
        coef_importance = np.abs(self.epinn_model.coef_.flatten()[:-1])  # Exclui empathy coef
        
        results = {
            'empathy_scores': empathy_scores,
            'predictions': predictions,
            'visitor_ids': visitor_ids,
            'feature_names': feature_names,
            'feature_variances': feature_variances,
            'coef_importance': coef_importance,
            'training_history': self.epinn_model.training_history,
            'empathy_coef': self.epinn_model.coef_.flatten()[-1]  # Coef da empathy
        }
        
        return results

def main():
    st.title("🧠 Sistema de Diagnóstico E-PINN")
    st.markdown("### Empathetic Physics-Informed Neural Network para Análise de Compradores")
    
    # Info sobre E-PINN
    st.info("🚀 **E-PINN em Produção** - Análise avançada com função de empatia integrada")
    
    # Sidebar
    st.sidebar.title("⚙️ Configurações E-PINN")
    
    # Inicializa sistema
    if 'epinn_system' not in st.session_state:
        st.session_state.epinn_system = EPINNDiagnosticSystem()
        st.session_state.data_generated = False
    
    # Parâmetros E-PINN
    st.sidebar.subheader("🔧 Parâmetros do Modelo")
    lambda_sat = st.sidebar.slider("Lambda Saturação", 0.001, 0.1, 0.01, step=0.001)
    lambda_cons = st.sidebar.slider("Lambda Conservação", 0.001, 0.1, 0.01, step=0.001)  
    lambda_lat = st.sidebar.slider("Lambda Latência", 0.001, 0.1, 0.01, step=0.001)
    
    # Atualiza parâmetros
    st.session_state.epinn_system.epinn_model.lambda_sat = lambda_sat
    st.session_state.epinn_system.epinn_model.lambda_cons = lambda_cons
    st.session_state.epinn_system.epinn_model.lambda_lat = lambda_lat
    
    # Gerar dados
    if st.sidebar.button("🔄 Gerar Dados E-PINN"):
        with st.spinner("Gerando dados para análise E-PINN..."):
            st.session_state.df = st.session_state.epinn_system.generate_comprehensive_data(2000)
            st.session_state.data_generated = True
        st.sidebar.success("Dados E-PINN gerados!")
    
    # Executar análise
    if st.session_state.data_generated and st.sidebar.button("🧠 Executar Análise E-PINN"):
        with st.spinner("Executando análise E-PINN..."):
            st.session_state.results = st.session_state.epinn_system.run_epinn_analysis(st.session_state.df)
        st.sidebar.success("Análise E-PINN completa!")
    
    # Tabs principais
    if st.session_state.data_generated and 'results' in st.session_state:
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Outputs E-PINN", 
            "👥 Análise de Compradores",
            "📊 Features & Variância", 
            "🧠 Diagnóstico Avançado"
        ])
        
        with tab1:
            st.subheader("🎯 Outputs da Empathy Function")
            
            results = st.session_state.results
            df = st.session_state.df
            
            # Métricas principais
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Samples Analisados", len(results['empathy_scores']))
            with col2:
                st.metric("Coef. Empathy", f"{results['empathy_coef']:.3f}")
            with col3:
                st.metric("Média Empathy", f"{np.mean(results['empathy_scores']):.3f}")
            with col4:
                st.metric("Std Empathy", f"{np.std(results['empathy_scores']):.3f}")
            
            # Tabela de outputs
            st.subheader("📋 Códigos Gerados pela Empathy Function")
            
            output_df = pd.DataFrame({
                'Visitor_ID': results['visitor_ids'][:50],  # Primeiros 50
                'Empathy_Score': results['empathy_scores'][:50],
                'E-PINN_Prediction': results['predictions'][:50],
                'Actual_Sales': df['sales'].iloc[:50],
                'Purchase_Label': df['purchase'].iloc[:50],
                'Moon_Phase': df['moon_phase_name'].iloc[:50]
            })
            
            st.dataframe(output_df, use_container_width=True)
            
            # Download dos códigos
            csv = output_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Códigos E-PINN",
                data=csv,
                file_name='epinn_codes.csv',
                mime='text/csv'
            )
        
        with tab2:
            st.subheader("👥 Legenda para Visualização de Compradores")
            
            results = st.session_state.results
            df = st.session_state.df
            
            # Segmentação por empathy score
            empathy_quartiles = np.percentile(results['empathy_scores'], [25, 50, 75])
            
            def get_empathy_segment(score):
                if score <= empathy_quartiles[0]:
                    return "Baixa Empatia"
                elif score <= empathy_quartiles[1]:
                    return "Empatia Moderada"
                elif score <= empathy_quartiles[2]:
                    return "Alta Empatia"
                else:
                    return "Empatia Extrema"
            
            df['empathy_segment'] = [get_empathy_segment(score) for score in results['empathy_scores']]
            df['empathy_score'] = results['empathy_scores']
            
            # Análise por segmento
            segment_analysis = df.groupby('empathy_segment').agg({
                'purchase': lambda x: (x == 's').mean(),
                'sales': 'mean',
                'visits': 'mean',
                'clicks': 'mean',
                'empathy_score': 'mean'
            }).round(3)
            
            segment_analysis.columns = [
                'Taxa_Conversão', 'Vendas_Médias', 'Visitas_Médias', 
                'Clicks_Médios', 'Empathy_Média'
            ]
            
            st.dataframe(segment_analysis, use_container_width=True)
            
            # Visualização dos segmentos
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots()
                segment_analysis['Taxa_Conversão'].plot(kind='bar', ax=ax, color='skyblue')
                ax.set_title("Taxa de Conversão por Segmento de Empatia")
                ax.set_ylabel("Taxa de Conversão")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots()
                scatter = ax.scatter(
                    results['empathy_scores'], 
                    df['sales'], 
                    c=df['purchase'].map({'s': 'green', 'n': 'red'}),
                    alpha=0.6
                )
                ax.set_xlabel("Empathy Score")
                ax.set_ylabel("Sales")
                ax.set_title("Vendas vs Empathy Score")
                st.pyplot(fig)
            
            # Legenda de cores
            st.subheader("🎨 Legenda de Visualização")
            legend_df = pd.DataFrame({
                'Cor': ['🟢 Verde', '🔴 Vermelho', '🔵 Azul Claro', '🟡 Amarelo'],
                'Significado': [
                    'Comprador (s)',
                    'Não Comprador (n)', 
                    'Empatia Baixa/Moderada',
                    'Empatia Alta/Extrema'
                ],
                'Ação_Recomendada': [
                    'Manter engajamento',
                    'Campanha de reativação',
                    'Desenvolver conexão emocional',
                    'Oferecer produtos premium'
                ]
            })
            
            st.dataframe(legend_df, use_container_width=True)
        
        with tab3:
            st.subheader("📊 Features Mais Associadas & Análise de Variância")
            
            results = st.session_state.results
            
            # Ranking de features por importância
            feature_importance_df = pd.DataFrame({
                'Feature': results['feature_names'],
                'Coef_Importância': results['coef_importance'],
                'Variância': results['feature_variances'],
                'Potencial_Investimento': results['feature_variances'] * results['coef_importance']
            }).sort_values('Potencial_Investimento', ascending=False)
            
            st.subheader("🏆 Ranking de Features para Investimento")
            st.dataframe(feature_importance_df, use_container_width=True)
            
            # Visualizações
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots()
                feature_importance_df.plot(
                    x='Feature', y='Coef_Importância', 
                    kind='bar', ax=ax, color='lightcoral'
                )
                ax.set_title("Importância dos Coeficientes E-PINN")
                ax.set_ylabel("Importância Absoluta")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots()
                feature_importance_df.plot(
                    x='Feature', y='Variância', 
                    kind='bar', ax=ax, color='lightgreen'
                )
                ax.set_title("Variância das Features")
                ax.set_ylabel("Variância")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            
            # Recomendações de investimento
            st.subheader("💡 Recomendações de Investimento por Variância")
            
            top_features = feature_importance_df.head(3)
            
            for idx, row in top_features.iterrows():
                feature_name = row['Feature']
                potential = row['Potencial_Investimento']
                variance = row['Variância']
                
                st.success(f"""
                **#{idx+1} {feature_name.title()}**
                - Potencial de Investimento: {potential:.2f}
                - Variância: {variance:.2f}
                - Recomendação: Foco prioritário para otimização
                """)
        
        with tab4:
            st.subheader("🧠 Diagnóstico Avançado E-PINN")
            
            results = st.session_state.results
            
            # Histórico de treinamento
            if results['training_history']:
                st.subheader("📈 Evolução do Treinamento")
                
                history_df = pd.DataFrame(results['training_history'])
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                
                # MSE
                axes[0,0].plot(history_df['epoch'], history_df['mse'])
                axes[0,0].set_title("MSE Evolution")
                axes[0,0].set_ylabel("MSE")
                
                # Penalidades
                axes[0,1].plot(history_df['epoch'], history_df['sat_penalty'], label='Saturation')
                axes[0,1].plot(history_df['epoch'], history_df['cons_penalty'], label='Conservation')
                axes[0,1].plot(history_df['epoch'], history_df['lat_penalty'], label='Latency')
                axes[0,1].set_title("Physics Penalties")
                axes[0,1].legend()
                
                # Loss total
                axes[1,0].plot(history_df['epoch'], history_df['total_loss'])
                axes[1,0].set_title("Total Loss")
                axes[1,0].set_ylabel("Loss")
                
                # Distribuição de empathy scores
                axes[1,1].hist(results['empathy_scores'], bins=30, alpha=0.7)
                axes[1,1].set_title("Empathy Score Distribution")
                axes[1,1].set_xlabel("Empathy Score")
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Métricas de performance
            st.subheader("⚡ Métricas de Performance E-PINN")
            
            mae = np.mean(np.abs(results['predictions'] - df['sales']))
            rmse = np.sqrt(np.mean((results['predictions'] - df['sales'])**2))
            mape = np.mean(np.abs((results['predictions'] - df['sales']) / (df['sales'] + 1e-8))) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MAE", f"{mae:.2f}")
            with col2:
                st.metric("RMSE", f"{rmse:.2f}") 
            with col3:
                st.metric("MAPE", f"{mape:.1f}%")
            
            # Correlação Empathy vs Vendas
            empathy_sales_corr = np.corrcoef(results['empathy_scores'], df['sales'])[0,1]
            st.metric("Correlação Empathy-Vendas", f"{empathy_sales_corr:.3f}")
            
            # Análise detalhada dos componentes de empathy
            st.subheader("🔍 Decomposição da Empathy Function")
            
            # Recalcular componentes individuais para análise
            epinn_data = {
                'visits': df['visits'].tolist(),
                'stories': df['stories'].tolist(),
                'clicks': df['clicks'].tolist(),
                'moon_phase': df['moon_phase'].tolist(),
                'age': df['age'].tolist(),
                'income': df['income'].tolist(),
                'email_engagement': df['email_engagement'].tolist(),
                'social_score': df['social_score'].tolist()
            }
            
            X, keys = st.session_state.epinn_system.empathy_scorer._to_matrix(epinn_data)
            Xz = st.session_state.epinn_system.empathy_scorer._zscore(X)
            
            global_component = st.session_state.epinn_system.empathy_scorer._global_mean(Xz)
            pc1_component = st.session_state.epinn_system.empathy_scorer._pc1(Xz)
            knn_component = st.session_state.epinn_system.empathy_scorer._knn_affinity(Xz)
            
            # Visualização dos componentes
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Global Mean Component
            axes[0,0].scatter(global_component, df['sales'], alpha=0.6, color='blue')
            axes[0,0].set_title("Global Mean Component vs Sales")
            axes[0,0].set_xlabel("Global Mean Score")
            axes[0,0].set_ylabel("Sales")
            
            # PC1 Component  
            axes[0,1].scatter(pc1_component, df['sales'], alpha=0.6, color='green')
            axes[0,1].set_title("PC1 Component vs Sales")
            axes[0,1].set_xlabel("PC1 Score")
            axes[0,1].set_ylabel("Sales")
            
            # KNN Component
            axes[1,0].scatter(knn_component, df['sales'], alpha=0.6, color='red')
            axes[1,0].set_title("KNN Affinity vs Sales")
            axes[1,0].set_xlabel("KNN Score")
            axes[1,0].set_ylabel("Sales")
            
            # Combined Empathy Score
            axes[1,1].scatter(results['empathy_scores'], df['sales'], alpha=0.6, color='purple')
            axes[1,1].set_title("Combined Empathy vs Sales")
            axes[1,1].set_xlabel("Empathy Score")
            axes[1,1].set_ylabel("Sales")
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Correlações dos componentes
            correlations = {
                'Global Mean': np.corrcoef(global_component, df['sales'])[0,1],
                'PC1': np.corrcoef(pc1_component, df['sales'])[0,1],
                'KNN Affinity': np.corrcoef(knn_component, df['sales'])[0,1],
                'Combined Empathy': empathy_sales_corr
            }
            
            corr_df = pd.DataFrame(list(correlations.items()), 
                                 columns=['Componente', 'Correlação_com_Vendas'])
            corr_df['Correlação_com_Vendas'] = corr_df['Correlação_com_Vendas'].round(3)
            
            st.dataframe(corr_df, use_container_width=True)
            
            # Insights finais
            st.subheader("💡 Insights do Diagnóstico E-PINN")
            
            best_component = corr_df.loc[corr_df['Correlação_com_Vendas'].idxmax()]
            worst_component = corr_df.loc[corr_df['Correlação_com_Vendas'].idxmin()]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"""
                **🎯 Melhor Componente:**
                {best_component['Componente']}
                Correlação: {best_component['Correlação_com_Vendas']}
                
                **Recomendação:** Focar otimizações neste componente
                """)
            
            with col2:
                st.warning(f"""
                **⚠️ Componente Mais Fraco:**
                {worst_component['Componente']}
                Correlação: {worst_component['Correlação_com_Vendas']}
                
                **Recomendação:** Revisar algoritmo ou pesos
                """)
            
            # Análise de outliers de empathy
            st.subheader("🔍 Análise de Outliers de Empathy")
            
            empathy_mean = np.mean(results['empathy_scores'])
            empathy_std = np.std(results['empathy_scores'])
            threshold = 2.5
            
            high_empathy_mask = results['empathy_scores'] > (empathy_mean + threshold * empathy_std)
            low_empathy_mask = results['empathy_scores'] < (empathy_mean - threshold * empathy_std)
            
            high_empathy_sales = df.loc[high_empathy_mask, 'sales'].mean()
            low_empathy_sales = df.loc[low_empathy_mask, 'sales'].mean()
            normal_empathy_sales = df.loc[~(high_empathy_mask | low_empathy_mask), 'sales'].mean()
            
            outlier_analysis = pd.DataFrame({
                'Segmento': ['Alta Empatia (Outliers)', 'Baixa Empatia (Outliers)', 'Empatia Normal'],
                'Quantidade': [high_empathy_mask.sum(), low_empathy_mask.sum(), 
                              (~(high_empathy_mask | low_empathy_mask)).sum()],
                'Vendas_Médias': [high_empathy_sales, low_empathy_sales, normal_empathy_sales],
                'Percentual': [
                    f"{high_empathy_mask.sum()/len(df)*100:.1f}%",
                    f"{low_empathy_mask.sum()/len(df)*100:.1f}%", 
                    f"{(~(high_empathy_mask | low_empathy_mask)).sum()/len(df)*100:.1f}%"
                ]
            })
            
            st.dataframe(outlier_analysis, use_container_width=True)
    
    else:
        # Instruções iniciais
        st.markdown("""
        ## 🚀 Como Usar o Sistema E-PINN
        
        1. **⚙️ Configure os parâmetros** na sidebar (lambdas de penalização)
        2. **🔄 Gere os dados** otimizados para análise E-PINN
        3. **🧠 Execute a análise** completa do modelo
        4. **📊 Explore os resultados** nas diferentes abas
        
        ### 🎯 O que você verá:
        - **Outputs E-PINN**: Códigos gerados pela empathy function
        - **Análise de Compradores**: Segmentação e legenda visual
        - **Features & Variância**: Ranking para investimento
        - **Diagnóstico Avançado**: Performance e insights profundos
        """)
        
        # Explicação técnica
        with st.expander("🔬 Detalhes Técnicos da E-PINN"):
            st.markdown("""
            ### Componentes da Empathy Function:
            
            1. **Global Mean Alignment (40%)**
               - Alinha cada amostra com a direção média global
               - Captura padrões gerais do comportamento
            
            2. **Principal Component (40%)**
               - Projeta dados na primeira componente principal
               - Identifica direção de maior variância
            
            3. **KNN Affinity (20%)**
               - Mede similaridade com vizinhos próximos
               - Captura estrutura local dos dados
            
            ### Penalidades Físicas:
            
            - **λ_sat**: Penaliza saturação não-física
            - **λ_cons**: Mantém conservação de massa/energia
            - **λ_lat**: Reduz latência temporal
            
            ### Output Format:
            - **Visitor IDs**: formato `score&class` 
            - **Empathy Scores**: Z-scores normalizados
            - **Predições**: Vendas esperadas pelo modelo
            """)

if __name__ == "__main__":
    main()
