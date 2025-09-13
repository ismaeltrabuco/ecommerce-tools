import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="🌙 Sistema de Predição de Compras",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

class MoonPhaseCalculator:
    """Calculadora de fases da lua"""
    
    @staticmethod
    def get_moon_phase(date_input):
        """Calcula a fase da lua para uma data específica"""
        if isinstance(date_input, str):
            date_input = datetime.strptime(date_input, '%Y-%m-%d').date()
        elif isinstance(date_input, datetime):
            date_input = date_input.date()
        
        # Algoritmo simplificado para calcular fase da lua
        # Baseado na data de uma lua nova conhecida
        known_new_moon = date(2000, 1, 6)  # Uma lua nova conhecida
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

class PurchasePredictionModel:
    """Modelo avançado de predição de compras"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = {}
        self.moon_calculator = MoonPhaseCalculator()
        
    def generate_synthetic_data(self, n_samples=5000):
        """Gera dados sintéticos para treinamento"""
        np.random.seed(42)
        
        data = []
        for i in range(n_samples):
            # Data aleatória nos últimos 2 anos
            random_date = date(2023, 1, 1) + pd.Timedelta(days=np.random.randint(0, 730))
            moon_phase = self.moon_calculator.get_moon_phase(random_date)
            
            # Features diversas
            age = np.random.normal(35, 12)
            income = np.random.lognormal(10, 0.5)
            website_time = np.random.exponential(5)
            page_views = np.random.poisson(8)
            previous_purchases = np.random.poisson(2)
            email_opens = np.random.binomial(10, 0.3)
            social_media_engagement = np.random.gamma(2, 2)
            season = random_date.month // 3
            day_of_week = random_date.weekday()
            hour_of_visit = np.random.randint(0, 24)
            device_type = np.random.choice(['Mobile', 'Desktop', 'Tablet'], p=[0.6, 0.3, 0.1])
            traffic_source = np.random.choice(['Organic', 'Paid', 'Social', 'Direct'], p=[0.4, 0.3, 0.2, 0.1])
            
            # Influência da fase da lua (hipótese de negócio)
            moon_influence = {
                'Nova': 0.1,
                'Crescente': 0.15,
                'Cheia': 0.2,
                'Minguante': 0.05
            }
            
            # Probabilidade de compra baseada nas features
            prob_base = (
                0.1 + 
                (age - 20) * 0.005 +
                np.log(income) * 0.02 +
                website_time * 0.02 +
                page_views * 0.01 +
                previous_purchases * 0.15 +
                email_opens * 0.03 +
                social_media_engagement * 0.01 +
                moon_influence[moon_phase]
            )
            
            # Adiciona ruído e normaliza
            prob_base = max(0, min(1, prob_base + np.random.normal(0, 0.1)))
            
            # Determina se houve compra
            purchase = np.random.binomial(1, prob_base)
            
            # Determina o nível de confiança
            if prob_base > 0.9:
                confidence = '1'  # Mais de 90%
            elif prob_base < 0.1:
                confidence = '0'  # Menos de 10%
            else:
                confidence = '5'  # Entre 10-90%
            
            data.append({
                'age': age,
                'income': income,
                'website_time_minutes': website_time,
                'page_views': page_views,
                'previous_purchases': previous_purchases,
                'email_opens_last_month': email_opens,
                'social_media_engagement_score': social_media_engagement,
                'moon_phase': moon_phase,
                'season': season,
                'day_of_week': day_of_week,
                'hour_of_visit': hour_of_visit,
                'device_type': device_type,
                'traffic_source': traffic_source,
                'date': random_date,
                'purchase_probability': prob_base,
                'purchase': 's' if purchase else 'n',
                'confidence_level': confidence
            })
        
        return pd.DataFrame(data)
    
    def preprocess_data(self, df):
        """Pré-processa os dados"""
        df_processed = df.copy()
        
        # Codifica variáveis categóricas
        categorical_cols = ['moon_phase', 'device_type', 'traffic_source']
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col + '_encoded'] = self.label_encoders[col].fit_transform(df_processed[col])
            else:
                df_processed[col + '_encoded'] = self.label_encoders[col].transform(df_processed[col])
        
        # Features numéricas
        numeric_features = [
            'age', 'income', 'website_time_minutes', 'page_views',
            'previous_purchases', 'email_opens_last_month', 
            'social_media_engagement_score', 'season', 'day_of_week',
            'hour_of_visit', 'moon_phase_encoded', 'device_type_encoded',
            'traffic_source_encoded'
        ]
        
        return df_processed[numeric_features]
    
    def empathy_function(self, X, y):
        """Função de empatia - balanceia classes e ajusta pesos"""
        class_counts = pd.Series(y).value_counts()
        weights = {cls: len(y) / (len(class_counts) * count) 
                  for cls, count in class_counts.items()}
        return weights
    
    def entropy_active_learning(self, X, y, n_queries=100):
        """Aprendizado ativo baseado em entropia"""
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        
        # Calcula incerteza para cada amostra
        probas = model.predict_proba(X)
        entropy = -np.sum(probas * np.log(probas + 1e-10), axis=1)
        
        # Seleciona amostras com maior incerteza
        uncertain_indices = np.argsort(entropy)[-n_queries:]
        
        return uncertain_indices, entropy
    
    def regularization_comparison(self, X, y):
        """Compara diferentes tipos de regularização"""
        results = {}
        
        # L1 Regularization (Lasso)
        l1_model = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
        l1_scores = cross_val_score(l1_model, X, y, cv=5)
        results['L1'] = {'mean': l1_scores.mean(), 'std': l1_scores.std()}
        
        # L2 Regularization (Ridge)
        l2_model = LogisticRegression(penalty='l2', random_state=42)
        l2_scores = cross_val_score(l2_model, X, y, cv=5)
        results['L2'] = {'mean': l2_scores.mean(), 'std': l2_scores.std()}
        
        # Elastic Net (L1 + L2)
        en_model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, random_state=42)
        en_scores = cross_val_score(en_model, X, y, cv=5)
        results['ElasticNet'] = {'mean': en_scores.mean(), 'std': en_scores.std()}
        
        return results
    
    def kmeans_vs_knn_analysis(self, X, y):
        """Compara K-Means clustering com KNN classification"""
        # K-Means Clustering
        kmeans = KMeans(n_clusters=2, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        
        # Calcula pureza dos clusters
        cluster_purity = []
        for i in range(2):
            cluster_mask = cluster_labels == i
            if np.sum(cluster_mask) > 0:
                cluster_y = y[cluster_mask]
                purity = max(np.mean(cluster_y == 's'), np.mean(cluster_y == 'n'))
                cluster_purity.append(purity)
        
        # KNN Classification
        knn = KNeighborsClassifier(n_neighbors=5)
        knn_scores = cross_val_score(knn, X, y, cv=5)
        
        return {
            'kmeans_purity': np.mean(cluster_purity),
            'knn_accuracy': knn_scores.mean(),
            'cluster_labels': cluster_labels
        }
    
    def train_models(self, df):
        """Treina múltiplos modelos"""
        X = self.preprocess_data(df)
        y = df['purchase'].values
        
        # Normaliza features
        X_scaled = self.scaler.fit_transform(X)
        
        # Divide dados
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Aplica função de empatia
        class_weights = self.empathy_function(X_train, y_train)
        
        # Treina modelos
        models_config = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100, 
                class_weight=class_weights,
                random_state=42
            ),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(
                class_weight=class_weights,
                random_state=42
            ),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
        
        results = {}
        for name, model in models_config.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.models[name] = model
            results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'test_labels': y_test
            }
        
        # Análises avançadas
        uncertain_indices, entropy = self.entropy_active_learning(X_scaled, y)
        reg_comparison = self.regularization_comparison(X_scaled, y)
        clustering_analysis = self.kmeans_vs_knn_analysis(X_scaled, y)
        
        # Feature importance
        if hasattr(models_config['RandomForest'], 'feature_importances_'):
            self.feature_importance = dict(zip(
                X.columns,
                models_config['RandomForest'].feature_importances_
            ))
        
        return results, reg_comparison, clustering_analysis
    
    def predict_purchase(self, customer_data, model_name='RandomForest'):
        """Prediz compra para um cliente"""
        if model_name not in self.models:
            return None, None
        
        # Pré-processa dados do cliente
        df_customer = pd.DataFrame([customer_data])
        X_customer = self.preprocess_data(df_customer)
        X_customer_scaled = self.scaler.transform(X_customer)
        
        model = self.models[model_name]
        prediction = model.predict(X_customer_scaled)[0]
        probability = model.predict_proba(X_customer_scaled)[0]
        
        # Determina nível de confiança
        max_prob = max(probability)
        if max_prob >= 0.9:
            confidence = '1'  # 90%+
        elif max_prob <= 0.1:
            confidence = '0'  # 10%-
        else:
            confidence = '5'  # 50%
        
        return prediction, max_prob, confidence

# Interface Streamlit
def main():
    st.title("🌙 Sistema Inteligente de Predição de Compras")
    st.markdown("### Análise avançada com influência das fases da lua")
    
    # Sidebar
    st.sidebar.title("⚙️ Configurações")
    
    # Inicializa modelo
    if 'model' not in st.session_state:
        st.session_state.model = PurchasePredictionModel()
        st.session_state.data_generated = False
    
    # Gera dados sintéticos
    if st.sidebar.button("🔄 Gerar Dados de Treinamento"):
        with st.spinner("Gerando dados sintéticos..."):
            st.session_state.df = st.session_state.model.generate_synthetic_data(3000)
            st.session_state.data_generated = True
        st.sidebar.success("Dados gerados!")
    
    # Treina modelos
    if st.session_state.data_generated and st.sidebar.button("🤖 Treinar Modelos"):
        with st.spinner("Treinando modelos de ML..."):
            results, reg_comp, cluster_analysis = st.session_state.model.train_models(st.session_state.df)
            st.session_state.results = results
            st.session_state.reg_comparison = reg_comp
            st.session_state.cluster_analysis = cluster_analysis
        st.sidebar.success("Modelos treinados!")
    
    # Tabs principais
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Análise dos Dados", 
        "🤖 Resultados dos Modelos",
        "🔮 Predição Individual",
        "🧠 Análises Avançadas",
        "🌙 Impacto da Lua"
    ])
    
    with tab1:
        if st.session_state.data_generated:
            st.subheader("📈 Visualização dos Dados")
            
            df = st.session_state.df
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total de Registros", len(df))
            with col2:
                st.metric("Taxa de Compra", f"{(df['purchase'] == 's').mean():.1%}")
            with col3:
                st.metric("Idade Média", f"{df['age'].mean():.1f}")
            with col4:
                st.metric("Renda Mediana", f"R$ {df['income'].median():,.0f}")
            
            # Gráficos
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots()
                df['moon_phase'].value_counts().plot(kind='bar', ax=ax)
                ax.set_title("Distribuição por Fase da Lua")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots()
                purchase_by_moon = df.groupby('moon_phase')['purchase'].apply(lambda x: (x == 's').mean())
                purchase_by_moon.plot(kind='bar', ax=ax)
                ax.set_title("Taxa de Compra por Fase da Lua")
                ax.set_ylabel("Taxa de Compra")
                plt.xticks(rotation=45)
                st.pyplot(fig)
        else:
            st.info("👆 Clique em 'Gerar Dados de Treinamento' na sidebar para começar")
    
    with tab2:
        if 'results' in st.session_state:
            st.subheader("🎯 Performance dos Modelos")
            
            results = st.session_state.results
            
            # Métricas de performance
            performance_data = []
            for model_name, result in results.items():
                performance_data.append({
                    'Modelo': model_name,
                    'Acurácia': f"{result['accuracy']:.3f}"
                })
            
            st.dataframe(pd.DataFrame(performance_data), use_container_width=True)
            
            # Matriz de confusão para o melhor modelo
            best_model = max(results.keys(), key=lambda k: results[k]['accuracy'])
            st.subheader(f"📊 Matriz de Confusão - {best_model}")
            
            y_test = results[best_model]['test_labels']
            y_pred = results[best_model]['predictions']
            
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
            ax.set_title(f"Matriz de Confusão - {best_model}")
            ax.set_xlabel("Predito")
            ax.set_ylabel("Real")
            st.pyplot(fig)
        else:
            st.info("👆 Treine os modelos primeiro na sidebar")
    
    with tab3:
        st.subheader("🔮 Predição para Cliente Individual")
        
        if 'model' in st.session_state and st.session_state.model.models:
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.slider("Idade", 18, 80, 35)
                income = st.number_input("Renda Mensal (R$)", 1000, 50000, 5000)
                website_time = st.slider("Tempo no Site (min)", 0.1, 30.0, 5.0)
                page_views = st.slider("Páginas Visitadas", 1, 20, 5)
                previous_purchases = st.slider("Compras Anteriores", 0, 10, 2)
                email_opens = st.slider("Emails Abertos (mês)", 0, 10, 3)
                engagement_score = st.slider("Score Redes Sociais", 0.0, 10.0, 3.0)
            
            with col2:
                moon_phase = st.selectbox("Fase da Lua", ['Nova', 'Crescente', 'Cheia', 'Minguante'])
                season = st.selectbox("Estação", [0, 1, 2, 3], format_func=lambda x: ['Verão', 'Outono', 'Inverno', 'Primavera'][x])
                day_of_week = st.selectbox("Dia da Semana", list(range(7)), format_func=lambda x: ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom'][x])
                hour_of_visit = st.slider("Hora da Visita", 0, 23, 14)
                device_type = st.selectbox("Dispositivo", ['Mobile', 'Desktop', 'Tablet'])
                traffic_source = st.selectbox("Fonte de Tráfego", ['Organic', 'Paid', 'Social', 'Direct'])
                model_choice = st.selectbox("Modelo", list(st.session_state.model.models.keys()))
            
            if st.button("🎯 Fazer Predição"):
                customer_data = {
                    'age': age,
                    'income': income,
                    'website_time_minutes': website_time,
                    'page_views': page_views,
                    'previous_purchases': previous_purchases,
                    'email_opens_last_month': email_opens,
                    'social_media_engagement_score': engagement_score,
                    'moon_phase': moon_phase,
                    'season': season,
                    'day_of_week': day_of_week,
                    'hour_of_visit': hour_of_visit,
                    'device_type': device_type,
                    'traffic_source': traffic_source
                }
                
                prediction, probability, confidence = st.session_state.model.predict_purchase(
                    customer_data, model_choice
                )
                
                # Formata output conforme solicitado
                if confidence == '1':
                    conf_text = "1 (>90%)"
                elif confidence == '0':
                    conf_text = "0 (<10%)"
                else:
                    conf_text = "5 (~50%)"
                
                st.success(f"**Predição:** {prediction}")
                st.info(f"**Probabilidade:** {probability:.1%}")
                st.info(f"**Nível de Confiança:** {conf_text}")
        else:
            st.info("👆 Treine os modelos primeiro para fazer predições")
    
    with tab4:
        if 'reg_comparison' in st.session_state:
            st.subheader("🧠 Análises de ML Avançadas")
            
            # Comparação de regularização
            st.subheader("📊 Comparação L0, L1, L2")
            reg_df = pd.DataFrame(st.session_state.reg_comparison).T
            st.dataframe(reg_df)
            
            # Análise K-Means vs KNN
            st.subheader("🔍 K-Means vs KNN")
            cluster_info = st.session_state.cluster_analysis
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Pureza K-Means", f"{cluster_info['kmeans_purity']:.3f}")
            with col2:
                st.metric("Acurácia KNN", f"{cluster_info['knn_accuracy']:.3f}")
            
            # Feature Importance
            if hasattr(st.session_state.model, 'feature_importance') and st.session_state.model.feature_importance:
                st.subheader("📈 Importância das Features")
                importance_df = pd.DataFrame(
                    list(st.session_state.model.feature_importance.items()),
                    columns=['Feature', 'Importância']
                ).sort_values('Importância', ascending=False)
                
                fig, ax = plt.subplots()
                importance_df.head(10).plot(x='Feature', y='Importância', kind='bar', ax=ax)
                ax.set_title("Top 10 Features Mais Importantes")
                plt.xticks(rotation=45)
                st.pyplot(fig)
    
    with tab5:
        if st.session_state.data_generated:
            st.subheader("🌙 Análise do Impacto das Fases da Lua")
            
            df = st.session_state.df
            
            # Análise estatística
            moon_stats = df.groupby('moon_phase').agg({
                'purchase': lambda x: (x == 's').mean(),
                'income': 'mean',
                'website_time_minutes': 'mean',
                'page_views': 'mean'
            }).round(3)
            
            moon_stats.columns = ['Taxa de Compra', 'Renda Média', 'Tempo Médio no Site', 'Páginas Médias']
            st.dataframe(moon_stats)
            
            # Gráfico de correlação
            st.subheader("📊 Comportamento por Fase da Lua")
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Taxa de compra
            moon_purchase = df.groupby('moon_phase')['purchase'].apply(lambda x: (x == 's').mean())
            moon_purchase.plot(kind='bar', ax=axes[0,0], title="Taxa de Compra")
            axes[0,0].set_ylabel("Taxa")
            
            # Tempo no site
            df.groupby('moon_phase')['website_time_minutes'].mean().plot(
                kind='bar', ax=axes[0,1], title="Tempo Médio no Site"
            )
            axes[0,1].set_ylabel("Minutos")
            
            # Páginas visualizadas
            df.groupby('moon_phase')['page_views'].mean().plot(
                kind='bar', ax=axes[1,0], title="Páginas Médias"
            )
            axes[1,0].set_ylabel("Páginas")
            
            # Distribuição de renda
            df.groupby('moon_phase')['income'].mean().plot(
                kind='bar', ax=axes[1,1], title="Renda Média"
            )
            axes[1,1].set_ylabel("R$")
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Insights
            st.subheader("💡 Insights sobre Fases da Lua")
            best_moon_phase = moon_purchase.idxmax()
            worst_moon_phase = moon_purchase.idxmin()
            
            st.success(f"🌕 **Melhor fase:** {best_moon_phase} ({moon_purchase[best_moon_phase]:.1%} de conversão)")
            st.warning(f"🌑 **Pior fase:** {worst_moon_phase} ({moon_purchase[worst_moon_phase]:.1%} de conversão)")
            
            improvement = (moon_purchase[best_moon_phase] - moon_purchase[worst_moon_phase]) / moon_purchase[worst_moon_phase]
            st.info(f"📈 **Potencial de melhoria:** {improvement:.1%} focando na melhor fase da lua")

if __name__ == "__main__":
    main()
