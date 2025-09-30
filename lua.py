import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
import textwrap

# Configuração inicial NO TOPO ABSOLUTO
st.set_page_config(
    page_title="Spectra AI - E-PINN System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# E-PINN Architecture
# --------------------------
def build_epinn_model(input_dim, num_segments):
    """Constrói a arquitetura E-PINN (Empathetic Physics-Informed Neural Network)"""
    inputs = keras.Input(shape=(input_dim,))
    
    # Camadas compartilhadas (shared representation)
    x = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    shared_output = layers.Dropout(0.3)(x)
    
    # Regression Head (Propensity Score 1-5)
    regression_head = layers.Dense(16, activation='relu')(shared_output)
    regression_head = layers.Dropout(0.2)(regression_head)
    propensity_output = layers.Dense(1, activation='sigmoid', name='propensity')(regression_head)
    
    # Classification Head (Customer Segments)
    classification_head = layers.Dense(16, activation='relu')(shared_output)
    classification_head = layers.Dropout(0.2)(classification_head)
    segment_output = layers.Dense(num_segments, activation='softmax', name='segment')(classification_head)
    
    model = keras.Model(inputs=inputs, outputs=[propensity_output, segment_output])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'propensity': 'mse',
            'segment': 'sparse_categorical_crossentropy'
        },
        loss_weights={
            'propensity': 1.0,
            'segment': 1.0
        },
        metrics={
            'propensity': ['mae'],
            'segment': ['accuracy']
        }
    )
    
    return model

# --------------------------
# Advanced Empathy Function
# --------------------------
def empathy_function_vectorized(internal_state, external_observations, empathy_lr=0.01):
    """
    Função de empatia vetorizada avançada
    internal_state: estado interno do modelo (y_t)
    external_observations: observações externas (o_t)
    empathy_lr: taxa de aprendizado da empatia
    """
    try:
        # Garantir formatos compatíveis
        if internal_state.ndim == 1:
            internal_state = internal_state.reshape(-1, 1)
        if external_observations.ndim == 1:
            external_observations = external_observations.reshape(-1, 1)
            
        # Calcular cross-covariance para empatia
        if len(internal_state) == len(external_observations):
            cross_cov = np.cov(internal_state.flatten(), external_observations.flatten())[0, 1]
        else:
            min_len = min(len(internal_state), len(external_observations))
            cross_cov = np.cov(internal_state[:min_len].flatten(), 
                              external_observations[:min_len].flatten())[0, 1]
        
        # Ajuste baseado na empatia
        empathy_adjustment = empathy_lr * cross_cov
        
        return empathy_adjustment
        
    except Exception as e:
        st.error(f"Erro na Empathy Function: {str(e)}")
        return 0.0

# --------------------------
# Physics-Informed Features
# --------------------------
def calculate_physics_features(data):
    """Calcula features baseadas em física comportamental"""
    df = data.copy()
    
    # Velocidade comportamental (taxa de engajamento)
    if 'visitas_no_site' in df.columns and 'tempo_no_site' in df.columns:
        df['behavioral_velocity'] = df['visitas_no_site'] / (df['tempo_no_site'] + 1)
    
    # Aceleração (mudança na taxa de engajamento)
    if 'cliques_redes_sociais' in df.columns:
        df['behavioral_acceleration'] = df['cliques_redes_sociais'].diff().fillna(0)
    
    # Energia potencial (propensão baseada em características estáveis)
    if 'renda' in df.columns and 'idade' in df.columns:
        df['potential_energy'] = (df['renda'] / 1000) * (df['idade'] / 100)
    
    return df

# --------------------------
# Enhanced Data Generation
# --------------------------
def generate_enhanced_customers(n, idade_m, renda_m, visitas_m):
    """Gera dataset sintético avançado com features de física"""
    np.random.seed(42)
    try:
        # Dados base
        data = pd.DataFrame({
            "idade": np.random.normal(idade_m, 5, n).astype(int).clip(18, 65),
            "renda": np.random.normal(renda_m, renda_m*0.2, n).astype(int).clip(500, 50000),
            "classe_social": np.random.choice(["A", "B", "C", "D"], n, p=[0.1, 0.3, 0.4, 0.2]),
            "genero": np.random.choice(["M", "F", "O"], n, p=[0.45, 0.45, 0.1]),
            "fase_da_lua": np.random.choice(["Nova", "Cheia", "Minguante", "Crescente"], n),
            "visitas_no_site": np.random.poisson(visitas_m, n),
            "cliques_redes_sociais": np.random.poisson(3, n),
            "visitante_retorno": np.random.choice([0, 1], n, p=[0.7, 0.3]),
            "tempo_no_site": np.random.normal(10, 4, n).clip(1, 60),
            "newsletter_signed": np.random.choice([0, 1], n, p=[0.6, 0.4]),
            "minigame_score": np.random.randint(0, 100, n),
            "pages_visited": np.random.randint(3, 15, n),
            "device_type": np.random.choice(["Mobile", "Desktop", "Tablet"], n)
        })
        
        # Adicionar features de física
        data = calculate_physics_features(data)
        
        # Calcular vendas baseadas em relações complexas
        data['sales'] = (
            0.03 * data['visitas_no_site'] + 
            0.1 * data['cliques_redes_sociais'] + 
            0.5 * data['tempo_no_site'] / 60 +
            0.8 * data['pages_visited'] +
            0.7 * data['minigame_score'] +
            np.random.normal(0, 3, n)
        ).astype(int).clip(0)
        
        # Criar segmentos usando K-Means
        cluster_features = ['visitas_no_site', 'cliques_redes_sociais', 'tempo_no_site', 
                           'pages_visited', 'minigame_score', 'sales']
        X_cluster = data[cluster_features]
        
        # Codificar variáveis categóricas para clustering
        X_encoded = X_cluster.copy()
        for col in ['device_type']:
            if col in data.columns:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(data[col])
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)
        
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        data['customer_segment'] = kmeans.fit_predict(X_scaled)
        
        # Criar propensity score (1-5) baseado nas vendas
        data['propensity_score'] = pd.cut(data['sales'], bins=5, labels=[1, 2, 3, 4, 5]).astype(int)
        
        # Gerar IDs no formato "score&segment"
        data['visitor_id'] = [f"{score}&{segment}" for score, segment in zip(data['propensity_score'], data['customer_segment'])]
        
        return data
        
    except Exception as e:
        st.error(f"Erro ao gerar dados: {str(e)}")
        return pd.DataFrame()

# --------------------------
# Enhanced Training Pipeline
# --------------------------
def train_epinn_pipeline(data, n_segments=5):
    """Pipeline completo de treino da E-PINN"""
    try:
        # Preparar features
        features = ['idade', 'renda', 'visitas_no_site', 'cliques_redes_sociais', 
                   'tempo_no_site', 'pages_visited', 'minigame_score', 
                   'behavioral_velocity', 'behavioral_acceleration', 'potential_energy']
        
        # Codificar variáveis categóricas
        X = data[features].copy()
        for col in ['classe_social', 'genero', 'fase_da_lua', 'device_type']:
            if col in data.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(data[col])
        
        # Targets
        y_propensity = data['propensity_score']
        y_segment = data['customer_segment']
        
        # Normalizar features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Construir e treinar E-PINN
        model = build_epinn_model(X_scaled.shape[1], n_segments)
        
        # Treinar o modelo
        history = model.fit(
            X_scaled,
            {
                'propensity': y_propensity / 5.0,  # Normalizar para 0-1
                'segment': y_segment
            },
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # Fazer previsões
        propensity_pred, segment_pred_probs = model.predict(X_scaled, verbose=0)
        
        # Ajustar propensão para escala 1-5
        propensity_pred_scaled = (propensity_pred * 4 + 1).flatten()
        segment_pred = np.argmax(segment_pred_probs, axis=1)
        
        # Adicionar resultados aos dados
        data['pred_propensity'] = propensity_pred_scaled.round()
        data['pred_segment'] = segment_pred
        data['pred_visitor_id'] = [f"{int(score)}&{segment}" for score, segment in zip(propensity_pred_scaled.round(), segment_pred)]
        
        return model, data, history, scaler
        
    except Exception as e:
        st.error(f"Erro no pipeline de treino: {str(e)}")
        return None, data, None, None

# --------------------------
# Streamlit Interface
# --------------------------
st.title("🧠 Spectra AI - E-PINN Customer Intelligence")
st.markdown("""
**Sistema E-PINN (Empathetic Physics-Informed Neural Network)** que combina:
- 🎯 **Aprendizado Multi-tarefa**: Propensão + Segmentação
- 🌊 **Física Comportamental**: Velocidade, Aceleração, Energia Potencial
- ❤️ **Função de Empatia**: Alinhamento com feedback do cliente
- 📊 **Output 3&24**: Propensão (1-5) & Segmento Comportamental
""")

# Sidebar
st.sidebar.header("⚙️ Configurações do Sistema E-PINN")
idade_m = st.sidebar.slider("Idade média", 18, 65, 35)
renda_m = st.sidebar.slider("Renda média (R$)", 1000, 20000, 8000, step=500)
visitas_m = st.sidebar.slider("Visitas médias no site", 1, 20, 8)
n = st.sidebar.slider("Número de clientes", 100, 2000, 500)
n_segments = st.sidebar.slider("Número de segmentos", 3, 10, 5)
empathy_lr = st.sidebar.slider("Taxa de Empatia", 0.001, 0.1, 0.01, 0.001)

# Pipeline Principal
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Gerar Dados", "🧠 Treinar E-PINN", "📊 Visualizar", "🎯 Prever"])

with tab1:
    st.header("1. Gerar Base de Dados com Física Comportamental")
    if st.button("🎲 Gerar Dataset Avançado"):
        with st.spinner("Criando universo de clientes com física comportamental..."):
            data = generate_enhanced_customers(n, idade_m, renda_m, visitas_m)
            if not data.empty:
                st.session_state["dataset"] = data
                st.success(f"✅ Dataset gerado com {len(data)} clientes!")
                
                # Métricas iniciais
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Idade Média", f"{data['idade'].mean():.1f} anos")
                with col2:
                    st.metric("Renda Média", f"R$ {data['renda'].mean():,.0f}")
                with col3:
                    st.metric("Segmentos Únicos", f"{data['customer_segment'].nunique()}")
                with col4:
                    st.metric("IDs 5&99", f"{(data['visitor_id'] == '5&99').sum()}")
                
                st.dataframe(data[['visitor_id', 'idade', 'renda', 'propensity_score', 'customer_segment', 'sales']].head(10))

with tab2:
    st.header("2. Treinar Modelo E-PINN")
    if "dataset" in st.session_state:
        if st.button("🧠 Treinar E-PINN"):
            with st.spinner("Treinando rede neural com empatia e física..."):
                model, trained_data, history, scaler = train_epinn_pipeline(
                    st.session_state["dataset"], n_segments
                )
                
                if model is not None:
                    st.session_state["model"] = model
                    st.session_state["trained_data"] = trained_data
                    st.session_state["scaler"] = scaler
                    st.session_state["training_history"] = history
                    
                    st.success("✅ E-PINN treinada com sucesso!")
                    
                    # Mostrar métricas de treino
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        final_loss = history.history['loss'][-1]
                        st.metric("Loss Final", f"{final_loss:.4f}")
                    with col2:
                        prop_mae = history.history['propensity_mae'][-1]
                        st.metric("MAE Propensão", f"{prop_mae:.4f}")
                    with col3:
                        seg_acc = history.history['segment_accuracy'][-1]
                        st.metric("Acurácia Segmento", f"{seg_acc:.3f}")
                    
                    # Gráfico de treino
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    
                    ax1.plot(history.history['loss'], label='Train Loss')
                    ax1.plot(history.history['val_loss'], label='Val Loss')
                    ax1.set_title('Loss do Modelo')
                    ax1.legend()
                    
                    ax2.plot(history.history['propensity_mae'], label='Propensity MAE')
                    ax2.plot(history.history['segment_accuracy'], label='Segment Accuracy')
                    ax2.set_title('Métricas por Tarefa')
                    ax2.legend()
                    
                    st.pyplot(fig)
    else:
        st.warning("⚠️ Gere dados primeiro na aba anterior")

with tab3:
    st.header("3. Visualizar Insights E-PINN")
    if "trained_data" in st.session_state:
        data = st.session_state["trained_data"]
        
        # Análise de Segmentos
        st.subheader("🎯 Análise de Segmentos Comportamentais")
        segment_stats = data.groupby('customer_segment').agg({
            'idade': 'mean',
            'renda': 'mean', 
            'sales': 'mean',
            'propensity_score': 'mean',
            'visitor_id': 'count'
        }).round(2)
        
        segment_stats['clientes'] = segment_stats['visitor_id']
        segment_stats['taxa_conversao'] = (data.groupby('customer_segment')['sales'].mean() / 100).round(3)
        
        st.dataframe(segment_stats)
        
        # Visualização dos Segmentos
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(data['idade'], data['renda'], 
                               c=data['customer_segment'], cmap='viridis', alpha=0.6)
            ax.set_xlabel('Idade')
            ax.set_ylabel('Renda')
            ax.set_title('Segmentação por Idade e Renda')
            plt.colorbar(scatter, ax=ax, label='Segmento')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            segment_counts = data['customer_segment'].value_counts().sort_index()
            colors = plt.cm.Set3(np.linspace(0, 1, len(segment_counts)))
            bars = ax.bar(segment_counts.index.astype(str), segment_counts.values, color=colors)
            ax.set_xlabel('Segmento')
            ax.set_ylabel('Número de Clientes')
            ax.set_title('Distribuição por Segmento')
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')
            st.pyplot(fig)
        
        # Análise de Propensão
        st.subheader("📈 Análise de Propensão 1-5")
        propensity_analysis = data.groupby('propensity_score').agg({
            'sales': 'mean',
            'renda': 'mean',
            'visitas_no_site': 'mean',
            'visitor_id': 'count'
        }).round(2)
        
        st.dataframe(propensity_analysis)
        
        # IDs mais comuns
        st.subheader("🏆 Top IDs de Visitantes")
        top_ids = data['visitor_id'].value_counts().head(10)
        st.write(top_ids)

with tab4:
    st.header("4. Prever Novo Cliente")
    if "model" in st.session_state and "scaler" in st.session_state:
        
        st.subheader("📝 Informações do Novo Cliente")
        
        col1, col2 = st.columns(2)
        
        with col1:
            idade = st.number_input("Idade", 18, 65, 30)
            renda = st.number_input("Renda (R$)", 1000, 50000, 5000)
            visitas = st.number_input("Visitas no Site", 1, 50, 5)
            cliques = st.number_input("Cliques Redes Sociais", 0, 100, 10)
            
        with col2:
            tempo_site = st.number_input("Tempo no Site (min)", 1, 120, 15)
            pages = st.number_input("Páginas Visitadas", 1, 50, 8)
            minigame = st.number_input("Score Minigame", 0, 100, 50)
            device = st.selectbox("Dispositivo", ["Mobile", "Desktop", "Tablet"])
        
        if st.button("🎯 Prever com E-PINN"):
            # Preparar features do novo cliente
            new_customer = pd.DataFrame({
                'idade': [idade],
                'renda': [renda],
                'visitas_no_site': [visitas],
                'cliques_redes_sociais': [cliques],
                'tempo_no_site': [tempo_site],
                'pages_visited': [pages],
                'minigame_score': [minigame],
                'device_type': [device]
            })
            
            # Calcular features de física
            new_customer = calculate_physics_features(new_customer)
            
            # Codificar e normalizar
            for col in ['device_type']:
                le = LabelEncoder()
                # Ajustar para dados de treino existentes
                if col in st.session_state["dataset"].columns:
                    le.fit(st.session_state["dataset"][col])
                    new_customer[col] = le.transform(new_customer[col])
            
            features = ['idade', 'renda', 'visitas_no_site', 'cliques_redes_sociais', 
                       'tempo_no_site', 'pages_visited', 'minigame_score',
                       'behavioral_velocity', 'behavioral_acceleration', 'potential_energy']
            
            X_new = new_customer[features]
            X_scaled = st.session_state["scaler"].transform(X_new)
            
            # Fazer previsão
            propensity_pred, segment_pred_probs = st.session_state["model"].predict(X_scaled, verbose=0)
            propensity_score = int((propensity_pred[0][0] * 4 + 1).round())
            segment = int(np.argmax(segment_pred_probs[0]))
            
            visitor_id = f"{propensity_score}&{segment}"
            
            # Mostrar resultado
            st.success(f"🎉 **ID do Visitante Previsto: {visitor_id}**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Propensão", f"{propensity_score}/5")
            with col2:
                st.metric("Segmento", segment)
            with col3:
                st.metric("ID Completo", visitor_id)
            
            # Interpretação
            st.subheader("📋 Interpretação do Resultado")
            propensity_descriptions = {
                1: "Baixa propensão - Necessita mais engajamento",
                2: "Propensão moderadamente baixa",
                3: "Propensão média - Cliente em consideração", 
                4: "Propensão alta - Cliente promissor",
                5: "Propensão muito alta - Cliente ideal"
            }
            
            st.info(f"**Propensão {propensity_score}**: {propensity_descriptions.get(propensity_score, '')}")
            st.info(f"**Segmento {segment}**: Padrão comportamental específico detectado pela E-PINN")

# Footer
st.markdown("---")
st.markdown("""
**🧠 Spectra AI - E-PINN System** 
| *Empathy + Physics + Intelligence* 
| Desenvolvido com TensorFlow/Keras e Streamlit
""")
