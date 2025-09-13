import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import time

st.set_page_config(page_title="Empathy Data AI", layout="wide")

# --------------------------
# Função Empatia
# --------------------------
def empathy_function(prob):
    return np.clip(prob * 0.9 + 0.05, 0, 1)

# --------------------------
# Geração de dataset sintético
# --------------------------
def generate_customers(n, idade_m, renda_m, visitas_m):
    np.random.seed(42)
    data = pd.DataFrame({
        "idade": np.random.normal(idade_m, 5, n).astype(int).clip(18, 65),
        "renda": np.random.normal(renda_m, renda_m*0.2, n).astype(int).clip(500, 50000),
        "classe_social": np.random.choice(["A", "B", "C", "D"], n),
        "genero": np.random.choice(["M", "F", "O"], n),
        "fase_da_lua": np.random.choice(["Nova", "Cheia", "Minguante", "Crescente"], n),
        "visitas_no_site": np.random.poisson(visitas_m, n),
        "cliques_redes_sociais": np.random.poisson(3, n),
        "visitante_retorno": np.random.choice([0, 1], n),
        "tempo_no_site": np.random.normal(10, 4, n).clip(1, 60),
        "newsletter_signed": np.random.choice([0, 1], n)
    })

    # Variável alvo
    probs = (
        0.3 * (data["classe_social"].map({"A": 0.8, "B": 0.6, "C": 0.4, "D": 0.2}))
        + 0.2 * data["visitante_retorno"]
        + 0.2 * data["newsletter_signed"]
        + 0.1 * (data["visitas_no_site"] / (1 + data["visitas_no_site"].max()))
    )
    probs = empathy_function(probs)
    y = np.where(probs > 0.6, 1, np.where(probs < 0.3, -1, 0))
    data["comprou"] = y

    return data

# --------------------------
# Treino e score (versão legível e segmentada)
# --------------------------
def train_and_score(data, n_clusters=6):
    features = data.drop(columns=["comprou"])
    target = data["comprou"]

    # Encode features categóricas
    encoded = features.copy()
    for col in encoded.select_dtypes(include="object").columns:
        encoded[col] = LabelEncoder().fit_transform(encoded[col])

    # Modelo RandomForest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(encoded, target)

    # Probabilidades e score
    probs = model.predict_proba(encoded)
    max_probs = probs.max(axis=1)
    scaled_scores = (max_probs * 5).round().astype(int)

    # KMeans → clusters
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(encoded)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(X_scaled)
    data["cluster"] = clusters

    # Criação de nomes legíveis para clusters (incluindo taxa de compra)
    cluster_profiles = data.groupby("cluster")[["idade","renda","visitas_no_site","comprou"]].mean().round(2)
    cluster_names = {}
    for c, row in cluster_profiles.iterrows():
        compra_pct = int(row['comprou']*100)
        cluster_names[c] = f"Cluster {c+1} - Idade {int(row['idade'])}, Renda R${int(row['renda'])}, Visitas {int(row['visitas_no_site'])}, Comprou {compra_pct}%"

    # Score final interpretável
    data["score_final"] = [f"{s} & {cluster_names[c]}" for s, c in zip(scaled_scores, clusters)]

    return model, data, encoded.columns, clusters, cluster_names

# --------------------------
# Narrativa inicial
# --------------------------
st.title("💎 Seu negócio precisa de Inteligência Artificial")
st.markdown("""
Mas não precisa te perder, precisa manter a humanidade e sua inteligência.

Nossos modelos são equipados com a inovadora **Empathy Function** que sente quem é você, seu negócio e seus clientes, antes de responder às suas perguntas.

Você pode se surpreender com as informações que a inteligência artificial, aliada ao humanismo e natural, pode fazer pelo seu negócio e por todos nós enquanto sociedade.
""")

st.sidebar.header("⚙️ Configurações do Público")
idade_m = st.sidebar.slider("Idade média", 18, 65, 30)
renda_m = st.sidebar.slider("Renda média (R$)", 1000, 20000, 5000, step=500)
visitas_m = st.sidebar.slider("Visitas médias no site", 1, 20, 5)
n = st.sidebar.slider("Número de clientes", 50, 1000, 200)

# --------------------------
# 1. Gerar Dados
# --------------------------
st.header("1️⃣ Gere um Banco de Dados")
if st.button("Gerar Dados"):
    data = generate_customers(n, idade_m, renda_m, visitas_m)
    st.session_state["df"] = data
    st.success("✅ Banco de dados gerado!")
    st.dataframe(data.head())
    st.metric("Idade Média", f"{data['idade'].mean():.1f} anos")
    st.metric("Renda Média", f"R$ {data['renda'].mean():,.0f}")
    st.metric("Taxa de Compra", f"{(data['comprou']==1).mean()*100:.1f}%")

# --------------------------
# 2. Treinar Modelo
# --------------------------
if "df" in st.session_state:
    st.header("2️⃣ Treine o Modelo")
    if st.button("Treinar Agora"):
        with st.spinner("Aprendendo com seu público..."):
            time.sleep(2)
            model, scored_data, feat_names, clusters, cluster_names = train_and_score(st.session_state["df"])
            st.session_state["model"] = model
            st.session_state["scored"] = scored_data
            st.session_state["clusters"] = clusters
            st.session_state["cluster_names"] = cluster_names
        st.success("✅ Modelo treinado com sucesso!")

# --------------------------
# 3. Visualizar Resultados
# --------------------------
if "scored" in st.session_state:
    scored_data = st.session_state["scored"]
    cluster_names = st.session_state["cluster_names"]

    st.header("3️⃣ Visualize os Resultados")

    # Importância das features
    importances = st.session_state["model"].feature_importances_
    feat_names = scored_data.drop(columns=["comprou", "score_final", "cluster"]).columns
    imp_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False)
    st.subheader("🔥 Importância das Features")
    st.bar_chart(imp_df.set_index("feature"))

    # PCA para visualização
    encoded = scored_data.drop(columns=["comprou", "score_final", "cluster"])
    for col in encoded.select_dtypes(include="object").columns:
        encoded[col] = LabelEncoder().fit_transform(encoded[col])
    X_scaled = StandardScaler().fit_transform(encoded)
    pcs = PCA(n_components=2).fit_transform(X_scaled)

    fig, ax = plt.subplots()
    sns.scatterplot(
        x=pcs[:,0], y=pcs[:,1],
        hue=scored_data["comprou"],
        palette={1:"green", 0:"orange", -1:"red"},
        alpha=0.7
    )
    ax.set_title("Mapa de Clientes por PCA")
    st.pyplot(fig)

    # Scores finais
    st.subheader("📝 Como ler o score final")
    st.markdown("""
    - Formato: `Score & Cluster`
    - Exemplo: `5 & Cluster 1 - Idade 32, Renda R$5200, Visitas 5, Comprou 70%`
    - Score 5 → comprador mais provável, 0 → indefinido
    - Cluster → mostra perfil médio de idade, renda, visitas e taxa de compra do grupo
    - A segmentação permite diversidade maior, mantendo foco no público médio comprador
    """)

# --------------------------
# Tabela automática de clusters
# --------------------------
st.subheader("📋 Resumo de Clusters")
cluster_summary = scored_data.groupby("cluster").agg(
    Clientes=('cluster', 'count'),
    Comprou=('comprou','mean')
).reset_index()
cluster_summary["Percentual"] = (cluster_summary["Clientes"] / cluster_summary["Clientes"].sum()) * 100
st.dataframe(cluster_summary)
