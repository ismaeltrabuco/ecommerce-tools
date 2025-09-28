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
import textwrap

# Configuração inicial NO TOPO ABSOLUTO
st.set_page_config(
    page_title="The Moon AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# Função Empatia
# --------------------------
def empathy_function(prob):
    return np.clip(prob * 0.9 + 0.05, 0.1, 0.9)

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
# Treino e score
# --------------------------
def train_and_score(data, n_clusters=6):
    features = data.drop(columns=["comprou"])
    target = data["comprou"]
    encoded = features.copy()
    le = LabelEncoder()  # Instanciar uma vez pra consistência
    for col in encoded.select_dtypes(include="object").columns:
        encoded[col] = le.fit_transform(encoded[col])
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(encoded, target)
    probs = model.predict_proba(encoded)
    max_probs = probs.max(axis=1)
    scaled_scores = (max_probs * 5).round().astype(int)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(encoded)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    data["cluster"] = clusters
    cluster_profiles = data.groupby("cluster")[["idade", "renda", "visitas_no_site", "comprou"]].mean().round(2)
    cluster_names = {}
    for c, row in cluster_profiles.iterrows():
        compra_pct = int(row['comprou'] * 100)
        cluster_names[c] = f"Cluster {c+1} - Idade {int(row['idade'])}, Renda R${int(row['renda'])}, Visitas {int(row['visitas_no_site'])}, Comprou {compra_pct}%"
    data["score_final"] = [f"{s} & {cluster_names[c]}" for s, c in zip(scaled_scores, clusters)]
    return model, data, encoded.columns, clusters, cluster_names

# --------------------------
# Interface
# --------------------------
st.title("💎 The Moon AI - Ilumine os Dados do Seu Negócio")
st.markdown("""
Nossos modelos usam a **Empathy Function** para entender clientes antes de decidir.
""")

st.sidebar.header("⚙️ Configurações do Público")
idade_m = st.sidebar.slider("Idade média", 18, 65, 30)
renda_m = st.sidebar.slider("Renda média (R$)", 1000, 20000, 5000, step=500)
visitas_m = st.sidebar.slider("Visitas médias no site", 1, 20, 5)
n = st.sidebar.slider("Número de clientes", 50, 1000, 200)

# --------------------------
# Gerar Dados
# --------------------------
st.header("1️⃣ Gere um Banco de Dados")
if st.button("Gerar Dados"):
    try:
        data = generate_customers(n, idade_m, renda_m, visitas_m)
        st.session_state["df"] = data
        st.success("✅ Banco de dados gerado!")
        st.dataframe(data.head())
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Idade Média", f"{data['idade'].mean():.1f} anos")
        with col2:
            st.metric("Renda Média", f"R$ {data['renda'].mean():,.0f}")
        with col3:
            st.metric("Taxa de Compra", f"{(data['comprou']==1).mean()*100:.1f}%")
    except Exception as e:
        st.error(f"Erro ao gerar dados: {str(e)}")

# --------------------------
# Treinar Modelo
# --------------------------
if "df" in st.session_state:
    st.header("2️⃣ Treine o Modelo")
    if st.button("Treinar Agora"):
        try:
            with st.spinner("Aprendendo com seu público..."):
                time.sleep(2)
                model, scored_data, feat_names, clusters, cluster_names = train_and_score(st.session_state["df"])
                # Aplicar o mesmo encoding ao scored_data pra visualização
                for col in scored_data.select_dtypes(include="object").columns:
                    scored_data[col] = LabelEncoder().fit_transform(scored_data[col])
                st.session_state["model"] = model
                st.session_state["scored"] = scored_data
                st.session_state["feat_names"] = feat_names  # Salvar feat_names
                st.session_state["clusters"] = clusters
                st.session_state["cluster_names"] = cluster_names
            st.success("✅ Modelo treinado com sucesso!")
        except Exception as e:
            st.error(f"Erro ao treinar modelo: {str(e)}")

# --------------------------
# Visualização
# --------------------------
if "scored" in st.session_state:
    try:
        scored_data = st.session_state["scored"]
        cluster_names = st.session_state["cluster_names"]
        feat_names = st.session_state["feat_names"]  # Usar feat_names salvo
        st.header("3️⃣ Visualize os Resultados")

        # Insights Lunares (Contos Lunares)
        st.subheader("🌙 Insights Lunares")
        imp_df = pd.DataFrame({"feature": feat_names, "importance": st.session_state["model"].feature_importances_}).sort_values("importance", ascending=False)
        for _, row in imp_df.head(3).iterrows():
            corr = scored_data[row["feature"]].corr(scored_data["comprou"])
            insight = f"🌕 **Insight Lunar**: '{row['feature']}' brilha com {row['importance']:.2f} de importância! Sua correlação com 'comprou' é {corr:.2f}. Considere focar em {row['feature']} para aumentar compras — teste um aumento de 10%!"
            st.write(textwrap.fill(insight, width=70))

        # Interatividade do Mapa da Lua (Seleção de Clusters)
        st.subheader("📍 Mapa da Lua - Explore Clusters")
        selected_cluster = st.selectbox("Escolha um Cluster", options=cluster_names.values())
        cluster_id = [k for k, v in cluster_names.items() if v == selected_cluster][0]
        cluster_data = scored_data[scored_data["cluster"] == cluster_id]
        st.dataframe(cluster_data[["idade", "renda", "visitas_no_site", "comprou"]].describe())
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(cluster_data["idade"], cluster_data["renda"], c=cluster_data["comprou"], cmap="RdYlGn")
        ax.set_title(f"Cluster {cluster_id} - Distribuição")
        st.pyplot(fig)

        # Importância das features
        st.subheader("🔥 Importância das Features")
        st.bar_chart(imp_df.set_index("feature"))

        # PCA
        encoded = scored_data.drop(columns=["comprou", "score_final", "cluster"])
        X_scaled = StandardScaler().fit_transform(encoded)  # Já encodado, só escalar
        pcs = PCA(n_components=2).fit_transform(X_scaled)
        st.subheader("📌 PCA com Cluster e Compra")
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(pcs[:, 0], pcs[:, 1], c=scored_data["cluster"], s=60, alpha=0.7, cmap='tab10')
        ax.set_title("Mapa de Clientes por PCA e Cluster")
        ax.set_xlabel("Componente Principal 1")
        ax.set_ylabel("Componente Principal 2")
        plt.colorbar(scatter, ax=ax, label="Cluster")
        st.pyplot(fig)

        # Clusters como bolhas
        st.subheader("📊 Clusters como Bolhas")
        cluster_stats = scored_data.groupby("cluster").agg(
            x=('idade', 'mean'), y=('renda', 'mean'), pct_comprou=('comprou', 'mean'), clientes=('cluster', 'count')
        ).reset_index()
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        scatter2 = ax2.scatter(
            cluster_stats['x'], cluster_stats['y'], s=cluster_stats['clientes'] * 3, c=cluster_stats['pct_comprou'],
            alpha=0.7, cmap='RdYlGn', edgecolors='black'
        )
        ax2.set_title("Clusters de Clientes - Tamanho e Taxa de Compra")
        ax2.set_xlabel("Idade média")
        ax2.set_ylabel("Renda média")
        plt.colorbar(scatter2, ax=ax2, label="Taxa de Compra")
        st.pyplot(fig2)

        # Análise de quem comprou
        st.subheader("🛍️ Quem Comprou e o que têm em comum")
        features_comp = ["idade", "renda", "visitas_no_site", "tempo_no_site", "cliques_redes_sociais"]
        medias = scored_data.groupby("comprou")[features_comp].mean().round(2)
        st.dataframe(medias)
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        colors = {1: "green", 0: "orange", -1: "red"}
        for compra_status in scored_data["comprou"].unique():
            subset = scored_data[scored_data["comprou"] == compra_status]
            ax3.scatter(subset["idade"], subset["renda"], c=colors.get(compra_status, 'blue'), s=subset["visitas_no_site"] * 10,
                        alpha=0.6, label=f"Comprou: {compra_status}")
        ax3.set_title("Distribuição de Clientes Compradores vs Não Compradores")
        ax3.set_xlabel("Idade")
        ax3.set_ylabel("Renda")
        ax3.legend()
        st.pyplot(fig3)
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        medias.T.plot(kind="bar", ax=ax4)
        ax4.set_title("Médias das Features por Grupo de Compra")
        ax4.set_xlabel("Feature")
        ax4.set_ylabel("Média")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig4)

        # Resumo de Clusters
        st.subheader("📋 Resumo de Clusters")
        cluster_summary = scored_data.groupby("cluster").agg(
            Clientes=('cluster', 'count'), Comprou=('comprou', 'mean')
        ).reset_index()
        cluster_summary["Percentual"] = (cluster_summary["Clientes"] / cluster_summary["Clientes"].sum()) * 100
        cluster_summary["Comprou"] = cluster_summary["Comprou"].round(3)
        cluster_summary["Percentual"] = cluster_summary["Percentual"].round(1)
        st.dataframe(cluster_summary)

    except Exception as e:
        st.error(f"Erro na visualização: {str(e)}")
        st.write("Detalhes do erro:")
        st.exception(e)

# Footer (mantido intacto)
st.markdown("---")
st.markdown("💡 **The Moon AI** - Transformando dados em insights emocionais")
