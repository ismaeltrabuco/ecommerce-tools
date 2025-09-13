import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# --------------------------
# Função Empatia (paper empathy function)
# --------------------------
def empathy_function(prob):
    """
    Ajusta a probabilidade do modelo para refletir mais "empatia".
    Ex: evita classificações extremas em populações diversas.
    """
    return np.clip(prob * 0.9 + 0.05, 0, 1)  # suaviza extremos


# --------------------------
# Geração de dataset sintético
# --------------------------
def generate_customers(n=200):
    np.random.seed(42)
    data = pd.DataFrame({
        "idade": np.random.randint(18, 65, n),
        "classe_social": np.random.choice(["A", "B", "C", "D"], n),
        "genero": np.random.choice(["M", "F", "O"], n),
        "fase_da_lua": np.random.choice(["Nova", "Cheia", "Minguante", "Crescente"], n),
        "visitas_no_site": np.random.poisson(5, n),
        "cliques_redes_sociais": np.random.poisson(3, n),
        "visitante_retorno": np.random.choice([0, 1], n),
        "tempo_no_site": np.random.normal(10, 4, n).clip(1, 60),
        "newsletter_signed": np.random.choice([0, 1], n)
    })

    # Variável alvo sintética
    probs = (
        0.3 * (data["classe_social"].map({"A": 0.8, "B": 0.6, "C": 0.4, "D": 0.2}))
        + 0.2 * (data["visitante_retorno"])
        + 0.2 * (data["newsletter_signed"])
        + 0.1 * (data["visitas_no_site"] / (1 + data["visitas_no_site"].max()))
    )

    probs = empathy_function(probs)  # aplica função de empatia
    y = np.where(probs > 0.6, 1, np.where(probs < 0.3, -1, 0))
    data["comprou"] = y

    return data


# --------------------------
# Treino + Scoring
# --------------------------
def train_and_score(data):
    features = data.drop(columns=["comprou"])
    target = data["comprou"]

    # Encode variáveis categóricas
    encoded = features.copy()
    for col in encoded.select_dtypes(include="object").columns:
        encoded[col] = LabelEncoder().fit_transform(encoded[col])

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(encoded, target)

    probs = model.predict_proba(encoded)
    max_probs = probs.max(axis=1)
    scaled_scores = (max_probs * 5).round().astype(int)

    # Clusterização para gerar classes ASCII (XX)
    kmeans = KMeans(n_clusters=6, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(encoded)

    ascii_classes = []
    for i, cluster in enumerate(clusters):
        letter = chr(65 + (cluster % 26))  # A, B, C...
        number = (cluster // 26) + 1
        ascii_classes.append(f"{letter}{number}")

    data["score_final"] = [
        f"{s}&{c}" for s, c in zip(scaled_scores, ascii_classes)
    ]

    return model, data, encoded.columns


# --------------------------
# Streamlit App
# --------------------------
st.title("📊 Inteligência de Vendas com Empatia")

st.markdown("""
Seu e-commerce pode vender mais **direcionando investimentos e estratégias onde realmente importa**.  
Aqui você gera um banco de dados dos seus clientes, treina um modelo e descobre **o ouro escondido nos seus dados**.  
A novidade? Unimos a **inteligência das máquinas** à **empatia humana**, para que todos saiam ganhando.
""")

n = st.slider("Número de clientes", 50, 1000, 200)
data = generate_customers(n)
model, scored_data, feat_names = train_and_score(data)

# Mostrar tabela
st.subheader("📋 Amostra dos Clientes")
st.dataframe(scored_data.head(10))

# Importância das features
st.subheader("🔥 Importância das Features")
importances = model.feature_importances_
imp_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False)

fig, ax = plt.subplots()
sns.barplot(data=imp_df, x="importance", y="feature", ax=ax)
st.pyplot(fig)

# Visualização grupos
st.subheader("🎯 Visualização dos Grupos")
fig2, ax2 = plt.subplots()
sns.scatterplot(
    data=scored_data, x="idade", y="visitas_no_site",
    hue="comprou", palette={1: "green", 0: "orange", -1: "red"},
    alpha=0.7
)
ax2.set_title("Distribuição: Compradores (1), Duvidosos (0), Não Compradores (-1)")
st.pyplot(fig2)

# Legenda explicativa
st.markdown("""
### 📝 Como ler o score final
- `5&A1 = 5 (Comprador quase certo) & A1 = semelhante a 20% de outros clientes`  
- `0&F3 = 0 (Indefinido) & F3 = semelhante a 40% de outros clientes`  

Isso ajuda a **rotular grupos semelhantes** e entender melhor o comportamento do público.
""")
