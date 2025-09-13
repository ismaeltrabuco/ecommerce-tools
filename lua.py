import streamlit as st
import pandas as pd
import numpy as np
import time

st.set_page_config(page_title="Empathy Data AI", layout="wide")

# ======================================================
# Núcleo: Empathy Step (do paper, simplificado)
# ======================================================
def empathy_step(y_t, o_t, lam=0.01):
    """
    Atualiza o estado y_t em direção ao alinhamento com o "outro" o_t.
    y_{t+1} = y_t + λ * cov(y_t, o_t)
    """
    y_t = np.asarray(y_t, dtype=float)
    o_t = np.asarray(o_t, dtype=float)
    cov = np.cov(y_t, o_t, bias=True)[0, 1]
    return y_t + lam * cov

# ======================================================
# Cabeçalho
# ======================================================
st.title("💎 Empathy Data AI")
st.markdown("""
🚀 Seu e-commerce pode vender mais.  
Unimos a **inteligência das máquinas** à **empatia humana** para transformar dados em estratégias.  
""")

# ======================================================
# Sidebar Configurações do Público
# ======================================================
st.sidebar.header("⚙️ Configurações do Público")
idade = st.sidebar.slider("Idade média", 18, 65, 30)
renda = st.sidebar.slider("Renda média (R$)", 1000, 20000, 5000, step=500)
classe = st.sidebar.selectbox("Classe social predominante", ["A", "B", "C", "D", "E"])
genero = st.sidebar.selectbox("Gênero predominante", ["Masculino", "Feminino", "Misto"])
n_samples = st.sidebar.slider("Número de clientes simulados", 50, 1000, 200)

# ======================================================
# 1. Gerar Banco de Dados
# ======================================================
st.header("📊 Gerar Banco de Dados")

if st.button("Gerar Dados"):
    df = pd.DataFrame({
        "idade": np.random.normal(idade, 5, n_samples).astype(int),
        "renda": np.random.normal(renda, renda * 0.2, n_samples).astype(int),
        "classe": np.random.choice([classe, "outros"], size=n_samples, p=[0.7, 0.3]),
        "genero": np.random.choice([genero, "outros"], size=n_samples, p=[0.8, 0.2]),
        "comprou": np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    })

    st.success("✅ Banco de dados gerado com sucesso!")
    st.dataframe(df.head())

    # Cards com médias
    col1, col2, col3 = st.columns(3)
    col1.metric("Idade Média", f"{df['idade'].mean():.1f} anos")
    col2.metric("Renda Média", f"R$ {df['renda'].mean():,.0f}")
    col3.metric("Taxa de Compra", f"{df['comprou'].mean() * 100:.1f}%")

    st.caption("📊 Esses dados representam o **perfil médio da sua clientela**. É com base neles que vamos treinar o modelo.")
    st.session_state["df"] = df

# ======================================================
# 2. Treinar Modelo (simulado + Empathy Step)
# ======================================================
if "df" in st.session_state:
    st.header("🤖 Treinar Modelo")

    if st.button("Treinar Agora"):
        with st.spinner("Aprendendo com seu público e histórico de vendas..."):
            time.sleep(2)  # simulação

            df = st.session_state["df"]
            # Empathy update: usa visitas simuladas e compras
            y_t = df["idade"].values
            o_t = df["comprou"].values
            y_next = empathy_step(y_t, o_t)

            acc = np.random.uniform(0.7, 0.95)
            feat_importance = "Idade" if idade > 40 else "Renda"

        st.success("✅ Modelo treinado com sucesso!")
        col1, col2 = st.columns(2)
        col1.metric("Acurácia (simulada)", f"{acc*100:.1f}%")
        col2.metric("Feature mais importante", feat_importance)

        st.caption("🧠 Nosso modelo foi previamente treinado por especialistas e agora está **aprendendo com os seus dados simulados**: quem comprou, o que vendeu e quando.")
        st.session_state["y_next"] = y_next

# ======================================================
# 3. Visualizar Resultados
# ======================================================
if "df" in st.session_state and "y_next" in st.session_state:
    st.header("✨ Visualizar Resultados")
    st.markdown("""
    Agora vamos ver o que o modelo aprendeu e pode nos ensinar.  
    Aqui estão os **padrões ocultos** que unem inteligência de máquina e empatia humana.
    """)

    df = st.session_state["df"]
    compras_por_classe = df.groupby("classe")["comprou"].mean()

    st.bar_chart(compras_por_classe)

    # Insights empáticos automáticos
    st.subheader("💡 Insights Empáticos")
    st.markdown(f"- 🌙 Seu público de classe **{classe}** tem taxa de compra de {compras_por_classe[classe]*100:.1f}%.")
    st.markdown(f"- 👥 A idade média do seu público é **{idade} anos** — invista em comunicação personalizada.")
    st.markdown("- 🚀 Produtos de maior valor atraem melhor o segmento com renda mais alta.")
    st.markdown("- 🤝 O modelo aplicou a *Empathy Function*, alinhando perfis de compra com características do público.")
