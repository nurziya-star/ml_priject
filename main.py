# main.py
# Streamlit ML Laboratory — file-based version (design preserved)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="ML Laboratory", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
body {background-color:#0f172a; color:white;}
.big-title {font-size:42px; font-weight:800; color:#38bdf8;}
.section {border:2px solid #38bdf8; border-radius:20px; padding:20px; margin-bottom:25px;}
.star {color: gold; font-size:20px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🧪 Машиналық оқыту — 11 лаборатория</div>', unsafe_allow_html=True)

menu = [
    "1️⃣ Первичный анализ данных",
    "2️⃣ PCA",
    "3️⃣ Линейная регрессия",
    "4️⃣ Наивный Байес",
    "5️⃣ SVM",
    "6️⃣ Бустинг",
    "7️⃣ Нейронные сети",
    "8️⃣ Кластеризация",
    "9️⃣ Ассоциативные правила",
    "🔟 Онлайн обработка",
    "1️⃣1️⃣ Распределённые вычисления"
]

choice = st.sidebar.selectbox("📂 Лаборатория таңда", menu)

# ---------------- HELPERS ----------------
def stars(n):
    st.markdown("<div class='star'>" + "⭐"*n + "</div>", unsafe_allow_html=True)

# ---------------- LABS ----------------
if choice.startswith("1️⃣ "):
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.header("Первичный анализ наборов данных")
    stars(2)
    file = st.file_uploader("CSV файл жүкте", type="csv")
    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())
        st.write(df.describe())
        fig, ax = plt.subplots()
        df.hist(ax=ax)
        st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

elif choice.startswith("2️⃣ "):
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.header("Анализ главных компонент (PCA)")
    stars(3)
    file = st.file_uploader("CSV файл жүкте", type="csv")
    if file:
        df = pd.read_csv(file)
        X = StandardScaler().fit_transform(df.select_dtypes(include=np.number))
        pca = PCA(n_components=2)
        comps = pca.fit_transform(X)
        fig, ax = plt.subplots()
        ax.scatter(comps[:,0], comps[:,1])
        ax.set_title("PCA проекция")
        st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

elif choice.startswith("3️⃣ "):
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.header("Линейная регрессия")
    stars(3)
    file = st.file_uploader("CSV файл (X,y)", type="csv")
    if file:
        df = pd.read_csv(file)
        X = df.iloc[:,:-1]
        y = df.iloc[:,-1]
        model = LinearRegression()
        model.fit(X,y)
        preds = model.predict(X)
        fig, ax = plt.subplots()
        ax.scatter(y, preds)
        ax.set_xlabel("y true")
        ax.set_ylabel("y pred")
        st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

elif choice.startswith("4️⃣ "):
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.header("Наивный байесовский классификатор")
    stars(4)
    file = st.file_uploader("CSV файл (features + class)", type="csv")
    if file:
        df = pd.read_csv(file)
        X = df.iloc[:,:-1]
        y = df.iloc[:,-1]
        model = GaussianNB()
        model.fit(X,y)
        preds = model.predict(X)
        st.write("Accuracy:", accuracy_score(y,preds))
    st.markdown('</div>', unsafe_allow_html=True)

elif choice.startswith("5️⃣ "):
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.header("Support Vector Machine")
    stars(4)
    file = st.file_uploader("CSV файл", type="csv")
    if file:
        df = pd.read_csv(file)
        X = df.iloc[:,:-1]
        y = df.iloc[:,-1]
        model = SVC()
        model.fit(X,y)
        preds = model.predict(X)
        st.write("Accuracy:", accuracy_score(y,preds))
    st.markdown('</div>', unsafe_allow_html=True)

elif choice.startswith("6️⃣ "):
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.header("Бустинг (AdaBoost)")
    stars(5)
    file = st.file_uploader("CSV файл", type="csv")
    if file:
        df = pd.read_csv(file)
        X = df.iloc[:,:-1]
        y = df.iloc[:,-1]
        model = AdaBoostClassifier()
        model.fit(X,y)
        preds = model.predict(X)
        st.write("Accuracy:", accuracy_score(y,preds))
    st.markdown('</div>', unsafe_allow_html=True)

elif choice.startswith("7️⃣ "):
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.header("Нейронные сети")
    stars(5)
    st.info("Бұл лабораторияны келесі этапта (TensorFlow / PyTorch) қосуға болады")
    st.markdown('</div>', unsafe_allow_html=True)

elif choice.startswith("8️⃣ "):
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.header("Кластеризация (KMeans / EM)")
    stars(3)
    file = st.file_uploader("CSV файл", type="csv")
    algo = st.selectbox("Алгоритм", ["KMeans","EM"])
    if file:
        df = pd.read_csv(file)
        X = df.select_dtypes(include=np.number)
        if algo=="KMeans":
            labels = KMeans(n_clusters=3).fit_predict(X)
        else:
            labels = GaussianMixture(n_components=3).fit_predict(X)
        fig, ax = plt.subplots()
        ax.scatter(X.iloc[:,0], X.iloc[:,1], c=labels)
        st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

elif choice.startswith("9️⃣ "):
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.header("Ассоциативные правила")
    stars(4)
    st.info("Apriori үшін бинарлық transaction dataset қажет")
    st.markdown('</div>', unsafe_allow_html=True)

elif choice.startswith("🔟"):
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.header("Онлайн обработка данных")
    stars(2)
    st.info("Streaming / incremental learning демонстрация")
    st.markdown('</div>', unsafe_allow_html=True)

elif choice.startswith("1️⃣1️⃣"):
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.header("Распределённые вычисления")
    stars(5)
    st.info("Spark / Dask концепцияларын визуалды түсіндіру")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("✨ Файл жүктеу арқылы жұмыс істейтін интерактивті ML лаборатория")
