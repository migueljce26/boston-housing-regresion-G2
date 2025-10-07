
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Boston Housing - Regresión", layout="wide")
st.title('Boston Housing - Análisis y Regresión')

# Cargar el dataset
@st.cache_data
def load_data():
    return pd.read_csv('housing.csv')
df = load_data()

st.header('Estadísticas Descriptivas')
st.dataframe(df.describe())

# Visualizaciones
st.header('Visualizaciones de Variables')
col1, col2 = st.columns(2)
with col1:
    st.subheader('Histograma de Precios (MEDV)')
    fig1, ax1 = plt.subplots()
    ax1.hist(df['MEDV'], bins=30, color='skyblue')
    ax1.set_xlabel('MEDV')
    ax1.set_ylabel('Frecuencia')
    st.pyplot(fig1)
with col2:
    st.subheader('Matriz de Correlación')
    fig2, ax2 = plt.subplots(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

st.header('Regresión Lineal Simple')
feature = st.selectbox('Selecciona la variable predictora:', [col for col in df.columns if col != 'MEDV'])
X = df[[feature]]
y = df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr_simple = LinearRegression()
lr_simple.fit(X_train, y_train)
y_pred_simple = lr_simple.predict(X_test)
st.write(f"MSE: {mean_squared_error(y_test, y_pred_simple):.2f}")
st.write(f"R2: {r2_score(y_test, y_pred_simple):.2f}")
fig3, ax3 = plt.subplots()
ax3.scatter(X_test, y_test, color='blue', label='Real')
ax3.scatter(X_test, y_pred_simple, color='red', label='Predicho')
ax3.set_xlabel(feature)
ax3.set_ylabel('MEDV')
ax3.legend()
st.pyplot(fig3)

st.header('Regresión Lineal Múltiple')
features = st.multiselect('Selecciona las variables predictoras:', [col for col in df.columns if col != 'MEDV'], default=[col for col in df.columns if col != 'MEDV'][:3])
if features:
    X_multi = df[features]
    y_multi = df['MEDV']
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)
    lr_multi = LinearRegression()
    lr_multi.fit(X_train_m, y_train_m)
    y_pred_multi = lr_multi.predict(X_test_m)
    st.write(f"MSE: {mean_squared_error(y_test_m, y_pred_multi):.2f}")
    st.write(f"R2: {r2_score(y_test_m, y_pred_multi):.2f}")
    st.write('Coeficientes:')
    st.write(dict(zip(features, lr_multi.coef_)))
