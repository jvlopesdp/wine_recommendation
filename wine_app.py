import streamlit as st
import pandas as pd
import pickle
from src.models.recommendation_system import WineRecommender


# Carregar modelo treinado
@st.cache_data
def load_model():
    with open('models//wine_recommender_knn.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# Interface Streamlit
st.title('🍇 Recomendação de Vinhos 🍷')

# Selecionar um vinho
selected_wine_idx = st.selectbox('Selecione um vinho', model.original_df['title'].unique())
wine_idx = model.original_df[model.original_df['title'] == selected_wine_idx].index[0]

# Mostrar detalhes do vinho selecionado
st.subheader('Detalhes do Vinho Selecionado')
wine_details = model.original_df.iloc[wine_idx][['title', 'country', 'province', 'variety', 'winery', 'points', 'price']]
wine_details_transposed = wine_details.transpose()
wine_details_df = pd.DataFrame(wine_details_transposed)
wine_details_df.columns = ['Ficha técnica']
st.table(wine_details_df)


# Recomendar 10 vinhos semelhantes
st.subheader('Top 10 Vinhos Recomendados')
recommendations = model.recommend(wine_idx)
recommendations['title'] = recommendations['title'].apply(lambda x: (x[:27] + '...') if len(x) > 30 else x)
st.write(recommendations.set_index('title'))

