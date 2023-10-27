import streamlit as st
import pandas as pd
import pickle
from src.models.recommendation_system import WineRecommender
from src.models.train_model import TrainSave
import os

def train_model():
    train_save = TrainSave()
    train_save.fit_model(input_dir='data\\processed\\base_with_features.parquet')
    # train_save.save_model(model=model, saving_dir='models\\wine_recommender_knn.pkl')

# Carregar modelo treinado
@st.cache_data
def load_model():
    with open('models//wine_recommender_knn.pkl', 'rb') as f:
        return pickle.load(f)

# Função para verificar se existe um modelo pré-treinado no diretório
# Caso positvo, o App utilizará o modelo pré-treinado
# Caso negativo, um novo modelo será treinado
def check_existence(file_to_check, in_dir):
    if not file_to_check in os.listdir(in_dir):
        print('Treinando novo modelo ...')
        train_model()
        print('Concluído')
    else:
        print('Utilizando modelo existente ...')
        
check_existence('wine_recommender_knn.pkl', 'models')
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

