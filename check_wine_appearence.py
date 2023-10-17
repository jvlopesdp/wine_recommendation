#%%
import streamlit as st
import pandas as pd
import pickle
from src.models.recommendation_system import WineRecommender


# Carregar modelo treinado
# @st.cache_data
def load_model():
    with open('src\\models\\wine_recommender_knn.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# Interface Streamlit
# st.title('Recomendação de Vinhos')
path = r'C:\Users\lci734\OneDrive - AFRY\Documents\02_gabriel\02_profissional\01_data_projects\01_wine_recommendation\data\processed\base_with_features.parquet'
df = pd.read_parquet(path)

#%%
for idx in df.index:
    print(idx)
    wine_name = df.loc[df.index == idx, 'title']
    print(wine_name)
    

#%%
print(model.recommend(6).index[0])

# Mostrar detalhes do vinho selecionado
# st.subheader('Detalhes do Vinho Selecionado')
# st.write(model.original_df.iloc[wine_idx][['title', 'country', 'province', 'variety', 'winery', 'points', 'price']])
#%%
new_df = df.copy(deep=True)
new_df = new_df.loc[:, ['title']]
# %%
dic = {
    'index':[],
    0:[],
    1:[],
    2:[],
    3:[],
    4:[],
    5:[],
    6:[],
    7:[],
    8:[],
    9:[]
       }

for index in new_df.index:
    recs = model.recommend(index)
    dic['index'].append(index)
    for i in range(0, 10):
        try:
            dic[i].append(recs.index[i])
        except:
            dic[i].append(None)
        
# %%
