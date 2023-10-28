#%%
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors
import pickle
import os
#%%
input_dir = 'data\\processed\\base_with_features.parquet'
#%%
class WineRecommender:
    
    def __init__(self, input_dir):
        self.pipeline = None
        self.df = None
        self.knn_model = None
        self.input_dir = input_dir

    def fit(self):
        self.original_df = pd.read_parquet(self.input_dir)
        self.df = self.original_df.copy(deep=True)

        # Colunas para tratamento
        categorical_features = ["province", "variety","winery"]
        numeric_features = self.df.drop(columns = ['country', 'description', 'designation','province',
       'region_1', 'region_2', 'taster_name', 'taster_twitter_handle', 'title',
       'variety', 'winery', 'pos_vader', 'neg_vader', 'neu_vader',
       'comp_vader']).columns.tolist()
        binary_features = [
                    col for col in numeric_features
                    if all(pd.Series(self.df[col].unique()).isin([0, 1]))
                ]    
        numeric_features = [col for col in numeric_features if col not in binary_features]
        
        #Filtrando o dataframe
        all_selected_columns = numeric_features + binary_features + categorical_features
        
        self.df = self.df[all_selected_columns].copy()



        # Construindo o pipeline
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.pipeline = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features),
                ('binary', 'passthrough', binary_features)
            ])

        # Aplicando o pipeline ao dataframe
        df_transformed = self.pipeline.fit_transform(self.df)

        # Treinando o modelo k-NN
        self.knn_model = NearestNeighbors(n_neighbors=51, algorithm='auto', metric='cosine').fit(df_transformed)

    def recommend(self, idx):
        single_row_df = self.df.iloc[[idx]]# type: ignore
        transformed_row = self.pipeline.transform(single_row_df) # type: ignore
        distances, indices = self.knn_model.kneighbors(transformed_row)# type: ignore
        columns_to_return = ['title', 'country', 'province', 'variety', 'winery', 'points', 'price']
        recommendations = self.original_df.iloc[indices[0][0:51]][columns_to_return].drop_duplicates(subset = ['title'])
        recommendations = recommendations[recommendations['title'].notna()]
        return recommendations[1:11]


    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

#%%
