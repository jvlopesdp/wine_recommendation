#%%
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import  TfidfVectorizer
import pandas as pd
import numpy as np
import os
from get_data import GetData
from make_dataset import CreateDataframe


class SentimentAnalysis:
    def __init__(self, dataframe) -> None:
        self.temp_df = dataframe.copy(deep = True)
        self.columns_corresp = {
            'pos': 'pos_vader',
            'neg': 'neg_vader',
            'neu': 'neu_vader',
            'compound': 'comp_vader'}
        pass
    def get_sentiments(self):
        sentiment_df = pd.DataFrame()
        analyzer = SentimentIntensityAnalyzer()
        for key, value in self.columns_corresp.items():
            sentiment_df[value] = self.temp_df['description'].apply(lambda x: analyzer.polarity_scores(x)[key])
        return sentiment_df
#%%

class Vectorizer:
    
    def __init__(self, dataframe, topn) -> None:
        self.temp_df = dataframe.copy(deep = True)
        self.topn = -1*topn
        pass
    
    def extract_main_words(self):
        self.temp_df['description'] = self.temp_df['description'].str.lower()
        vectorizer = TfidfVectorizer(ngram_range = (2, 3), min_df=5, 
                                stop_words='english',
                                max_df=.5)
        X = self.temp_df['description']
        X_tfidf = vectorizer.fit_transform(X)
        top_40_indices = X_tfidf.sum(axis=0).argsort()[0, self.topn:].tolist()
        feature_names = vectorizer.get_feature_names_out()
        top_40_words = [feature_names[i] for i in top_40_indices]
        X_tfidf_top_40 = pd.DataFrame(X_tfidf[:, top_40_indices[0]].toarray(), columns=top_40_words[0]) # type: ignore
        return X_tfidf_top_40

# %%
###
input_directory = '..\\..\\data\\raw'
output_directory = '..\\..\\data\\processed'
topn = 40

class Orchestrador:
    
    def __init__(self, input_directory, output_directory, topn) -> None:
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.topn = topn
        pass
    
    def extrai_arquivos(self):
        print('Extração dos arquivos')
        get_data = GetData(self.input_directory)
        get_data.process_zip_files()
    
    def cria_parquet(self):
        print('Criação dos parquets')
        create_data = CreateDataframe(input_directory=input_directory,output_directory=output_directory)
        create_data.create_dataframes()
    
    def retorna_dataframe(self):
        print('Criação do dataframe - cache')
        self.parquet_files = [file for file in os.listdir(output_directory) if file.endswith('.parquet')]
        if not self.parquet_files:
            print('Nenhum arquivo excel encontrado')
            return None
        self.most_recent_file = max(self.parquet_files, key=lambda x: os.path.getmtime(os.path.join(self.output_directory, x)))
        most_recent_directory = os.path.join(self.output_directory, self.most_recent_file)
        self.temp_df = pd.read_parquet(most_recent_directory)
        return self.temp_df
        
    def run(self):
        self.extrai_arquivos()
        self.cria_parquet()
        self.retorna_dataframe()
        return self.temp_df
#%%
classe = Orchestrador(input_directory, output_directory, topn)
# %%
df = classe.run()
#%%
new_df = df[:4]
# %%
vader = SentimentAnalysis(new_df)
novas_colunas = vader.get_sentiments()
## %%
