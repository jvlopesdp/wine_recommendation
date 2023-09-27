#%%
import os
import pandas as pd
import numpy as np
from get_data import GetData
from make_dataset import CreateDataframe
from FeatureEngineering import Vectorizer, SentimentAnalysis
#%%
#Defining variables
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

    def tratamento_sentimento(self):
        vader = SentimentAnalysis(self.temp_df)
        print('Realizando análise de sentimento e criando colunas do vader')
        sentiment_columns = vader.get_sentiments()
        self.temp_df = pd.concat([self.temp_df, sentiment_columns])
        return self.temp_df
    
    def vectorizer_columns(self):
        vectorizer = Vectorizer(self.temp_df, self.topn)
        print('Realizando vetorização das palavras e adicionando no dataframe')
        df_tfidf = vectorizer.extract_main_words()
        self.temp_df = pd.concat([self.temp_df, df_tfidf])
        return self.temp_df

    
    def run(self):
        self.extrai_arquivos()
        self.cria_parquet()
        self.retorna_dataframe()
        self.tratamento_sentimento()
        self.vectorizer_columns()
        return self.temp_df

if __name__ == 'main':
    orquestrador = Orchestrador(input_directory=input_directory,
                                output_directory=output_directory,
                                topn=topn)
    df = orquestrador.run()
    print(df)


# %%
orquestrador = Orchestrador(input_directory=input_directory,
                            output_directory=output_directory,
                            topn=topn)
df = orquestrador.run()
print(df)
# %%
columns_corresp = {
            'pos': 'pos_vader',
            'neg': 'neg_vader',
            'neu': 'neu_vader',
            'compound': 'comp_vader'}
# %%
list(columns_corresp.keys())
# %%
teste = pd.DataFrame()
# %%
