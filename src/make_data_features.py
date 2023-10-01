#%%
import os
import pandas as pd
import numpy as np
from data.get_data import GetData
from data.make_dataset import CreateDataframe
from features.FeatureEngineering import Vectorizer, SentimentAnalysis, BinaryFeature, NullTreatment

#%%
#Defining variables
input_directory = '..\\data\\raw'
output_directory = '..\\data\\processed'
topn = 50

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
    
    def concatena_df(self):
        print('Criação dos parquets')
        create_data = CreateDataframe(input_directory=input_directory,output_directory=output_directory)
        create_data.create_dataframes()
    
    def retorna_dataframe(self):
        print('Criação do dataframe - cache')
        self.parquet_files = [file for file in os.listdir(self.input_directory) if file.endswith('.parquet')]
        if not self.parquet_files:
            print('Nenhum arquivo excel encontrado')
            return None
        self.most_recent_file = max(self.parquet_files, key=lambda x: os.path.getmtime(os.path.join(self.input_directory, x)))
        most_recent_directory = os.path.join(self.input_directory, self.most_recent_file)
        self.temp_df = pd.read_parquet(most_recent_directory)
        return self.temp_df

    def fill_price_nulls(self):
        filler = NullTreatment(self.temp_df)
        print('Preenchendo valores vazios na coluna "price"')
        filled_column = 'price'
        from_columns = ['winery', 'variety']
        self.temp_df = filler.fill_mean_by_columns(filled_column, from_columns)
        return self.temp_df
    
    def fill_province_nulls(self):
        filler = NullTreatment(self.temp_df)
        print('Preenchendo valores vazios na coluna "province"')
        filled_column = 'province'
        from_columns = ['winery']
        self.temp_df = filler.fill_value_by_columns(filled_column, from_columns)
        return self.temp_df
    
    def drop_nulls(self):
        dropper = NullTreatment(self.temp_df)
        print(f'Removendo nulos restantes')
        columns_names = ['province', 'price', 'variety']
        dropper.drop_null(columns_names)
        return self.temp_df
    
    def tratamento_sentimento(self):
        vader = SentimentAnalysis(self.temp_df)
        print('Realizando análise de sentimento e criando colunas do vader')
        sentiment_columns = vader.get_sentiments()
        self.temp_df = pd.concat([self.temp_df, sentiment_columns],axis=1)
        return self.temp_df
    
    def vectorizer_columns(self):
        vectorizer = Vectorizer(self.temp_df, self.topn)
        print('Realizando vetorização das palavras e adicionando no dataframe')
        df_tfidf = vectorizer.extract_main_words()
        self.temp_df = pd.concat([self.temp_df, df_tfidf], axis=1) 
        return self.temp_df
    
    def colunas_binarias(self):
        binary = BinaryFeature(self.temp_df)
        binary_column = binary.binary_value('comp_vader', 'binary_comp', 0.5)
        print('Criando coluna binária')
        self.temp_df = pd.concat([self.temp_df, binary_column], axis=1) 
        return self.temp_df
    
    def cria_parquet(self):
        print('Salvando parquet com dados e features tratadas')
        self.temp_df.to_parquet(os.path.join(self.output_directory, "base_with_features.parquet"), index=False)

    
    def run(self):
        self.extrai_arquivos()
        self.concatena_df()
        self.retorna_dataframe()
        self.fill_price_nulls()
        self.fill_province_nulls()
        self.drop_nulls()
        self.tratamento_sentimento()
        self.colunas_binarias()
        self.vectorizer_columns()
        self.cria_parquet()
        

if __name__ == '__main__':
    orquestrador = Orchestrador(input_directory=input_directory,
                                output_directory=output_directory,
                                topn=topn)
    orquestrador.run()
    
