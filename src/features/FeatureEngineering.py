#%%
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import  TfidfVectorizer
from itertools import combinations
import pandas as pd
import numpy as np
import os


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
    
    
class BinaryFeature:
    def __init__(self, dataframe) -> None:
        self.temp_df = dataframe.copy(deep = True)
        pass
    
    def binary_value(self, source_column, new_column, intersection):
        binary_df = pd.DataFrame()
        binary_df[new_column] = self.temp_df[source_column].apply(lambda x: 1 if x >= intersection else 0)
        return binary_df
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



class NullTreatment:
    
    def __init__(self, dataframe) -> None:
        self.temp_df = dataframe.copy(deep = True)
        pass
    
    def fill_mean_by_columns(self, filled_column, from_columns):
        for n in range(len(from_columns)):
            for combination in combinations(from_columns, len(from_columns)-n):
                combination = list(combination)
                self.temp_df[filled_column] = self.temp_df.groupby(combination)[filled_column].transform(lambda grp: grp.fillna(np.mean(grp)))
        return self.temp_df
    
    def fill_value_by_columns(self, filled_column, columns_from):
        for n in range(len(columns_from)):
            for combination in combinations(columns_from, len(columns_from)-n):
                combination = list(combination)
                self.temp_df[filled_column] = self.temp_df.groupby(combination)[filled_column].transform(lambda x: x.bfill().ffill())
        return self.temp_df
    
    def drop_null(self, columns_names):
        for column in columns_names:
            self.temp_df = self.temp_df.dropna(subset=[column])
            nulls = self.temp_df[column].isna().sum()
            print(f'Restaram {nulls} nulos na coluna {column}')
        return self.temp_df
# %%
