# Primeira coisa a se fazer: instalar a biblioteca vaderSentiment
# pip install vaderSentiment

# Importar biblioteca
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Dicionário de correspondências para criar novas colunas de sentimento
## As chaves devem permanecer com esse nome
## Os valores podem ser alterados
columns_corresp = {
    'pos': 'pos_vader',
    'neg': 'neg_vader',
    'neu': 'neu_vader',
    'compound': 'comp_vader'
}


# Função para criar colunas de sentimentos a partir do dicionário 'columns_corresp'.
# 'dataframe é o nome do DataFrame a ser trabalhado
# 'description_column' é o nome da coluna onde está a descrição
def get_sentiments(dataframe, columns_corresp, description_column):
    analyzer = SentimentIntensityAnalyzer()
    for key, value in columns_corresp.items():
        dataframe[value] = dataframe[description_column].apply(lambda x: analyzer.polarity_scores(x)[key])


# Função para criar nova coluna binária a partir de um valor de interseção
# 'dataframe é o nome do DataFrame a ser trabalhado
# 'source_column' é o nome da coluna onde estão os valores a serem analisados
# 'new_column' é o nome da coluna a ser criada, com os valores binários
# 'intersection' é o valor de interseção que irá dividir os valores binários.
def binary_value(dataframe, source_column, new_column, intersection):
    dataframe[new_column] = dataframe[source_column].apply(lambda x: 1 if x >= intersection else 0)
    


#get_sentiments(dataframe=df, columns_corresp=columns_corresp, description_column='description')

#binary_value(dataframe=df, source_column='comp_vader', new_column='binary_comp', intersection=0.5)