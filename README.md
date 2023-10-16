Sistema de recomendação de vinhos
==============================
**[PT/BR]** 
>Este projeto se concentra na construção de um sistema de recomendação de vinhos usando dados extraídos de duas fontes do Kaggle. O objetivo é fornecer sugestões de vinhos com base em um rótulo de vinho específico.

**[EN/US]**
> This project focuses on building a wine recommendation system using data extracted from two Kaggle sources. The goal is to provide wine suggestions based on a specific wine label.

Contribuidores
------------
*  [Gabriel Rocha Carvalhaes](https://www.linkedin.com/in/gabriel-carvalhaes/)
*  [João Victor Lopes](https://www.linkedin.com/in/joaovictorlopesdepaula/)
  
Agradecemos por visitar nosso projeto e esperamos que ele possa ajudá-lo a encontrar um vinho perfeito para o seu gosto!
Feedbacks e sugestões são sempre bem-vindos.

------------

Fonte de Dados
------------
Os dados foram obtidos das seguintes fontes do Kaggle:
* [Wine Recommender](https://www.kaggle.com/code/sudhirnl7/wine-recommender/input)
* [Wine Reviews](https://www.kaggle.com/datasets/zynicide/wine-reviews)

Etapas do Projeto
------------
1. **Extração dos dados e criação do dataframe**:
Os dados de ambas as fontes foram extraídos e concatenados para formar um único dataframe, salvo em _raw_.

2. **Feature Engineering**:  Nesta etapa, foi realizada a análise de sentimento dos comentários dos vinhos. Além disso, utilizamos o método TFIDF (Term Frequency-Inverse Document Frequency) para identificar e selecionar as palavras mais relevantes presentes nas avaliações.

3. **Limpeza e Tratamento de Dados**:  Removemos os dados nulos e realizamos tratamentos nas linhas para garantir a integridade e a qualidade dos dados para as etapas subsequentes.

4. **Preparação da Base e Treino do Modelo**:  Utilizamos o modelo k-Nearest Neighbors (KNN) com a métrica de similaridade cosseno para treinar o sistema de recomendação (_Content-Based Filtering_)

5. **Aplicativo de Recomendação**: Aplicação realizada em _streamlit_ capaz de acessar o modelo pré-treinado e armazenar os dados em cache para melhor performance. Após a seleção de um determinado rótulo, ele retorna uma seleção dos 10 vinhos mais próximos com base no nosso aplicativo.

Como usar
==============================
Para utilizar o app, basta rodar a seguinte linha de comando na pasta principal do projeto:

> streamlit run wine_app.py

Estrutura das pastas
------------

    ├── LICENSE
    ├── README.md           <- The top-level README for developers using this project.
    │
    ├── data
    │   ├── processed          <- Base de dados final contendo tratamentos e transformações realizadas
    │   └── raw                     <- Dados originais a partir das fontes selecionadas.
    │
    ├── models                      <- Modelo treinado a partir de base de dados tratadas
    │
    ├── notebooks                 <- Arquivo contendo análises exploratórias e investigações realizadas na base de dados
    │
    ├── requirements.txt         <- Feito a partir da biblioteca pipreqs
    │
    ├── src                                 <- Códigos, classes e funções criadas para funcionamento do projeto
    │   ├── make_data_Features.py        <- Código capaz de extrair e ler os arquivos zip contendo as bases originais, realizar tratamentos e feature engineering
    │   │
    │   ├── data                                          <- Scripts com módulos e funções capaz de gerar e construir os dados
    │   │   └── get_data.py                         <- Extrai arquivos csv dentro do .zip
    │   │   └── make_dataset.py                 <- Cria o parquet contendo os dois dataframes e armazena em raw
    │   │
    │   ├── features                                     <- Scripts necessários para transformar os dados _raw_ em features para modelagem
    │   │   └── FeatureEngineering.py         <- Scripts que permitem construção de features a partir da análise de sentimento, TFIDF das principais palavras e tratamento de dados nulos e faltantes
    │   │
    │   ├── models                 <- Scripts para treinar modelo de recomendação
    │   │   ├── recommendatio_system.py         <- Modelo de recomendação com treino e função que permite realizar as predições de recomendação
    │   │
    │   └── visualization  <- Remover
    │       └── visualize.py
    │
    └── wine_app.py            <- Função que cria o dashboard em streamlit e acessa o modelo treinado salvo


:construction: Próximos passos :construction:
------------
* Utilização de dados reais
    * Crawler de dados em site com recomendações e avaliações reais
* Estruturação e armazenamento de dados em cloud e aplicações de técnicas de MLOps
* Construção de modelo de recomendação utilizando técnicas de _Colaborative-Filtering_ e _Content-Based Filtering_
    * Aplicação de técnicas de _embedding_ para melhor aproveitamento dos comentários das descrições e avaliações 
  
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
