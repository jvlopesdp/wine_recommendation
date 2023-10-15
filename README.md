Sistema de recomendação de vinhos
==============================
**[PT/BR]** 
>Este projeto se concentra na construção de um sistema de recomendação de vinhos usando dados extraídos de duas fontes do Kaggle. O objetivo é fornecer sugestões de vinhos com base em um rótulo de vinho específico.

[EN/US]
> Traduzir texto

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
    ├── Makefile           <- Remover
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Remover
    │   ├── interim        <- Remover
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- Remover
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Remover
    │
    ├── reports            <- Remover
    │   └── figures        <- Remover
    │
    ├── requirements.txt   <- Feito com pipreqs
    │
    ├── setup.py           <- Remover
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Remover
    │       └── visualize.py
    │
    └── tox.ini            <- Remover


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
