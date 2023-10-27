from src.models.recommendation_system import WineRecommender

class TrainSave:
    
    def __init__(self) -> None:
        self.saving_dir = 'models\\wine_recommender_knn.pkl'
    
    def fit_model(self, input_dir):
        # Função para chamar WineRecommender, rodar o fit e salvar no diretório especificado
        recommender = WineRecommender(input_dir=input_dir)
        recommender.fit()
        recommender.save_model(filename=self.saving_dir)
        