"""
Implementação do classificador Naive Bayes completo combinando os 
três atributos (age, chol, cp) e avaliação no conjunto de teste.
Cobre os Tópicos 5 e 7 do Estudo Dirigido.
"""

import os
import numpy as np
import pandas as pd

from distributions import NormalFeatureModel, DiscreteFeatureModel
from bayes import compute_priors
from data_prep import CP_LABELS

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

class NaiveBayesCompleto:
    def __init__(self):
        self.priors = {}
        self.model_age = NormalFeatureModel("age")
        self.model_chol = NormalFeatureModel("chol")
        self.model_cp = DiscreteFeatureModel("cp", categories=sorted(CP_LABELS.keys()), laplace_alpha=1.0)
        self.classes = []

    def fit(self, df_train: pd.DataFrame, y_train: np.ndarray):
        """Treina os modelos univariados e calcula as priori."""
        self.classes = np.unique(y_train)
        self.priors = compute_priors(y_train)
        
        # Ajusta distribuições contínuas
        self.model_age.fit(df_train["age"].values, y_train)
        self.model_chol.fit(df_train["chol"].values, y_train)
        
        # Ajusta distribuição categórica
        self.model_cp.fit(df_train["cp"].values, y_train)

    def predict_single(self, row: pd.Series) -> int:
        """Aplica a regra de decisão Bayesiana usando a soma dos logaritmos."""
        log_posteriors = {}
        
        for c in self.classes:
            # log P(Y=c)
            log_prior = np.log(self.priors[c])
            
            # log p(xj | Y=c)
            # Soma-se um valor ínfimo (1e-9) nas contínuas para evitar log(0)
            log_age = np.log(self.model_age.pdf(row["age"], c) + 1e-9)
            log_chol = np.log(self.model_chol.pdf(row["chol"], c) + 1e-9)
            log_cp = np.log(self.model_cp.pmf(row["cp"], c))
            
            # log P(Y=c) + sum(log p(xj|Y=c))
            log_posteriors[c] = log_prior + log_age + log_chol + log_cp
            
        # Retorna a classe com a maior log-probabilidade a posteriori
        return max(log_posteriors, key=log_posteriors.get)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.array([self.predict_single(row) for _, row in df.iterrows()])


def load_prepared_data():
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError("Bases não encontradas. Execute data_prep.py primeiro.")
    return pd.read_csv(train_path), pd.read_csv(test_path)


def avaliar_modelo(y_true, y_pred):
    """Gera a Matriz de Confusão e calcula as métricas do Tópico 7"""
    
    # Cálculo das métricas da Matriz de Confusão
    VP = np.sum((y_pred == 1) & (y_true == 1))
    VN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    
    print("\n--- MATRIZ DE CONFUSÃO ---")
    print(f"          | Predito: 0 | Predito: 1 |")
    print(f"----------|------------|------------|")
    print(f" Real: 0  | VN = {VN:<5} | FP = {FP:<5} |")
    print(f" Real: 1  | FN = {FN:<5} | VP = {VP:<5} |")
    print("-------------------------------------")
    
    # Cálculo das métricas
    acuracia = (VP + VN) / (VP + VN + FP + FN) if (VP + VN + FP + FN) > 0 else 0
    precisao = VP / (VP + FP) if (VP + FP) > 0 else 0
    recall = VP / (VP + FN) if (VP + FN) > 0 else 0
    f1_score = 2 * (precisao * recall) / (precisao + recall) if (precisao + recall) > 0 else 0
    
    print("\n--- MÉTRICAS DE AVALIAÇÃO ---")
    print(f"Acurácia : {acuracia:.4f} ({(acuracia*100):.2f}%)")
    print(f"Precisão : {precisao:.4f} ({(precisao*100):.2f}%)")
    print(f"Recall   : {recall:.4f} ({(recall*100):.2f}%)")
    print(f"F1-Score : {f1_score:.4f} ({(f1_score*100):.2f}%)")

def main():
    print("Carregando os dados separados em Treino e Teste (Tópico 6)...")
    train_df, test_df = load_prepared_data()
    y_train = train_df["target"].values
    y_test = test_df["target"].values
    
    print("\nTreinando o Classificador Naive Bayes Completo (Tópico 5)...")
    nb = NaiveBayesCompleto()
    nb.fit(train_df, y_train)
    
    print("Realizando predições no conjunto de teste...")
    y_pred = nb.predict(test_df)
    
    # Imprime os resultados pedidos no Tópico 7
    avaliar_modelo(y_test, y_pred)

if __name__ == "__main__":
    main()

