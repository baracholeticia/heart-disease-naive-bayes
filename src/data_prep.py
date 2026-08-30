"""
prepara  a base de dados para o estudo dirigido.
- importa a base heart diseade da UCI
- seleciona as 3 featurees do estudo (age, chol e cp) e a variável alvo (num)
- ajusta num para 0=sem doença, 1=com doença (antes era categorizado em mais de uma classe com doença, a depender do tipo)
- caso exista linhas com valores faltantes, remove
- divide em treino (70%) e teste (30%)
- salva as bases preparadas em data/train.csv e data/test.csv
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

RANDOM_STATE = 42
TEST_SIZE = 0.30

# features escolhidas, age e chol são númericas e cp é categórica
FEATURE_COLS = ["age", "chol", "cp"]

# rótulos para as categorias de cp
CP_LABELS = {
    1: "typical_angina",
    2: "atypical_angina",
    3: "non_anginal_pain",
    4: "asymptomatic",
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def download_raw_data() -> pd.DataFrame:
    """baixa o dataset"""
    heart_disease = fetch_ucirepo(id=45)

    X = heart_disease.data.features
    y = heart_disease.data.targets 

    df = X.copy()
    df["num"] = y["num"] if "num" in y.columns else y.iloc[:, 0]
    return df


def clean_and_binarize(df: pd.DataFrame) -> pd.DataFrame:
    """seleciona as 3 features do estudo, binariza o alvo e trata NaN"""
    df = df[FEATURE_COLS + ["num"]].copy()

    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["chol"] = pd.to_numeric(df["chol"], errors="coerce")
    df["cp"] = pd.to_numeric(df["cp"], errors="coerce")

    n_before = len(df)
    df = df.dropna(subset=["age", "chol", "cp", "target"])
    n_removed = n_before - len(df)
    if n_removed:
        print(f"[data_prep] removidas {n_removed} linhas com valores "
              f"faltantes em age/chol/cp.")

    df["cp"] = df["cp"].astype(int)
    df["cp_label"] = df["cp"].map(CP_LABELS)

    return df.reset_index(drop=True)


def split_train_test(df: pd.DataFrame):
    """divide em treino e teste"""
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["target"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def summarize(df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame):
    print("\nHeart Disease UCI")
    print(f"Total de observações: {len(df)}")
    print("\nDistribuição do alvo (0=sem doença, 1=com doença):")
    print(df["target"].value_counts())
    print((df["target"].value_counts(normalize=True) * 100).round(1).astype(str) + "%")

    print("\nDistribuição de chest pain type:")
    print(df["cp_label"].value_counts())

    print(f"\nsplit: {int((1 - TEST_SIZE) * 100)}% treino / "
          f"{int(TEST_SIZE * 100)}% teste (random_state={RANDOM_STATE})")
    print(f"treino: {len(train_df)} observações | teste: {len(test_df)} observações")

    print("\nproporção de classes no treino:")
    print((train_df["target"].value_counts(normalize=True) * 100).round(1))
    print("\nproporção de classes no teste:")
    print((test_df["target"].value_counts(normalize=True) * 100).round(1))


def prepare_data(save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """baixa, limpa, divide e salva a base"""
    raw_df = download_raw_data()
    df = clean_and_binarize(raw_df)
    train_df, test_df = split_train_test(df)
    summarize(df, train_df, test_df)

    if save:
        os.makedirs(DATA_DIR, exist_ok=True)
        train_df.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
        test_df.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)
        print(f"\n[data_prep] arquivos salvos em {DATA_DIR}/")

    return train_df, test_df


if __name__ == "__main__":
    prepare_data()