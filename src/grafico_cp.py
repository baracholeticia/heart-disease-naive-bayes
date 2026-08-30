"""
gera gráfico de barras para cp (chest pain type) mostrando P(cp=ak|Y=c) para cada categoria em outputs/grafico_cp.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_prep import CP_LABELS
from distributions import DiscreteFeatureModel

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def load_prepared_data():
    train_path = os.path.join(DATA_DIR, "train.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            "data/train.csv não encontrado"
        )
    return pd.read_csv(train_path)


def main():
    train_df = load_prepared_data()
    y_train = train_df["target"].values

    categories = sorted(CP_LABELS.keys())
    cp_model = DiscreteFeatureModel("cp", categories, laplace_alpha=1.0)
    cp_model.fit(train_df["cp"].values, y_train)

    probs_y0 = [cp_model.pmf(cat, 0) for cat in categories]
    probs_y1 = [cp_model.pmf(cat, 1) for cat in categories]
    labels = [CP_LABELS[cat] for cat in categories]

    x = np.arange(len(categories))
    width = 0.35                    

    fig, ax = plt.subplots(figsize=(9, 6))
    bars0 = ax.bar(x - width / 2, probs_y0, width, label="Y=0 (sem doença)",
                    color="#4C72B0")
    bars1 = ax.bar(x + width / 2, probs_y1, width, label="Y=1 (com doença)",
                    color="#C44E52")

    ax.set_ylabel("P(cp = categoria | Y = c)")
    ax.set_xlabel("Tipo de dor no peito (cp)")
    ax.set_title("Distribuição de 'chest pain type' (cp) por classe\n"
                  "(probabilidades estimadas no conjunto de treino)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bars in (bars0, bars1):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    fig.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "grafico_cp.png")
    fig.savefig(output_path, dpi=150)
    print(f"Gráfico salvo em: {output_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()