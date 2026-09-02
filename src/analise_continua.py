"""
realiza a análise bayesiana das features contínuas "age" e "chol"
(Etapas 1 a 5 do Estudo Dirigido)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from distributions import NormalFeatureModel
from bayes import compute_priors, explain_example, find_decision_boundaries

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

def load_prepared_data():
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")

    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError(
            "data/train.csv e/ou data/test.csv não encontrados. Execute data_prep.py primeiro."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def plot_decision_boundary(feature_name: str, feature_model, priors: dict, x_min: float, x_max: float, boundaries: list):
    xs = np.linspace(x_min, x_max, 1000)
    
    # pdfs
    pdf0 = np.array([feature_model.pdf(x, 0) for x in xs])
    pdf1 = np.array([feature_model.pdf(x, 1) for x in xs])
    
    # unnormalized posteriors (p(x|Y=c)P(Y=c))
    post0 = pdf0 * priors[0]
    post1 = pdf1 * priors[1]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Plot 1: Distribuições condicionais (Verossimilhanças)
    ax1.plot(xs, pdf0, label=f'p({feature_name} | Y=0)', color="#4C72B0")
    ax1.plot(xs, pdf1, label=f'p({feature_name} | Y=1)', color="#C44E52")
    ax1.set_title(f'Distribuições Condicionais - {feature_name}')
    ax1.set_ylabel('Densidade')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Fronteiras de decisão (posteriors não normalizados)
    ax2.plot(xs, post0, label=f'p({feature_name}|Y=0)P(Y=0)', color="#4C72B0")
    ax2.plot(xs, post1, label=f'p({feature_name}|Y=1)P(Y=1)', color="#C44E52")
    
    # Sombreamento das regiões de decisão
    y_pred = np.where(post1 > post0, 1, 0)
    
    ax2.fill_between(xs, 0, np.maximum(post0, post1), where=(y_pred==0), color="#4C72B0", alpha=0.2, label='Decisão Y=0')
    ax2.fill_between(xs, 0, np.maximum(post0, post1), where=(y_pred==1), color="#C44E52", alpha=0.2, label='Decisão Y=1')
    
    # Desenhar fronteiras
    for b in boundaries:
        ax1.axvline(b, color='black', linestyle='--', alpha=0.7)
        ax2.axvline(b, color='black', linestyle='--', alpha=0.7, label=f'Fronteira x={b:.2f}')
    
    ax2.set_title(f'Regiões e Fronteiras de Decisão - {feature_name}')
    ax2.set_xlabel(feature_name)
    ax2.set_ylabel('Densidade * Priori')
    
    # Evitar labels duplicados na legenda
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys())
    ax2.grid(alpha=0.3)
    
    fig.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"grafico_{feature_name}.png")
    fig.savefig(output_path, dpi=150)
    print(f"  Gráfico salvo em: {output_path}")
    plt.close(fig)

def analyze_continuous_feature(feature_name: str, train_df: pd.DataFrame, y_train: np.ndarray, priors: dict):
    print(f"\n=======================================================")
    print(f"ANÁLISE DA CARACTERÍSTICA: {feature_name.upper()}")
    print(f"=======================================================\n")
    
    # Etapa 1 - Hipótese sobre a distribuição
    print("1 - Hipótese sobre a distribuição")
    model = NormalFeatureModel(feature_name)
    model.fit(train_df[feature_name].values, y_train)
    print(model.summary())
    
    # Escolher dois pontos de exemplo (ex: médias de cada classe)
    x_exemplo_0 = model.params[0][0]
    x_exemplo_1 = model.params[1][0]
    exemplos = [x_exemplo_0, x_exemplo_1]
    
    # Etapa 2 - Verossimilhança das observações
    print("\n2 - Verossimilhança das observações (Densidades)")
    for x in exemplos:
        p0 = model.pdf(x, 0)
        p1 = model.pdf(x, 1)
        mais_compativel = "Y=1" if p1 > p0 else "Y=0"
        print(f"  Para {feature_name} = {x:.2f}:")
        print(f"    p(x|Y=0) = {p0:.6f}")
        print(f"    p(x|Y=1) = {p1:.6f}")
        print(f"    Mais compatível com {mais_compativel}\n")

    # Etapa 3 - Razão de verossimilhanças
    print("3 - Razão de verossimilhanças")
    for x in exemplos:
        lr = model.likelihood_ratio(x)
        favorece = "Y=1" if lr > 1 else ("Y=0" if lr < 1 else "neutro")
        print(f"  Lambda({x:.2f}) = {lr:.4f} -> favorece {favorece}")
        
    # Etapa 4 - Aplicação do Teorema de Bayes
    print("\n4 - Aplicação do Teorema de Bayes")
    for x in exemplos:
        print(explain_example(x, model, priors, feature_name))
        print()

    # Etapa 5 - Regra e fronteira de decisão Bayesiana
    print("5 - Regra e fronteira de decisão Bayesiana")
    x_min = train_df[feature_name].min() - 10
    x_max = train_df[feature_name].max() + 10
    boundaries = find_decision_boundaries(model, priors, x_min, x_max)
    
    if boundaries:
        print(f"  Fronteiras de decisão encontradas em: {', '.join([f'{b:.2f}' for b in boundaries])}")
    else:
        print("  Nenhuma fronteira de decisão encontrada no intervalo de dados.")
        
    plot_decision_boundary(feature_name, model, priors, x_min, x_max, boundaries)
    
    
def main():
    train_df, test_df = load_prepared_data()
    y_train = train_df["target"].values
    priors = compute_priors(y_train)
    
    print(f"Treino: {len(train_df)} observações | Teste: {len(test_df)} observações")
    print(f"Priors: P(Y=0)={priors[0]:.3f}, P(Y=1)={priors[1]:.3f}")
    
    analyze_continuous_feature("age", train_df, y_train, priors)
    analyze_continuous_feature("chol", train_df, y_train, priors)

if __name__ == "__main__":
    main()
