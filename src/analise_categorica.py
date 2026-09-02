"""realiza a análise bayesiana da feature categórica "cp" (chest pain type)"""

import os
import pandas as pd

from data_prep import CP_LABELS
from distributions import DiscreteFeatureModel
from bayes import compute_priors, explain_example, decision_rule_categorical

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def load_prepared_data():
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")

    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError(
            "data/train.csv e/ou data/test.csv não encontrados. "
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def main():
    train_df, test_df = load_prepared_data()
    print(f"Treino: {len(train_df)} observações | Teste: {len(test_df)} observações")

    y_train = train_df["target"].values
    priors = compute_priors(y_train)
    print(f"Priors estimados no treino: P(Y=0)={priors[0]:.3f}, "
          f"P(Y=1)={priors[1]:.3f}")

    # etapa 1 - hipótese sobre a distribuição (P(cp = ak | y = c))
    print("\n1 - Hipótese sobre a distribuição ")
    categories = sorted(CP_LABELS.keys())
    cp_model = DiscreteFeatureModel("cp", categories, laplace_alpha=1.0)
    cp_model.fit(train_df["cp"].values, y_train)
    print("\n" + cp_model.summary())

    # etapa 2 - verossimilhança das observações (P(cp = ak | Y = 0) e P(cp = ak | Y = 1))
    print("\n2 - Verossimilhança por categoria")
    print("Comparação de P(cp=ak|Y=0) e P(cp=ak|Y=1) para cada categoria:\n")
    for cat in categories:
        p0 = cp_model.pmf(cat, 0)
        p1 = cp_model.pmf(cat, 1)
        mais_compativel = "Y=1" if p1 > p0 else "Y=0"
        print(f"  cp={cat} ({CP_LABELS[cat]}):")
        print(f"    P(cp={cat}|Y=0) = {p0:.4f}")
        print(f"    P(cp={cat}|Y=1) = {p1:.4f}")
        print(f"    mais compatível com {mais_compativel}\n")

    # etapa 3 - razão de verossimilhanças
    # Λ(x) = P(cp=x|Y=1) / P(cp=x|Y=0)
    # Λ>1 favorece Y=1, Λ<1 favorece Y=0
    print("\n3 - Razão de verossimilhanças")
    for cat in categories:
        lr = cp_model.likelihood_ratio(cat)
        favorece = "Y=1" if lr > 1 else ("Y=0" if lr < 1 else "nenhuma classe (neutro)")
        print(f"  Λ(cp={cat}, {CP_LABELS[cat]}) = {lr:.3f}  -> favorece {favorece}")

    # etapa 4 - Aplicação do Teorema de Bayes
    print("\n4 - Aplicação do Teorema de Bayes")
    for cat in categories:
        print(explain_example(cat, cp_model, priors, "cp"))
        print()

    # etapa 5: regra de decisão 
    print("\n5 - Regra de decisão")
    rule = decision_rule_categorical(cp_model, priors)
    for cat, decisao in rule.items():
        print(f"  cp={cat} ({CP_LABELS[cat]}) -> decisão Bayesiana: Y={decisao}")

    # verificação de frequência zero
    print("\nVERIFICAÇÃO - categorias com frequência zero")
    zeros = cp_model.zero_frequency_categories()
    if zeros:
        print("categorias que tiveram contagem 0 em alguma classe no treino:")
        for cat, c in zeros:
            print(f"  cp={cat} ({CP_LABELS[cat]}) nunca apareceu na classe Y={c}")
    else:
        print("nenhuma categoria teve contagem zero em nenhuma classe no treino.")


if __name__ == "__main__":
    main()