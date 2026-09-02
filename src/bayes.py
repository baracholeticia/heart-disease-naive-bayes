"""aplica o Teorema de Bayes para obter P(Y=c | x) e a regra e fronteira de decisão Bayesiana"""

import numpy as np

def compute_priors(y_train: np.ndarray) -> dict:
    """P(Y=c): frequência relativa das classes no conjunto de treinamento."""
    classes, counts = np.unique(y_train, return_counts=True)
    n = len(y_train)
    return {int(c): count / n for c, count in zip(classes, counts)}


def posterior(x, feature_model, priors: dict) -> dict:
    """
    aplica o Teorema de Bayes para uma feature
    P(Y=c | x) = p(x|Y=c) * P(Y=c) / sum_k [ p(x|Y=k) * P(Y=k) ]
    """
    likelihood_fn = getattr(feature_model, "pdf", None) or feature_model.pmf

    numerators = {c: likelihood_fn(x, c) * priors[c] for c in priors}
    total = sum(numerators.values())
    return {c: num / total for c, num in numerators.items()}

def explain_example(x, feature_model, priors: dict, feature_name: str) -> str:
    """gera um exemplo numérico completo de verossimilhança x posteriori"""
    likelihood_fn = getattr(feature_model, "pdf", None) or feature_model.pmf
    post = posterior(x, feature_model, priors)

    verossimilhancas = [
        f"  p({feature_name}={x} | Y={c}) = {likelihood_fn(x, c):.5f}   (verossimilhança)"
        for c in sorted(priors)
    ]
    posteriores = [
        f"  P(Y={c} | {feature_name}={x}) = {post[c]:.5f}   "
        f"(probabilidade da classe após considerar a observação)"
        for c in sorted(priors)
    ]

    pred = max(post, key=post.get)
    decisao = f"  **Decisão Bayesiana: Y = {pred}** (maior probabilidade a posteriori)"

    linhas = [f"Exemplo numérico para {feature_name} = {x}",
              *verossimilhancas, *posteriores, decisao]
    return "\n".join(linhas)

def find_decision_boundaries(feature_model, priors: dict, x_min: float, x_max: float, resolution: int = 20000) -> list:
    """encontra P(Y=0|x) = P(Y=1|x)"""
    xs = np.linspace(x_min, x_max, resolution)
    diffs = np.array([
        feature_model.pdf(x, 1) * priors[1] - feature_model.pdf(x, 0) * priors[0]
        for x in xs
    ])
    sign_changes = np.where(np.diff(np.sign(diffs)) != 0)[0]
    boundaries = [float(xs[i]) for i in sign_changes]
    return boundaries


def decision_rule_categorical(feature_model, priors: dict) -> dict:
    """decide qual classe tem maior posteriori por categoria (não eite fronteira)"""
    rule = {}
    for a in feature_model.categories:
        post = posterior(a, feature_model, priors)
        rule[a] = max(post, key=post.get)
    return rule