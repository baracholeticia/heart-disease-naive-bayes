"""
responsável pelas seguintes etapas:
  - hipótese de distribuição para cada característica
  - verossimilhança das observações (p(x|Y=0) vs p(x|Y=1)).
  - razão de verossimilhanças Λ(x) = p(x|Y=1) / p(x|Y=0).
"""

import numpy as np

class NormalFeatureModel:
    """ modela X|Y=c como uma distribuição normal"""

    def __init__(self, name: str):
        self.name = name
        self.params: dict[int, tuple[float, float]] = {} 

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        """estima mu e sigma de cada classe usando o treino"""
        for c in np.unique(y_train):
            x_c = x_train[y_train == c]
            mu = float(np.mean(x_c))
            sigma = float(np.std(x_c, ddof=1)) 
            self.params[int(c)] = (mu, sigma)
        return self

    def pdf(self, x: float, c: int) -> float:
        """densidade de probabilidade p(x | Y=c) sob a normal estimada."""
        mu, sigma = self.params[c]
        coef = 1.0 / (sigma * np.sqrt(2 * np.pi))
        expo = -((x - mu) ** 2) / (2 * sigma ** 2)
        return coef * np.exp(expo)

    def likelihood_ratio(self, x: float) -> float:
        """Λ(x) = p(x | Y=1) / p(x | Y=0)."""
        return self.pdf(x, 1) / self.pdf(x, 0)

    def summary(self) -> str:
        lines = [f"Distribuição hipotetizada para '{self.name}': Normal(mu, sigma^2)"]
        for c, (mu, sigma) in sorted(self.params.items()):
            lines.append(f"  Y={c}: mu={mu:.2f}, sigma={sigma:.2f}")
        return "\n".join(lines)


class DiscreteFeatureModel:
    """modela X|Y=c como uma distribuição discreta sobre k categorias"""

    def __init__(self, name: str, categories: list, laplace_alpha: float = 1.0):
        self.name = name
        self.categories = list(categories)
        self.alpha = laplace_alpha
        self.params: dict[int, dict] = {}
        self.raw_counts: dict[int, dict] = {}

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        """estima P(X=ak|Y=c) para cada classe e categoria"""
        K = len(self.categories)
        for c in np.unique(y_train):
            x_c = x_train[y_train == c]
            n_c = len(x_c)
            counts = {a: int(np.sum(x_c == a)) for a in self.categories}
            self.raw_counts[int(c)] = counts

            probs = {
                a: (counts[a] + self.alpha) / (n_c + K * self.alpha)
                for a in self.categories
            }
            self.params[int(c)] = probs
        return self

    def pmf(self, x, c: int) -> float:
        """P(X=x | Y=c) sob a distribuição discreta estimada."""
        return self.params[c][x]

    def likelihood_ratio(self, x) -> float:
        """Λ(x) = P(x | Y=1) / P(x | Y=0)."""
        return self.pmf(x, 1) / self.pmf(x, 0)

    def zero_frequency_categories(self) -> list:
        """lista categorias que tiveram contagem 0 em alguma classe no treino (antes de laplace)"""
        zeros = []
        for c, counts in self.raw_counts.items():
            for a, n in counts.items():
                if n == 0:
                    zeros.append((a, c))
        return zeros

    def summary(self) -> str:
        lines = [f"Distribuição hipotetizada para '{self.name}': "
                 f"discreta sobre {len(self.categories)} categorias "
                 f"(Laplace alpha={self.alpha})"]
        for c, probs in sorted(self.params.items()):
            probs_str = ", ".join(f"{a}={p:.3f}" for a, p in probs.items())
            lines.append(f"  Y={c}: {probs_str}")
        zeros = self.zero_frequency_categories()
        if zeros:
            lines.append(f" Categorias com contagem 0 antes da "
                          f"suavização: {zeros}")
        return "\n".join(lines)