# Classificador Naive Bayes - Heart Disease

## Base de Dados Utilizada
A base de dados escolhida para este estudo foi a **Heart Disease**, proveniente do UCI Machine Learning Repository. O objetivo é prever a presença ou ausência de doenças cardíacas utilizando características clínicas do paciente.

- **Link para a base de dados:** [Heart Disease - UCI](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Variável alvo (Y):** `target` (0 = Saudável, 1 = Doente)
- **Atributos preditores (X):**
  - `age` (Idade - Contínua)
  - `chol` (Colesterol - Contínua)
  - `cp` (Tipo de dor no peito - Categórica)

---

## Como instalar

Certifique-se de ter o Python instalado na sua máquina (recomendado Python 3.9+). 

1. Clone o repositório para o seu ambiente local:
   ```bash
   git clone <LINK_DO_SEU_REPOSITORIO>
   cd heart-disease-naive-bayes
   ```

2. Instale as dependências necessárias utilizando o arquivo `requirements.txt`:
   ```bash
   pip install -r src/requirements.txt
   ```

---

## Como executar o estudo

Os scripts foram divididos em etapas lógicas para facilitar a compreensão de cada fase da modelagem bayesiana. Para reproduzir todos os resultados, execute os arquivos na seguinte ordem (sempre a partir da raiz do projeto):

### 1. Preparação da Base de Dados
Baixa a base da UCI, limpa dados ausentes, seleciona as features e divide em Treino e Teste (70/30). Gera os arquivos `train.csv` e `test.csv` na pasta `data/`.
```bash
python src/data_prep.py
```

### 2. Análise Univariada - Característica Categórica (`cp`)
Realiza a análise isolada do tipo de dor no peito, calcula as razões de verossimilhança e testa decisões bayesianas (Etapas 1 a 5).
```bash
python src/analise_categorica.py
```

### 3. Análise Univariada - Características Contínuas (`age` e `chol`)
Realiza a análise isolada das características contínuas. Utiliza distribuições Gaussianas, encontra as fronteiras de decisão e gera os gráficos salvos na pasta `outputs/`.
```bash
python src/analise_continua.py
```

### 4. Classificador Naive Bayes Completo e Avaliação
Combina as três características por meio da soma de log-probabilidades, faz a predição final sobre a base de Teste e exibe a **Matriz de Confusão** e métricas de desempenho (Acurácia, Precisão, Recall e F1-Score).
```bash
python src/classificador_nb.py
```

---

## Estrutura do Projeto

- `src/`: Contém todo o código-fonte desenvolvido.
  - `data_prep.py`: Download e preparação dos dados.
  - `distributions.py`: Classes que modelam as distribuições probabilísticas (Gaussiana e Discreta com Laplace).
  - `bayes.py`: Funções utilitárias sobre o Teorema de Bayes e fronteiras de decisão.
  - `analise_categorica.py` e `analise_continua.py`: Experimentos isolados.
  - `classificador_nb.py`: Classificador final e matriz de confusão.
  - `grafico_cp.py`: Geração isolada de gráfico de barras categóricas.
- `data/`: Armazena os dados gerados (treino e teste).
- `outputs/`: Armazena os gráficos gerados de fronteiras de decisão.