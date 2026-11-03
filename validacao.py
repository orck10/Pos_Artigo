import pandas as pd
import numpy as np

from sklearn.preprocessing import KBinsDiscretizer

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import BayesianEstimator
from pgmpy.estimators import BIC

from pgmpy.inference import VariableElimination

import networkx as nx
import matplotlib.pyplot as plt

# =========================================================
# 1. CARREGAR CSV
# =========================================================

df = pd.read_csv(
    "arquivo1.csv",
    sep=",",
    encoding="utf-8"
)

print("\n==============================")
print("DADOS ORIGINAIS")
print("==============================")

print(df.head())

print("\nColunas:")
print(df.columns)

# =========================================================
# 2. REMOVER COLUNAS NÃO UTILIZADAS
# =========================================================

if "report" in df.columns:
    df = df.drop(columns=["report"])

# =========================================================
# 3. DEFINIR VARIÁVEIS
# =========================================================

entradas = [
    "Epocas",
    "Tamanho do bacth",
    "taxa de aprendizagem",
    "Numero de workers",
    "Tamnho do Data ser de treino %",
    "Camadas da Rede"
]

saidas = [
    "Tempo de treino",
    "Acuracy do treino",
    "Melhor Loss",
    "Acuracy real"
]

colunas = entradas + saidas

# =========================================================
# 4. CONVERTER TEMPO PARA SEGUNDOS
# =========================================================

df["Tempo de treino"] = pd.to_timedelta(
    df["Tempo de treino"]
).dt.total_seconds()

# =========================================================
# 5. CONVERTER DEMAIS COLUNAS PARA NUMÉRICO
# =========================================================

for coluna in colunas:

    if coluna != "Tempo de treino":

        df[coluna] = pd.to_numeric(
            df[coluna],
            errors="coerce"
        )

# =========================================================
# 6. VALIDAR TIPOS
# =========================================================

print("\n==============================")
print("TIPOS DAS COLUNAS")
print("==============================")

print(df[colunas].dtypes)

# =========================================================
# 7. REMOVER LINHAS INVÁLIDAS
# =========================================================

print("\nQuantidade antes do dropna:")
print(len(df))

df = df.dropna(subset=colunas)

print("\nQuantidade depois do dropna:")
print(len(df))

# =========================================================
# 8. VALIDAR DATAFRAME
# =========================================================

if len(df) == 0:
    raise Exception(
        "Todos os dados ficaram inválidos."
    )

# =========================================================
# 9. DISCRETIZAÇÃO
# =========================================================

discretizador = KBinsDiscretizer(
    n_bins=3,
    encode="ordinal",
    strategy="quantile"
)

df_discreto = df.copy()

df_discreto[colunas] = discretizador.fit_transform(
    df[colunas]
)

df_discreto[colunas] = (
    df_discreto[colunas]
    .astype(int)
)

print("\n==============================")
print("DADOS DISCRETIZADOS")
print("==============================")

print(df_discreto.head())

# =========================================================
# 10. APRENDER ESTRUTURA DA REDE
# =========================================================

print("\n==============================")
print("APRENDENDO ESTRUTURA")
print("==============================")

hc = HillClimbSearch(df_discreto)

estrutura = hc.estimate(
    scoring_method=BIC(df_discreto)
)

# =========================================================
# 11. MOSTRAR ARESTAS
# =========================================================

print("\n==============================")
print("ARESTAS ENCONTRADAS")
print("==============================")

for edge in estrutura.edges():
    print(f"{edge[0]} ---> {edge[1]}")

# =========================================================
# 12. CRIAR MODELO
# =========================================================

modelo = DiscreteBayesianNetwork(
    estrutura.edges()
)

# =========================================================
# 13. TREINAR MODELO
# =========================================================

modelo.fit(
    df_discreto,
    estimator=BayesianEstimator,
    prior_type="BDeu"
)

print("\n==============================")
print("MODELO TREINADO")
print("==============================")

# =========================================================
# 14. MOSTRAR INFLUÊNCIAS NAS SAÍDAS
# =========================================================

print("\n==============================")
print("INFLUÊNCIAS NAS SAÍDAS")
print("==============================")

for saida in saidas:

    pais = modelo.get_parents(saida)

    print(f"\nResultado: {saida}")

    if len(pais) == 0:
        print("Sem influência direta.")
    else:

        print("Variáveis que influenciam:")

        for p in pais:
            print(f"{p} ---> {saida}")

# =========================================================
# 15. EXEMPLO DE INFERÊNCIA
# =========================================================

print("\n==============================")
print("INFERÊNCIA")
print("==============================")

inferencia = VariableElimination(modelo)

try:

    resultado = inferencia.query(
        variables=["Acuracy real"],
        evidence={
            "Epocas": 2,
            "Camadas da Rede": 1
        }
    )

    print(resultado)

except Exception as e:
    print("Erro na inferência:")
    print(e)

# =========================================================
# 16. DESENHAR GRAFO
# =========================================================

plt.figure(figsize=(15, 10))

G = nx.DiGraph()

G.add_edges_from(modelo.edges())

pos = nx.spring_layout(
    G,
    seed=42,
    k=2
)

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=3500,
    font_size=9,
    arrowsize=20
)

plt.title(
    "Rede Bayesiana - Relação entre Entradas e Resultados"
)

plt.show()