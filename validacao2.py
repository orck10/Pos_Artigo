import pandas as pd
import numpy as np

from sklearn.preprocessing import KBinsDiscretizer

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
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
# 6. REMOVER LINHAS INVÁLIDAS
# =========================================================

print("\nQuantidade antes do dropna:")
print(len(df))

df = df.dropna(subset=colunas)

print("\nQuantidade depois do dropna:")
print(len(df))

# =========================================================
# 7. VALIDAR DATAFRAME
# =========================================================

if len(df) == 0:
    raise Exception(
        "Todos os dados ficaram inválidos."
    )

# =========================================================
# 8. DISCRETIZAÇÃO
#
# Usando 2 bins:
# 0 = baixo
# 1 = alto
#
# Mais robusto para datasets pequenos
# =========================================================

discretizador = KBinsDiscretizer(
    n_bins=2,
    encode="ordinal",
    strategy="quantile",
    quantile_method="averaged_inverted_cdf"
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
# 9. MOSTRAR ESTADOS POSSÍVEIS
# =========================================================

print("\n==============================")
print("ESTADOS POSSÍVEIS")
print("==============================")

for coluna in colunas:

    valores = sorted(
        df_discreto[coluna].unique()
    )

    print(f"{coluna}: {valores}")

# =========================================================
# 10. DEFINIR ESTRUTURA CAUSAL
# =========================================================

edges = [

    # =====================================================
    # ENTRADAS -> TEMPO
    # =====================================================

    ("Epocas", "Tempo de treino"),
    ("Tamanho do bacth", "Tempo de treino"),
    ("Numero de workers", "Tempo de treino"),
    ("Camadas da Rede", "Tempo de treino"),

    # =====================================================
    # ENTRADAS -> ACCURACY TREINO
    # =====================================================

    ("Epocas", "Acuracy do treino"),
    ("taxa de aprendizagem", "Acuracy do treino"),
    ("Camadas da Rede", "Acuracy do treino"),

    # =====================================================
    # ENTRADAS -> LOSS
    # =====================================================

    ("taxa de aprendizagem", "Melhor Loss"),
    ("Tamanho do bacth", "Melhor Loss"),
    ("Camadas da Rede", "Melhor Loss"),

    # =====================================================
    # MÉTRICAS -> RESULTADO FINAL
    # =====================================================

    ("Acuracy do treino", "Acuracy real"),
    ("Melhor Loss", "Acuracy real"),

    # =====================================================
    # DATASET -> RESULTADOS
    # =====================================================

    ("Tamnho do Data ser de treino %", "Acuracy real"),
    ("Tamnho do Data ser de treino %", "Acuracy do treino")
]

# =========================================================
# 11. CRIAR MODELO
# =========================================================

modelo = DiscreteBayesianNetwork(edges)

# =========================================================
# 12. TREINAR MODELO
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
# 13. MOSTRAR RELAÇÕES
# =========================================================

print("\n==============================")
print("RELAÇÕES DEFINIDAS")
print("==============================")

for origem, destino in modelo.edges():
    print(f"{origem} ---> {destino}")

# =========================================================
# 14. MOSTRAR INFLUÊNCIAS
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

        for p in pais:
            print(f"{p} ---> {saida}")

# =========================================================
# 15. INFERÊNCIA
# =========================================================

print("\n==============================")
print("INFERÊNCIA")
print("==============================")

inferencia = VariableElimination(modelo)

# =========================================================
# OBSERVAÇÃO:
#
# Como usamos n_bins=2:
#
# 0 = baixo
# 1 = alto
#
# =========================================================

resultado = inferencia.query(
    variables=["Acuracy real"],
    evidence={

        # muitas épocas
        "Epocas": 1,

        # learning rate alta
        "taxa de aprendizagem": 1,

        # rede profunda
        "Camadas da Rede": 1
    }
)

print(resultado)

# =========================================================
# 16. DESENHAR GRAFO
# =========================================================

plt.figure(figsize=(18, 12))

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
    node_size=4000,
    font_size=9,
    arrowsize=20
)

plt.title(
    "Rede Bayesiana Causal - Influência dos Hiperparâmetros"
)

plt.show()

# =========================================================
# 17. MOSTRAR TABELAS DE PROBABILIDADE
# =========================================================

print("\n==============================")
print("TABELAS DE PROBABILIDADE")
print("==============================")

for cpd in modelo.get_cpds():

    print("\n-----------------------------------")
    print(cpd)