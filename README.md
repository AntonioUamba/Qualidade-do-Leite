# Otimização da Qualidade do Leite — Uma Abordagem CRISP-DM

> Predição dos teores de **gordura** e **proteína** do leite bovino com modelos de aprendizado de máquina, seguindo a metodologia **CRISP-DM**.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#) [![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](#) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#)

---

## 📌 Visão Geral

A qualidade do leite impacta diretamente o valor nutricional e econômico do produto. Este projeto aplica **modelos de regressão Linear** para estimar os componentes **DailyFat_P** (gordura diária) e **DailyProtein_P** (proteína diária) a partir de variáveis zootécnicas e de produção.  
O processo segue o **CRISP-DM**, da compreensão do negócio até a implantação experimental.

---

## 🧭 Objetivos

- **Analisar** fatores que influenciam gordura e proteína no leite.
- **Construir** modelos preditivos interpretáveis (Regressão Linear).
- **Apoiar decisões** de manejo do rebanho e formulação de dieta.
- **Propor** recomendações práticas para otimização da produção.

---

## 🗂️ Dataset

- Arquivo principal: `milk_sessions.csv`
- Período observado: janeiro a abril (conforme base).  
- Divisão: **treino** e **teste** (proporções típicas 80/20 ou próximas).

### 📑 Dicionário (amostra de variáveis)

| Variável            | Descrição                                      | Tipo        | Observações |
|---------------------|-------------------------------------------------|-------------|-------------|
| `Date`              | Data da coleta                                  | data        | Extraídos `Month`, `DayOfWeek` |
| `CowID`             | Identificador do animal                         | categ./int  | Vários animais |
| `DailyYield_KG`     | Produção diária de leite (kg)                   | numérico    | Distribuição assimétrica típica |
| `DailyFat_P`        | Percentual de gordura diário                    | numérico    | **Alvo 1** |
| `DailyProtein_P`    | Percentual de proteína diário                   | numérico    | **Alvo 2** |
| `LactationNumber`   | Número de lactações                             | inteiro     | Relaciona-se a variação de composição |
| `DIM`               | Dias em lactação                                | inteiro     | Faixas usadas em features derivadas |
| `Fat`, `Protein`    | Gordura/Proteína brutas                         | numérico    | Evitar vazamento quando alvo é % |
| `Lactose`           | Lactose                                         | numérico    | Opcional |
| `Mufa`, `Pufa`, `Sfa`, `Ufa` | Ácidos graxos                         | numérico    | Correlações relevantes com gordura |
| `Disease`           | Indicador de doença                             | booleano    | Convertido para 0/1 |

> **Pré-processamento aplicado**: conversão de datas, criação de features temporais, remoção de colunas com alta nulidade/baixa variância, conversão de booleanos para inteiros, **normalização (StandardScaler)** e split treino/teste.

---

## 🔬 Metodologia (CRISP-DM)

1. **Entendimento do Negócio** — maximizar qualidade (gordura/proteína) e rentabilidade do leite.  
2. **Entendimento dos Dados** — EDA com estatísticas, distribuições e **correlação** entre variáveis.  
3. **Preparação dos Dados** — limpeza, engenharia de atributos e normalização.  
4. **Modelagem** — regressões independentes para `DailyFat_P` e `DailyProtein_P`.  
5. **Avaliação** — métricas: **MSE** e **R²** em conjunto de teste.  
6. **Implantação (piloto)** — salvamento de modelos (`joblib`) e uso em pipeline de inferência.

---

## 📈 Principais Achados (EDA)

- **Gordura diária** apresenta **maior variabilidade** que proteína.  
- **Correlação** fraca e **negativa** entre gordura e proteína (trade-off).  
- **Ácidos graxos** (p.ex., MUFA/PUFA/SFA) correlacionam-se com a gordura.  
- **LactationNumber** e **DIM** impactam composição.  
- Heatmap de correlação destaca grupos de variáveis relevantes.

---

## 🤖 Modelagem

- **Alvos**: `DailyFat_P` e `DailyProtein_P`  
- **Modelo base**: **Regressão Linear** (pela interpretabilidade)  
- **Features**: todas as colunas pós-preparo, **exceto** as variáveis-alvo e colunas que causariam **vazamento** (ex.: `Fat`, `Protein` brutas ao prever `%`).  
- **Normalização**: `StandardScaler`  
- **Persistência**: modelos e scaler salvos via `joblib`

### 📏 Métricas (teste)

| Alvo               | MSE (↓) | R² (↑) | Observações |
|--------------------|---------|--------|-------------|
| `DailyFat_P`       | baixo   | bom    | Explica fração substantiva da variância |
| `DailyProtein_P`   | baixo   | moderado| Variância menor que gordura |

> *Leitura geral*: modelos úteis para **apoio à decisão** em manejo e nutrição; há espaço para evolução com features ambientais/ sazonais.

---


---
## 📈 Comclusão 

Os modelos preditivos desenvolvidos apresentaram
bom desempenho e fornecem uma base sólida para a tomada de decisões visando a
otimização da produção leiteira.
A compreensão dos fatores que influenciam os níveis de gordura e proteína no leite,
juntamente com a capacidade de prever esses componentes, representa um avanço
significativo para a indústria de laticínios. As recomendações propostas, baseadas nos
insights obtidos, têm o potencial de contribuir para o aumento da rentabilidade dos
produtores e para a melhoria da qualidade do leite fornecido aos consumidores.
---

## ✅ Recomendações de Manejo

- **Estratificar o rebanho** por número de lactações e **ajustar dieta** por estágio de lactação.  
- **Monitoramento contínuo** de componentes do leite para ajustes em tempo real.  
- **Expandir variáveis** ambientais/estacionais (clima, estação, dieta) na próxima iteração do modelo.

---


## 📚 Referências & Ferramentas

- CRISP-DM — *Cross-Industry Standard Process for Data Mining*  
- **scikit-learn** (modelos e métricas)  
- **pandas** (manipulação de dados)  
- **matplotlib** / **seaborn** (visualização)
- dataset do leite (https://github.com/nitaytech/ActiveLearningForRegression/tree/main/data)

---

## 🔗 Link no Github do Projeto
'''
https://github.com/AntonioUamba/Qualidade-do-Leite.git
```



