# Fase 3 - Modelo final, teste unico e diagnostico de ajuste

Gerado em: `2026-09-13T13:46:28+00:00`

## 1. Modelo escolhido

- **Modelo:** `logistic_balanced`
- **Cenario:** Early Warning (`early_warning`), 24 features
- **Alvo:** Desistente = 1; Graduado + Matriculado = 0 (D1)
- **Hiperparametros:** `{'model__C': '0.1'}`
- **Limiar de decisao:** 0.40 (escolhido na **validacao**, nunca no teste)

**Justificativa da escolha (secao 13 do plano):**

- Melhor F2 da Fase 2: `random_forest_balanced` (0.7721)
- Dentro da margem de 0.01: `logistic_balanced`, `random_forest_balanced`
- Escolhido: `logistic_balanced` — Empate tecnico dentro da margem de 0.01 em F2; escolhido o modelo com menor gap treino-CV (mais estavel e mais interpretavel).

## 2. Desempenho nos tres niveis

| nivel | f2 | recall | precision | roc_auc |
|---|---|---|---|---|
| Treino | 0.8070 | 0.8481 | 0.6760 | 0.9013 |
| Validacao cruzada (media) | 0.7712 | 0.7817 | 0.7336 | 0.8933 |
| Teste (uma vez) | 0.8136 | 0.8568 | 0.6772 | 0.9071 |


## 3. Diagnostico de overfitting / underfitting

**Resultado: `ajuste_adequado`**

Treino e validacao ficam proximos e em nivel util: nao ha sinal relevante de overfitting nem de underfitting.

| Medida | Valor |
|---|---|
| Gap treino − CV | 0.0358 |
| Gap CV − teste | -0.0424 |
| Tolerancia adotada | 0.05 |

### Curva de aprendizado

| train_size | train_mean | train_std | cv_mean | cv_std | gap |
|---|---|---|---|---|---|
| 371.0000 | 0.7573 | 0.0268 | 0.7652 | 0.0204 | -0.0079 |
| 742.0000 | 0.7914 | 0.0135 | 0.7780 | 0.0311 | 0.0134 |
| 1114.0000 | 0.7859 | 0.0162 | 0.7762 | 0.0258 | 0.0096 |
| 1485.0000 | 0.7806 | 0.0096 | 0.7681 | 0.0276 | 0.0124 |
| 1980.0000 | 0.7817 | 0.0088 | 0.7731 | 0.0234 | 0.0087 |
| 2476.0000 | 0.7819 | 0.0043 | 0.7712 | 0.0200 | 0.0107 |


### Curva de validacao (`model__C`)

| param_value | train_mean | cv_mean | cv_std | gap |
|---|---|---|---|---|
| 0.001 | 0.7119 | 0.7111 | 0.0206 | 0.0008 |
| 0.01 | 0.7598 | 0.7548 | 0.0179 | 0.0050 |
| 0.1 | 0.7822 | 0.7712 | 0.0200 | 0.0110 |
| 1.0 | 0.7811 | 0.7691 | 0.0191 | 0.0121 |
| 10.0 | 0.7816 | 0.7670 | 0.0192 | 0.0146 |
| 100.0 | 0.7820 | 0.7661 | 0.0199 | 0.0159 |


## 4. Metricas no teste

Avaliado **uma unica vez**, com o limiar congelado em 0.40.

| limiar | threshold | accuracy | precision | recall | f1 | f2 | roc_auc | tn | fp | fn | tp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.40 | 0.4000 | 0.8229 | 0.6772 | 0.8568 | 0.7565 | 0.8136 | 0.9071 | 727 | 174 | 61 | 365 |
| 0.50 (padrao) | 0.5000 | 0.8515 | 0.7473 | 0.8122 | 0.7784 | 0.7983 | 0.9071 | 784 | 117 | 80 | 346 |


### Matriz de confusao (limiar escolhido)

| | Previsto nao evasao | Previsto evasao |
|---|---|---|
| **Real nao evasao** | 727 | 174 |
| **Real evasao** | 61 | 365 |

- Evasoes nao detectadas (**falsos negativos**): **61**
- Alarmes falsos (**falsos positivos**): **174**

## 5. Escolha do limiar (na validacao)

| threshold | accuracy | precision | recall | f1 | f2 | roc_auc | tn | fp | fn | tp |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.2000 | 0.6567 | 0.4820 | 0.9276 | 0.6343 | 0.7828 | 0.8929 | 1111.0000 | 991.0000 | 72.0000 | 922.0000 |
| 0.2500 | 0.7167 | 0.5349 | 0.9024 | 0.6717 | 0.7934 | 0.8929 | 1322.0000 | 780.0000 | 97.0000 | 897.0000 |
| 0.3000 | 0.7626 | 0.5867 | 0.8813 | 0.7045 | 0.8009 | 0.8929 | 1485.0000 | 617.0000 | 118.0000 | 876.0000 |
| 0.3500 | 0.7907 | 0.6266 | 0.8612 | 0.7254 | 0.8012 | 0.8929 | 1592.0000 | 510.0000 | 138.0000 | 856.0000 |
| 0.4000 | 0.8159 | 0.6693 | 0.8431 | 0.7462 | 0.8015 | 0.8929 | 1688.0000 | 414.0000 | 156.0000 | 838.0000 |
| 0.4500 | 0.8298 | 0.7036 | 0.8119 | 0.7539 | 0.7876 | 0.8929 | 1762.0000 | 340.0000 | 187.0000 | 807.0000 |
| 0.5000 | 0.8382 | 0.7323 | 0.7817 | 0.7562 | 0.7713 | 0.8929 | 1818.0000 | 284.0000 | 217.0000 | 777.0000 |
| 0.5500 | 0.8472 | 0.7688 | 0.7495 | 0.7590 | 0.7533 | 0.8929 | 1878.0000 | 224.0000 | 249.0000 | 745.0000 |
| 0.6000 | 0.8482 | 0.7950 | 0.7103 | 0.7503 | 0.7257 | 0.8929 | 1920.0000 | 182.0000 | 288.0000 | 706.0000 |


## 6. Artefatos

- `models/model.joblib` (pipeline completo)
- `models/model_metadata.json` (metadados)
- `reports/figures/phase3_*.png`
- `reports/tables/phase3_*.csv`

## 7. Ambiente

- python: `3.12.14`
- scikit_learn: `1.5.2`
- pandas: `2.2.3`
- numpy: `2.1.3`
