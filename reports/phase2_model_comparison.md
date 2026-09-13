# Fase 2 - Comparacao de modelos e cenarios

Metrica de otimizacao: **F2-score** (D8). Classe positiva: evasao (`Desistente = 1`).

## 1. Divisao treino/teste (congelada)

- `RANDOM_STATE` = 42
- Teste = 30% (estratificado)
- Treino: 3096 linhas
- Teste: 1327 linhas (preservado para a Fase 3)
- Indices salvos em `train_test_split.json`

## 2. Alvo: principal

Alvo principal (D1): `Desistente = 1`; `Graduado` + `Matriculado` = 0.

### Melhores resultados (F2 na validacao cruzada)

| scenario_label | model | n_features | train_mean | cv_mean | cv_std | gap |
|---|---|---|---|---|---|---|
| Completo | logistic_balanced | 35 | 0.8115 | 0.7945 | 0.0181 | 0.0170 |
| Completo sem macro | logistic_balanced | 32 | 0.8088 | 0.7913 | 0.0191 | 0.0174 |
| Completo sem macro | random_forest_balanced | 32 | 0.8212 | 0.7867 | 0.0250 | 0.0345 |


### Hiperparametros escolhidos

| scenario_label | model | best_params |
|---|---|---|
| Early Warning | dummy_prior | {} |
| Early Warning | logistic | {"model__C": "10.0"} |
| Early Warning | logistic_balanced | {"model__C": "0.1"} |
| Early Warning | random_forest | {"model__max_depth": "None", "model__min_samples_leaf": "1"} |
| Early Warning | random_forest_balanced | {"model__max_depth": "12", "model__min_samples_leaf": "5"} |
| Early Warning | hist_gradient_boosting | {"model__learning_rate": "0.05", "model__max_iter": "100"} |
| Completo | dummy_prior | {} |
| Completo | logistic | {"model__C": "10.0"} |
| Completo | logistic_balanced | {"model__C": "1.0"} |
| Completo | random_forest | {"model__max_depth": "None", "model__min_samples_leaf": "1"} |
| Completo | random_forest_balanced | {"model__max_depth": "6", "model__min_samples_leaf": "5"} |
| Completo | hist_gradient_boosting | {"model__learning_rate": "0.05", "model__max_iter": "100"} |
| Early Warning sem financeiras | dummy_prior | {} |
| Early Warning sem financeiras | logistic | {"model__C": "10.0"} |
| Early Warning sem financeiras | logistic_balanced | {"model__C": "0.1"} |
| Early Warning sem financeiras | random_forest | {"model__max_depth": "12", "model__min_samples_leaf": "1"} |
| Early Warning sem financeiras | random_forest_balanced | {"model__max_depth": "6", "model__min_samples_leaf": "5"} |
| Early Warning sem financeiras | hist_gradient_boosting | {"model__learning_rate": "0.05", "model__max_iter": "100"} |
| Early Warning sem macro | dummy_prior | {} |
| Early Warning sem macro | logistic | {"model__C": "10.0"} |
| Early Warning sem macro | logistic_balanced | {"model__C": "0.1"} |
| Early Warning sem macro | random_forest | {"model__max_depth": "12", "model__min_samples_leaf": "1"} |
| Early Warning sem macro | random_forest_balanced | {"model__max_depth": "12", "model__min_samples_leaf": "5"} |
| Early Warning sem macro | hist_gradient_boosting | {"model__learning_rate": "0.05", "model__max_iter": "100"} |
| Completo sem macro | dummy_prior | {} |
| Completo sem macro | logistic | {"model__C": "10.0"} |
| Completo sem macro | logistic_balanced | {"model__C": "0.1"} |
| Completo sem macro | random_forest | {"model__max_depth": "None", "model__min_samples_leaf": "1"} |
| Completo sem macro | random_forest_balanced | {"model__max_depth": "6", "model__min_samples_leaf": "5"} |
| Completo sem macro | hist_gradient_boosting | {"model__learning_rate": "0.05", "model__max_iter": "200"} |


### Tabela completa (media dos 5 folds)

| scenario_label | model | gap_f2 | f2 | recall | precision | f1 | roc_auc | accuracy |
|---|---|---|---|---|---|---|---|---|
| Completo | logistic_balanced | 0.0170 | 0.7945 | 0.8048 | 0.7568 | 0.7797 | 0.9111 | 0.8540 |
| Completo sem macro | logistic_balanced | 0.0174 | 0.7913 | 0.7998 | 0.7612 | 0.7794 | 0.9116 | 0.8547 |
| Completo sem macro | random_forest_balanced | 0.0345 | 0.7867 | 0.7988 | 0.7428 | 0.7695 | 0.9072 | 0.8466 |
| Completo | random_forest_balanced | 0.0436 | 0.7800 | 0.7907 | 0.7412 | 0.7647 | 0.9073 | 0.8440 |
| Early Warning | random_forest_balanced | 0.0581 | 0.7721 | 0.7837 | 0.7296 | 0.7554 | 0.8937 | 0.8372 |
| Early Warning | logistic_balanced | 0.0110 | 0.7712 | 0.7817 | 0.7336 | 0.7563 | 0.8933 | 0.8382 |
| Early Warning sem macro | logistic_balanced | 0.0104 | 0.7697 | 0.7777 | 0.7412 | 0.7584 | 0.8930 | 0.8411 |
| Early Warning sem macro | random_forest_balanced | 0.0532 | 0.7693 | 0.7797 | 0.7315 | 0.7544 | 0.8916 | 0.8372 |
| Early Warning sem financeiras | logistic_balanced | 0.0062 | 0.7487 | 0.7626 | 0.6983 | 0.7288 | 0.8736 | 0.8178 |
| Completo | hist_gradient_boosting | 0.1485 | 0.7423 | 0.7213 | 0.8421 | 0.7764 | 0.9111 | 0.8669 |
| Completo | random_forest | 0.2581 | 0.7419 | 0.7213 | 0.8404 | 0.7756 | 0.9090 | 0.8663 |
| Early Warning sem financeiras | random_forest_balanced | 0.0193 | 0.7407 | 0.7555 | 0.6872 | 0.7195 | 0.8702 | 0.8110 |
| Completo sem macro | hist_gradient_boosting | 0.2253 | 0.7401 | 0.7203 | 0.8336 | 0.7723 | 0.9056 | 0.8640 |
| Completo | logistic | 0.0155 | 0.7388 | 0.7163 | 0.8455 | 0.7755 | 0.9105 | 0.8669 |
| Completo sem macro | logistic | 0.0132 | 0.7382 | 0.7153 | 0.8476 | 0.7756 | 0.9102 | 0.8673 |
| Completo sem macro | random_forest | 0.2631 | 0.7369 | 0.7153 | 0.8406 | 0.7722 | 0.9070 | 0.8650 |
| Early Warning | hist_gradient_boosting | 0.1358 | 0.7146 | 0.6932 | 0.8162 | 0.7494 | 0.8874 | 0.8511 |
| Early Warning sem macro | hist_gradient_boosting | 0.1267 | 0.7107 | 0.6891 | 0.8145 | 0.7460 | 0.8861 | 0.8495 |
| Early Warning | random_forest | 0.2930 | 0.7070 | 0.6871 | 0.8013 | 0.7393 | 0.8895 | 0.8446 |
| Early Warning sem macro | random_forest | 0.1607 | 0.7051 | 0.6841 | 0.8040 | 0.7391 | 0.8932 | 0.8450 |
| Early Warning sem macro | logistic | 0.0083 | 0.7042 | 0.6791 | 0.8264 | 0.7455 | 0.8911 | 0.8511 |
| Early Warning | logistic | 0.0085 | 0.7019 | 0.6761 | 0.8286 | 0.7445 | 0.8923 | 0.8511 |
| Early Warning sem financeiras | hist_gradient_boosting | 0.1561 | 0.6695 | 0.6458 | 0.7878 | 0.7090 | 0.8644 | 0.8301 |
| Early Warning sem financeiras | random_forest | 0.2021 | 0.6599 | 0.6358 | 0.7812 | 0.7003 | 0.8713 | 0.8256 |
| Early Warning sem financeiras | logistic | 0.0036 | 0.6485 | 0.6197 | 0.7985 | 0.6974 | 0.8722 | 0.8275 |
| Completo | dummy_prior | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.6789 |
| Early Warning | dummy_prior | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.6789 |
| Completo sem macro | dummy_prior | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.6789 |
| Early Warning sem financeiras | dummy_prior | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.6789 |
| Early Warning sem macro | dummy_prior | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.6789 |


## 2. Alvo: sensibilidade

Analise de sensibilidade (D1): `Desistente` vs. `Graduado`, `Matriculado` excluido.

### Hiperparametros escolhidos

_(sem resultados)_


### Tabela completa (media dos 5 folds)

_(sem resultados)_


## 3. Observacoes metodologicas

- Todo o pre-processamento (imputacao, padronizacao e one-hot) e reajustado dentro de cada fold: **sem vazamento**.
- `class_weight` foi testado apenas em Regressao Logistica e Random Forest, que oferecem suporte ao parametro.
- O desempenho do `dummy_prior` serve de referencia minima: qualquer modelo util precisa supera-lo com folga.
- O conjunto de **teste nao foi avaliado** nesta fase (D8).

## 4. Arquivos gerados

- `reports/phase2_model_comparison.md`
- `reports/tables/phase2_cv_comparison.csv`
- `reports/tables/phase2_tuned_comparison.csv`
- `reports/figures/*.png`
