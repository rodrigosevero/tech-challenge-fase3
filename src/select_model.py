"""Fase 3: selecao do modelo final, teste unico, over/underfitting e serializacao.

Fluxo
-----
1. Escolhe o modelo final a partir dos resultados da Fase 2, com regra explicita
   (cenario de deploy definido em D2 + criterios da secao 13 do plano).
2. Reajusta hiperparametros **somente no treino** (``GridSearchCV`` com F2).
3. Escolhe o **limiar de decisao na validacao** (probabilidades *out-of-fold*),
   nunca no teste (D8).
4. Avalia o conjunto de **teste uma unica vez**, com o limiar congelado.
5. Diagnostica over/underfitting com curvas de aprendizado e de validacao.
6. Serializa o ``Pipeline`` completo + metadados.

Uso
---
    python -m src.select_model
    python -m src.select_model --scenario full     # usa o benchmark completo
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

from . import config
from .data_loader import load_processed_data
from .evaluate import (
    best_threshold_by,
    compute_threshold_metrics,
    cross_validate_pipeline,
    ensure_output_dirs,
    make_scorers,
    plot_confusion_matrix,
    plot_learning_curve_analysis,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_threshold_tradeoff,
    plot_validation_curve_analysis,
    save_table,
    summarize_cv,
    threshold_analysis_table,
    to_markdown,
)
from .preprocessing import build_pipeline
from .train import (
    create_or_load_split,
    get_model_zoo,
    get_param_grids,
    resolve_scenario,
    split_train_holdout,
)

#: Margem (em F2) para preferir um modelo mais simples e estavel (secao 13).
SIMPLICITY_MARGIN: float = 0.01

#: Valores de ``C`` avaliados na curva de validacao da Regressao Logistica.
VALIDATION_CURVE_C: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)


def select_final_candidate(
    frame: pd.DataFrame,
    scenario_key: str,
    margin: float = SIMPLICITY_MARGIN,
) -> tuple[str, dict]:
    """Escolhe o modelo final conforme os criterios do plano (secao 13).

    Regra (definida **antes** de ver o teste, para nao haver *cherry-picking*):

    1. Restringe ao cenario de deploy (Early Warning, por D2).
    2. Descarta o baseline trivial (``dummy_prior``).
    3. Toma o melhor F2 medio na validacao cruzada.
    4. Se algum modelo ficar a menos de ``margin`` do melhor **e** tiver gap
       treino-CV menor, prefere esse (mais simples e mais estavel).
    """
    subset = frame[
        (frame["target"] == "principal")
        & (frame["stage"] == "tuned")
        & (frame["metric"] == "f2")
        & (frame["scenario"] == scenario_key)
        & (frame["model"] != "dummy_prior")
    ].sort_values("cv_mean", ascending=False)

    if subset.empty:
        raise ValueError(
            f"Sem resultados da Fase 2 para o cenario '{scenario_key}'. "
            "Rode 'python -m src.train' primeiro."
        )

    best = subset.iloc[0]
    near = subset[subset["cv_mean"] >= float(best["cv_mean"]) - margin]
    chosen = near.sort_values("gap").iloc[0]

    justification = {
        "scenario": scenario_key,
        "scenario_label": str(chosen["scenario_label"]),
        "best_by_f2": str(best["model"]),
        "best_f2": float(best["cv_mean"]),
        "chosen_model": str(chosen["model"]),
        "chosen_f2": float(chosen["cv_mean"]),
        "chosen_gap": float(chosen["gap"]),
        "margin": margin,
        "within_margin": sorted(near["model"].tolist()),
        "reason": (
            "Empate tecnico dentro da margem de "
            f"{margin:.2f} em F2; escolhido o modelo com menor gap treino-CV "
            "(mais estavel e mais interpretavel)."
            if str(chosen["model"]) != str(best["model"])
            else "Melhor F2 na validacao cruzada."
        ),
    }
    return str(chosen["model"]), justification


def tune_on_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    final_columns: list[str],
    model_name: str,
    cv,
    scorers: dict,
    n_jobs: int,
):
    """Reajusta os hiperparametros do modelo escolhido usando so o treino."""
    model = get_model_zoo()[model_name]
    grid = get_param_grids().get(model_name, {})
    pipeline = build_pipeline(final_columns, model)

    if not grid:
        return pipeline.fit(X_train, y_train), {}

    search = GridSearchCV(
        pipeline,
        grid,
        scoring=scorers["f2"],
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, dict(search.best_params_)


def tune_threshold_on_validation(
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv,
    n_jobs: int,
    thresholds: tuple[float, ...] = config.THRESHOLDS,
) -> tuple[float, pd.DataFrame, np.ndarray]:
    """Escolhe o limiar usando probabilidades *out-of-fold* do **treino**.

    Cada probabilidade vem de um modelo que **nao viu** aquela linha, o que
    simula o comportamento em dados novos sem tocar no conjunto de teste (D8).
    """
    out_of_fold = cross_val_predict(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=n_jobs,
    )[:, 1]

    table = threshold_analysis_table(y_train, out_of_fold, thresholds)
    threshold = best_threshold_by(table, metric="f2")
    return threshold, table, out_of_fold


def overfitting_diagnosis(
    train_mean: float,
    cv_mean: float,
    test_value: float,
    tolerance: float = 0.05,
) -> dict:
    """Classifica o ajuste do modelo a partir dos tres niveis de desempenho."""
    gap_train_cv = train_mean - cv_mean
    generalization = cv_mean - test_value

    if gap_train_cv > tolerance:
        diagnosis = "overfitting"
        explanation = (
            "O desempenho no treino e bem superior ao da validacao cruzada: "
            "o modelo memoriza o treino."
        )
    elif train_mean < 0.5 and cv_mean < 0.5:
        diagnosis = "underfitting"
        explanation = (
            "Treino e validacao tem desempenho baixo e proximos: o modelo e "
            "simples demais para o problema."
        )
    else:
        diagnosis = "ajuste_adequado"
        explanation = (
            "Treino e validacao ficam proximos e em nivel util: nao ha sinal "
            "relevante de overfitting nem de underfitting."
        )

    return {
        "train_mean": float(train_mean),
        "cv_mean": float(cv_mean),
        "test_value": float(test_value),
        "gap_train_cv": float(gap_train_cv),
        "gap_cv_test": float(generalization),
        "tolerance": tolerance,
        "diagnosis": diagnosis,
        "explanation": explanation,
    }


def run(scenario_key: str = "early_warning", n_jobs: int | None = None) -> dict:
    """Executa a Fase 3 completa e devolve o resumo final."""
    ensure_output_dirs()
    n_jobs = config.N_JOBS if n_jobs is None else n_jobs

    df = load_processed_data()
    train_index, test_index, split_info = create_or_load_split(df)
    cv = StratifiedKFold(
        n_splits=config.N_SPLITS,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )
    scorers = make_scorers()

    print("=" * 72)
    print("FASE 3 - Selecao do modelo, teste unico e diagnostico")
    print("=" * 72)
    print(
        f"Treino: {split_info['n_train']} linhas | "
        f"Teste: {split_info['n_test']} linhas (avaliado UMA vez)"
    )
    print(f"Cenario de deploy: {scenario_key} ({config.SCENARIOS[scenario_key]['label']})")
    print(flush=True)

    # --- 1. Selecao do modelo --------------------------------------------- #
    phase2 = pd.read_csv(config.CV_COMPARISON_CSV)
    model_name, justification = select_final_candidate(phase2, scenario_key)
    print("Modelo final:")
    print(f"  escolhido ..........: {model_name}")
    print(f"  melhor F2 da Fase 2 : {justification['best_by_f2']} "
          f"({justification['best_f2']:.4f})")
    print(f"  motivo .............: {justification['reason']}")
    print(flush=True)

    # --- 2. Ajuste no treino ---------------------------------------------- #
    X_train_full, y_train, X_test_full, y_test = split_train_holdout(
        df, config.TARGET_COLUMN, train_index, test_index
    )
    raw_columns, final_columns = resolve_scenario(df, scenario_key)
    X_train = X_train_full[raw_columns]
    X_test = X_test_full[raw_columns]

    print("Ajustando hiperparametros no treino (F2)...", flush=True)
    pipeline, best_params = tune_on_train(
        X_train, y_train, final_columns, model_name, cv, scorers, n_jobs
    )
    print(f"  melhores parametros: {best_params or '(grade vazia)'}", flush=True)

    # --- 3. Metricas de treino e validacao (sem teste) -------------------- #
    raw_cv = cross_validate_pipeline(
        pipeline, X_train, y_train, cv=cv, scorers=scorers, n_jobs=n_jobs
    )
    cv_summary = {row["metric"]: row for row in summarize_cv(raw_cv)}

    # --- 4. Limiar escolhido na validacao --------------------------------- #
    threshold, threshold_table, _ = tune_threshold_on_validation(
        pipeline, X_train, y_train, cv, n_jobs
    )
    save_table(threshold_table, config.TABLES_DIR / "phase3_threshold_validation.csv")
    print(f"Limiar escolhido na validacao: {threshold:.2f}", flush=True)

    # --- 5. AVALIACAO UNICA NO TESTE -------------------------------------- #
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    test_metrics_tuned = compute_threshold_metrics(y_test, test_proba, threshold)
    test_metrics_default = compute_threshold_metrics(y_test, test_proba, 0.5)

    train_proba = pipeline.predict_proba(X_train)[:, 1]
    train_metrics_tuned = compute_threshold_metrics(
        y_train, train_proba, threshold
    )

    print()
    print("Desempenho no teste (limiar %.2f):" % threshold)
    print(
        "  F2={f2:.4f} | Recall={recall:.4f} | Precision={precision:.4f} "
        "| F1={f1:.4f} | ROC-AUC={roc_auc:.4f} | Accuracy={accuracy:.4f}".format(
            **test_metrics_tuned
        )
    )
    print(flush=True)

    # --- 6. Diagnostico de over/underfitting ------------------------------ #
    diagnosis = overfitting_diagnosis(
        train_mean=train_metrics_tuned["f2"],
        cv_mean=cv_summary["f2"]["cv_mean"],
        test_value=test_metrics_tuned["f2"],
    )

    fresh_pipeline = build_pipeline(
        final_columns, get_model_zoo()[model_name]
    )
    if best_params:
        fresh_pipeline.set_params(**best_params)

    learning_path, learning_table = plot_learning_curve_analysis(
        fresh_pipeline,
        X_train,
        y_train,
        cv=cv,
        path=config.FIGURES_DIR / "phase3_curva_aprendizado.png",
        scoring=scorers["f2"],
        n_jobs=n_jobs,
        title=f"Curva de aprendizado - {model_name} ({scenario_key})",
    )
    save_table(learning_table, config.TABLES_DIR / "phase3_learning_curve.csv")

    validation_path = None
    validation_table = None
    if model_name.startswith("logistic"):
        validation_path, validation_table = plot_validation_curve_analysis(
            fresh_pipeline,
            X_train,
            y_train,
            param_name="model__C",
            param_range=VALIDATION_CURVE_C,
            cv=cv,
            path=config.FIGURES_DIR / "phase3_curva_validacao_C.png",
            scoring=scorers["f2"],
            n_jobs=n_jobs,
            title=f"Curva de validacao - {model_name} ({scenario_key})",
        )
        save_table(
            validation_table, config.TABLES_DIR / "phase3_validation_curve.csv"
        )

    # --- 7. Figuras do teste ---------------------------------------------- #
    test_pred = (test_proba >= threshold).astype(int)
    figures = [
        plot_confusion_matrix(
            y_test,
            test_pred,
            config.FIGURES_DIR / "phase3_matriz_confusao_teste.png",
            title=f"Teste - {model_name} (limiar {threshold:.2f})",
        ),
        plot_roc_curve(
            y_test, test_proba, config.FIGURES_DIR / "phase3_curva_roc_teste.png"
        ),
        plot_precision_recall_curve(
            y_test,
            test_proba,
            config.FIGURES_DIR / "phase3_curva_precision_recall_teste.png",
        ),
        plot_threshold_tradeoff(
            threshold_table,
            config.FIGURES_DIR / "phase3_metricas_vs_limiar.png",
            chosen_threshold=threshold,
        ),
        learning_path,
    ]
    if validation_path:
        figures.append(validation_path)

    # --- 8. Serializacao --------------------------------------------------- #
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.MODEL_PATH)

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_name": model_name,
        "scenario": scenario_key,
        "scenario_label": config.SCENARIOS[scenario_key]["label"],
        "target_column": config.TARGET_COLUMN,
        "target_definition": (
            "Desistente = 1; Graduado + Matriculado = 0 (D1)"
        ),
        "random_state": config.RANDOM_STATE,
        "test_size": config.TEST_SIZE,
        "n_splits": config.N_SPLITS,
        "best_params": {k: str(v) for k, v in best_params.items()},
        "decision_threshold": threshold,
        "input_columns": raw_columns,
        "feature_columns_after_engineering": final_columns,
        "n_features": len(final_columns),
        "selection_justification": justification,
        "train_metrics": train_metrics_tuned,
        "cv_metrics": {
            metric: {
                "mean": cv_summary[metric]["cv_mean"],
                "std": cv_summary[metric]["cv_std"],
                "train_mean": cv_summary[metric]["train_mean"],
            }
            for metric in cv_summary
        },
        "test_metrics_threshold_tuned": test_metrics_tuned,
        "test_metrics_threshold_0_5": test_metrics_default,
        "diagnosis": diagnosis,
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "split": {
            "n_train": split_info["n_train"],
            "n_test": split_info["n_test"],
        },
    }
    config.MODEL_METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- 9. Relatorio ------------------------------------------------------ #
    report_path = write_report(
        metadata=metadata,
        cv_summary=cv_summary,
        threshold_table=threshold_table,
        test_metrics_default=test_metrics_default,
        learning_table=learning_table,
        validation_table=validation_table,
    )

    print("Diagnostico de ajuste:", diagnosis["diagnosis"])
    print(f"  {diagnosis['explanation']}")
    print()
    print(f"Modelo salvo ....: {config.MODEL_PATH}")
    print(f"Metadados .......: {config.MODEL_METADATA_PATH}")
    print(f"Relatorio .......: {report_path}")
    print(f"Figuras .........: {len(figures)}")

    return metadata


def write_report(
    metadata: dict,
    cv_summary: dict,
    threshold_table: pd.DataFrame,
    test_metrics_default: dict,
    learning_table: pd.DataFrame,
    validation_table: pd.DataFrame | None,
    path: Path | None = None,
) -> Path:
    """Gera o relatorio Markdown da Fase 3."""
    path = Path(path or config.PHASE3_REPORT_MD)
    path.parent.mkdir(parents=True, exist_ok=True)

    test = metadata["test_metrics_threshold_tuned"]
    diagnosis = metadata["diagnosis"]
    justification = metadata["selection_justification"]

    lines: list[str] = []
    lines.append("# Fase 3 - Modelo final, teste unico e diagnostico de ajuste")
    lines.append("")
    lines.append(f"Gerado em: `{metadata['trained_at']}`")
    lines.append("")

    lines.append("## 1. Modelo escolhido")
    lines.append("")
    lines.append(f"- **Modelo:** `{metadata['model_name']}`")
    lines.append(
        f"- **Cenario:** {metadata['scenario_label']} "
        f"(`{metadata['scenario']}`), {metadata['n_features']} features"
    )
    lines.append(f"- **Alvo:** {metadata['target_definition']}")
    lines.append(f"- **Hiperparametros:** `{metadata['best_params']}`")
    lines.append(
        f"- **Limiar de decisao:** {metadata['decision_threshold']:.2f} "
        "(escolhido na **validacao**, nunca no teste)"
    )
    lines.append("")
    lines.append("**Justificativa da escolha (secao 13 do plano):**")
    lines.append("")
    lines.append(
        f"- Melhor F2 da Fase 2: `{justification['best_by_f2']}` "
        f"({justification['best_f2']:.4f})"
    )
    lines.append(
        f"- Dentro da margem de {justification['margin']:.2f}: "
        + ", ".join(f"`{m}`" for m in justification["within_margin"])
    )
    lines.append(f"- Escolhido: `{justification['chosen_model']}` — {justification['reason']}")
    lines.append("")

    lines.append("## 2. Desempenho nos tres niveis")
    lines.append("")
    three_levels = pd.DataFrame(
        [
            {
                "nivel": "Treino",
                "f2": metadata["train_metrics"]["f2"],
                "recall": metadata["train_metrics"]["recall"],
                "precision": metadata["train_metrics"]["precision"],
                "roc_auc": metadata["train_metrics"]["roc_auc"],
            },
            {
                "nivel": "Validacao cruzada (media)",
                "f2": cv_summary["f2"]["cv_mean"],
                "recall": cv_summary["recall"]["cv_mean"],
                "precision": cv_summary["precision"]["cv_mean"],
                "roc_auc": cv_summary["roc_auc"]["cv_mean"],
            },
            {
                "nivel": "Teste (uma vez)",
                "f2": test["f2"],
                "recall": test["recall"],
                "precision": test["precision"],
                "roc_auc": test["roc_auc"],
            },
        ]
    )
    lines.append(to_markdown(three_levels))
    lines.append("")

    lines.append("## 3. Diagnostico de overfitting / underfitting")
    lines.append("")
    lines.append(f"**Resultado: `{diagnosis['diagnosis']}`**")
    lines.append("")
    lines.append(diagnosis["explanation"])
    lines.append("")
    lines.append("| Medida | Valor |")
    lines.append("|---|---|")
    lines.append(f"| Gap treino − CV | {diagnosis['gap_train_cv']:.4f} |")
    lines.append(f"| Gap CV − teste | {diagnosis['gap_cv_test']:.4f} |")
    lines.append(f"| Tolerancia adotada | {diagnosis['tolerance']:.2f} |")
    lines.append("")
    lines.append("### Curva de aprendizado")
    lines.append("")
    lines.append(to_markdown(learning_table))
    lines.append("")
    if validation_table is not None:
        lines.append("### Curva de validacao (`model__C`)")
        lines.append("")
        lines.append(to_markdown(validation_table))
        lines.append("")

    lines.append("## 4. Metricas no teste")
    lines.append("")
    lines.append(
        f"Avaliado **uma unica vez**, com o limiar congelado em "
        f"{metadata['decision_threshold']:.2f}."
    )
    lines.append("")
    lines.append(
        to_markdown(
            pd.DataFrame(
                [
                    {"limiar": f"{metadata['decision_threshold']:.2f}", **test},
                    {"limiar": "0.50 (padrao)", **test_metrics_default},
                ]
            )
        )
    )
    lines.append("")
    lines.append("### Matriz de confusao (limiar escolhido)")
    lines.append("")
    lines.append("| | Previsto nao evasao | Previsto evasao |")
    lines.append("|---|---|---|")
    lines.append(f"| **Real nao evasao** | {test['tn']} | {test['fp']} |")
    lines.append(f"| **Real evasao** | {test['fn']} | {test['tp']} |")
    lines.append("")
    lines.append(
        f"- Evasoes nao detectadas (**falsos negativos**): **{test['fn']}**"
    )
    lines.append(f"- Alarmes falsos (**falsos positivos**): **{test['fp']}**")
    lines.append("")

    lines.append("## 5. Escolha do limiar (na validacao)")
    lines.append("")
    lines.append(to_markdown(threshold_table))
    lines.append("")

    lines.append("## 6. Artefatos")
    lines.append("")
    lines.append(f"- `{config.MODEL_PATH.relative_to(config.PROJECT_ROOT)}` (pipeline completo)")
    lines.append(
        f"- `{config.MODEL_METADATA_PATH.relative_to(config.PROJECT_ROOT)}` (metadados)"
    )
    lines.append("- `reports/figures/phase3_*.png`")
    lines.append("- `reports/tables/phase3_*.csv`")
    lines.append("")

    lines.append("## 7. Ambiente")
    lines.append("")
    for key, value in metadata["environment"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fase 3 - selecao do modelo final e avaliacao no teste."
    )
    parser.add_argument(
        "--scenario",
        default="early_warning",
        choices=list(config.SCENARIOS),
        help=(
            "Cenario do modelo final. 'early_warning' e o candidato ao deploy "
            "(D2); 'full' e o benchmark."
        ),
    )
    parser.add_argument("--jobs", type=int, default=config.N_JOBS)
    args = parser.parse_args()

    run(scenario_key=args.scenario, n_jobs=args.jobs)


if __name__ == "__main__":
    main()
