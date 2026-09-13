"""Fase 2: baseline, modelos, validacao cruzada e comparacao de cenarios.

O que este modulo faz
---------------------
1. Carrega a base processada (Fase 1) e reconstroi a divisao treino/teste
   **estratificada**, congelando os indices em disco.
2. Compara 6 modelos (incluindo ``DummyClassifier`` como baseline) em 5
   cenarios de features, via ``StratifiedKFold(5)``.
3. Ajusta hiperparametros com ``GridSearchCV`` otimizando **F2-score** (D8),
   sempre **apenas no treino**.
4. Repete a comparacao para o alvo de sensibilidade (D1).
5. Gera tabelas, graficos e um relatorio em Markdown.

O conjunto de **teste nunca e avaliado aqui** -- ele fica congelado para a
Fase 3, conforme a decisao D8.

Uso
---
    python -m src.train
    python -m src.train --no-tuning          # apenas comparacao com padroes
    python -m src.train --targets main        # ignora o alvo de sensibilidade
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)

from . import config
from .data_loader import load_processed_data
from .evaluate import (
    build_table,
    cross_validate_pipeline,
    ensure_output_dirs,
    make_scorers,
    plot_metric_by_model,
    plot_scenario_heatmap,
    plot_train_vs_cv_gap,
    save_table,
    summarize_cv,
    to_markdown,
)
from .features import engineered_feature_columns, resolve_scenario_columns
from .preprocessing import build_pipeline

# Aviso benigno: o scipy mais novo removeu a opcao de solver 'iprint' que o
# scikit-learn 1.5.x ainda passa. Nao afeta resultados, mas poluiria o log.
warnings.filterwarnings("ignore", message="Unknown solver options: iprint")

# --------------------------------------------------------------------------- #
# Modelos candidatos (D-requisitos: baseline + 3 familias + variantes de peso)
# --------------------------------------------------------------------------- #


def get_model_zoo() -> "OrderedDict[str, object]":
    """Modelos candidatos.

    ``class_weight`` e aplicado **apenas** nos modelos que oferecem suporte
    (Regressao Logistica e Random Forest), conforme a decisao do autor.
    ``DummyClassifier`` e ``HistGradientBoostingClassifier`` nao possuem esse
    parametro e por isso aparecem em uma unica versao.
    """
    random_state = config.RANDOM_STATE
    return OrderedDict(
        [
            (
                "dummy_prior",
                DummyClassifier(strategy="prior", random_state=random_state),
            ),
            (
                "logistic",
                LogisticRegression(max_iter=2000, random_state=random_state),
            ),
            (
                "logistic_balanced",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=200, random_state=random_state, n_jobs=1
                ),
            ),
            (
                "random_forest_balanced",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
            (
                "hist_gradient_boosting",
                HistGradientBoostingClassifier(random_state=random_state),
            ),
        ]
    )


def get_param_grids() -> dict[str, dict]:
    """Grades pequenas e explicitas (didaticas para o video)."""
    return {
        "dummy_prior": {},
        "logistic": {"model__C": [0.01, 0.1, 1.0, 10.0]},
        "logistic_balanced": {"model__C": [0.01, 0.1, 1.0, 10.0]},
        "random_forest": {
            "model__max_depth": [None, 6, 12],
            "model__min_samples_leaf": [1, 5],
        },
        "random_forest_balanced": {
            "model__max_depth": [None, 6, 12],
            "model__min_samples_leaf": [1, 5],
        },
        "hist_gradient_boosting": {
            "model__learning_rate": [0.05, 0.1],
            "model__max_iter": [100, 200],
        },
    }


# --------------------------------------------------------------------------- #
# Divisao treino/teste (congelada)
# --------------------------------------------------------------------------- #


def create_or_load_split(
    df: pd.DataFrame,
    stratify_column: str = config.TARGET_COLUMN,
    path: Path | None = None,
) -> tuple[list[int], list[int], dict]:
    """Cria (ou reutiliza) a divisao estratificada 70/30.

    Os indices posicionais sao salvos em ``data/processed/train_test_split.json``
    para que a Fase 3 use **exatamente** o mesmo teste.
    """
    path = Path(path or config.SPLIT_INDICES_PATH)

    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        consistent = (
            payload.get("n_rows") == int(df.shape[0])
            and payload.get("random_state") == config.RANDOM_STATE
            and payload.get("stratify_column") == stratify_column
            and payload.get("test_size") == config.TEST_SIZE
        )
        if consistent:
            return (
                list(payload["train_index"]),
                list(payload["test_index"]),
                {**payload, "reused": True},
            )

    positions = np.arange(int(df.shape[0]))
    train_index, test_index = train_test_split(
        positions,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df[stratify_column].to_numpy(),
    )

    payload = {
        "n_rows": int(df.shape[0]),
        "test_size": config.TEST_SIZE,
        "random_state": config.RANDOM_STATE,
        "stratify_column": stratify_column,
        "n_train": int(len(train_index)),
        "n_test": int(len(test_index)),
        "train_index": sorted(int(i) for i in train_index),
        "test_index": sorted(int(i) for i in test_index),
        "reused": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload["train_index"], payload["test_index"], payload


def split_train_holdout(
    df: pd.DataFrame,
    target_column: str,
    train_index: list[int],
    test_index: list[int],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Materializa treino/teste para um alvo, descartando linhas sem rotulo.

    No alvo de sensibilidade as linhas ``Matriculado`` ficam ausentes; elas sao
    removidas de ambos os lados, preservando a divisao.
    """
    target = pd.to_numeric(df[target_column], errors="coerce")
    valid = target.notna()

    train_rows = [i for i in train_index if bool(valid.iloc[i])]
    test_rows = [i for i in test_index if bool(valid.iloc[i])]

    return (
        df.iloc[train_rows],
        target.iloc[train_rows],
        df.iloc[test_rows],
        target.iloc[test_rows],
    )


# --------------------------------------------------------------------------- #
# Cenario: colunas e avaliacao
# --------------------------------------------------------------------------- #


def resolve_scenario(df: pd.DataFrame, scenario_key: str) -> tuple[list[str], list[str]]:
    """Descobre as colunas de entrada e as colunas finais de um cenario.

    Returns
    -------
    tuple[list[str], list[str]]
        ``(colunas_de_entrada, colunas_apos_engineering)``. As colunas de entrada
        sao apenas as observadas no dados (as derivadas sao criadas pela pipeline).
    """
    scenario = config.SCENARIOS[scenario_key]
    engineered = set(config.ENGINEERED_FEATURES)

    raw_columns = [
        column for column in scenario["columns"] if column not in engineered
    ]
    raw_columns = [column for column in raw_columns if column in df.columns]

    available_after = engineered_feature_columns(raw_columns)
    final_columns = resolve_scenario_columns(
        available_after, scenario["columns"]
    )
    return raw_columns, final_columns


def evaluate_models_on_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    final_columns: list[str],
    cv,
    scorers: dict,
    scenario_key: str,
    target_name: str,
    stage: str,
    tune: bool,
    n_jobs: int | None = None,
    verbose: bool = True,
) -> list[dict]:
    """Roda CV (e opcionalmente GridSearch) para todos os modelos do cenario."""
    rows: list[dict] = []
    n_jobs = config.N_JOBS if n_jobs is None else n_jobs
    scenario_label = config.SCENARIOS[scenario_key]["label"]
    model_zoo = get_model_zoo()
    param_grids = get_param_grids()

    for model_name, model in model_zoo.items():
        started = time.perf_counter()

        if tune and param_grids.get(model_name):
            pipeline = build_pipeline(final_columns, model)
            search = GridSearchCV(
                pipeline,
                param_grids[model_name],
                scoring=scorers["f2"],
                cv=cv,
                n_jobs=n_jobs,
                refit=True,
                error_score="raise",
            )
            search.fit(X_train, y_train)
            estimator = search.best_estimator_
            best_params = json.dumps(
                {k: str(v) for k, v in search.best_params_.items()},
                ensure_ascii=False,
            )
        else:
            estimator = build_pipeline(final_columns, model)
            best_params = "{}"

        raw = cross_validate_pipeline(
            estimator, X_train, y_train, cv=cv, scorers=scorers, n_jobs=n_jobs
        )

        for summary in summarize_cv(raw):
            rows.append(
                {
                    "scenario": scenario_key,
                    "scenario_label": scenario_label,
                    "target": target_name,
                    "stage": stage,
                    "model": model_name,
                    "n_features": len(final_columns),
                    "best_params": best_params,
                    **summary,
                }
            )

        if verbose:
            f2_means = [
                row["cv_mean"] for row in summarize_cv(raw) if row["metric"] == "f2"
            ]
            best_f2 = f2_means[0] if f2_means else float("nan")
            print(
                f"    {model_name:<26} F2(cv)={best_f2:.4f}"
                f"  [{time.perf_counter() - started:.1f}s]",
                flush=True,
            )

    return rows


# --------------------------------------------------------------------------- #
# Relatorio
# --------------------------------------------------------------------------- #


def _key_table(frame: pd.DataFrame, target: str, stage: str) -> pd.DataFrame:
    """Tabela resumo (cenarios x modelos) com as metricas principais."""
    metrics = ["f2", "recall", "precision", "f1", "roc_auc", "accuracy"]
    subset = frame[(frame["target"] == target) & (frame["stage"] == stage)]
    if subset.empty:
        return pd.DataFrame()

    pivot = (
        subset[subset["metric"].isin(metrics)]
        .pivot_table(
            index=["scenario_label", "model"],
            columns="metric",
            values="cv_mean",
        )
        .reindex(columns=metrics)
        .reset_index()
    )

    gaps = (
        subset[subset["metric"] == "f2"]
        .set_index(["scenario_label", "model"])["gap"]
        .rename("gap_f2")
        .reset_index()
    )

    result = pivot.merge(gaps, on=["scenario_label", "model"], how="left")

    columns = ["scenario_label", "model", "gap_f2"] + [
        metric for metric in metrics if metric in result.columns
    ]
    result = result[columns]

    if "f2" in result.columns:
        result = result.sort_values("f2", ascending=False)
    return result.reset_index(drop=True)


def _best_rows(frame: pd.DataFrame, target: str, stage: str) -> pd.DataFrame:
    subset = frame[
        (frame["target"] == target)
        & (frame["stage"] == stage)
        & (frame["metric"] == "f2")
        & (frame["model"] != "dummy_prior")
    ]
    if subset.empty:
        return subset
    return subset.sort_values("cv_mean", ascending=False).head(3)


def write_report(
    frame: pd.DataFrame,
    tuning_done: bool,
    targets: dict[str, str],
    split_info: dict,
    path: Path | None = None,
) -> Path:
    """Gera o relatorio Markdown da Fase 2."""
    path = Path(path or config.PHASE2_REPORT_MD)
    path.parent.mkdir(parents=True, exist_ok=True)

    stage = "tuned" if tuning_done else "default"
    lines: list[str] = []
    lines.append("# Fase 2 - Comparacao de modelos e cenarios")
    lines.append("")
    lines.append(
        "Metrica de otimizacao: **F2-score** (D8). Classe positiva: evasao "
        "(`Desistente = 1`)."
    )
    lines.append("")
    lines.append("## 1. Divisao treino/teste (congelada)")
    lines.append("")
    lines.append(f"- `RANDOM_STATE` = {config.RANDOM_STATE}")
    lines.append(f"- Teste = {config.TEST_SIZE:.0%} (estratificado)")
    lines.append(f"- Treino: {split_info['n_train']} linhas")
    lines.append(f"- Teste: {split_info['n_test']} linhas (preservado para a Fase 3)")
    lines.append(f"- Indices salvos em `{config.SPLIT_INDICES_PATH.name}`")
    lines.append("")

    for target_name, target_column in targets.items():
        lines.append(f"## 2. Alvo: {target_name}")
        lines.append("")
        if target_column == config.TARGET_SENSITIVITY_COLUMN:
            lines.append(
                "Analise de sensibilidade (D1): `Desistente` vs. `Graduado`, "
                "`Matriculado` excluido."
            )
            lines.append("")
        else:
            lines.append(
                "Alvo principal (D1): `Desistente = 1`; `Graduado` + "
                "`Matriculado` = 0."
            )
            lines.append("")

        best = _best_rows(frame, target_name, stage)
        if not best.empty:
            lines.append("### Melhores resultados (F2 na validacao cruzada)")
            lines.append("")
            columns = [
                column
                for column in (
                    "scenario_label",
                    "model",
                    "n_features",
                    "train_mean",
                    "cv_mean",
                    "cv_std",
                    "gap",
                )
                if column in best.columns
            ]
            lines.append(to_markdown(best[columns]))
            lines.append("")

        if tuning_done and stage == "tuned":
            params = (
                frame[
                    (frame["target"] == target_name)
                    & (frame["stage"] == "tuned")
                    & (frame["metric"] == "f2")
                ][["scenario_label", "model", "best_params"]]
                .drop_duplicates()
                .reset_index(drop=True)
            )
            lines.append("### Hiperparametros escolhidos")
            lines.append("")
            lines.append(to_markdown(params))
            lines.append("")

        lines.append("### Tabela completa (media dos 5 folds)")
        lines.append("")
        table = _key_table(frame, target_name, stage)
        lines.append(to_markdown(table))
        lines.append("")

    lines.append("## 3. Observacoes metodologicas")
    lines.append("")
    lines.append(
        "- Todo o pre-processamento (imputacao, padronizacao e one-hot) e "
        "reajustado dentro de cada fold: **sem vazamento**."
    )
    lines.append(
        "- `class_weight` foi testado apenas em Regressao Logistica e Random "
        "Forest, que oferecem suporte ao parametro."
    )
    lines.append(
        "- O desempenho do `dummy_prior` serve de referencia minima: qualquer "
        "modelo util precisa supera-lo com folga."
    )
    lines.append(
        "- O conjunto de **teste nao foi avaliado** nesta fase (D8)."
    )
    lines.append("")
    lines.append("## 4. Arquivos gerados")
    lines.append("")
    lines.append(f"- `{config.PHASE2_REPORT_MD.relative_to(config.PROJECT_ROOT)}`")
    lines.append(f"- `{config.CV_COMPARISON_CSV.relative_to(config.PROJECT_ROOT)}`")
    if tuning_done:
        lines.append(
            f"- `{config.TUNED_COMPARISON_CSV.relative_to(config.PROJECT_ROOT)}`"
        )
    lines.append("- `reports/figures/*.png`")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Execucao
# --------------------------------------------------------------------------- #


def run(
    scenarios: list[str] | None = None,
    tune: bool = True,
    include_sensitivity: bool = True,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Executa a Fase 2 completa e devolve a tabela de resultados."""
    ensure_output_dirs()

    df = load_processed_data()
    scenarios = scenarios or list(config.SCENARIOS)

    train_index, test_index, split_info = create_or_load_split(df)

    print("=" * 72)
    print("FASE 2 - Comparacao de modelos e cenarios")
    print("=" * 72)
    print(
        f"Base: {df.shape[0]} linhas | treino: {split_info['n_train']} | "
        f"teste: {split_info['n_test']} (preservado)"
    )
    print(f"Cenarios: {', '.join(scenarios)}")
    print(f"Ajuste de hiperparametros: {'sim' if tune else 'nao'}")
    print(flush=True)

    cv = StratifiedKFold(
        n_splits=config.N_SPLITS,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )
    scorers = make_scorers()

    targets: dict[str, str] = {"principal": config.TARGET_COLUMN}
    if include_sensitivity:
        targets["sensibilidade"] = config.TARGET_SENSITIVITY_COLUMN

    all_rows: list[dict] = []

    for target_name, target_column in targets.items():
        tuning_here = tune and target_column == config.TARGET_COLUMN

        for scenario_key in scenarios:
            X_train, y_train, _, _ = split_train_holdout(
                df, target_column, train_index, test_index
            )
            # X_train contem todas as colunas; o cenario seleciona quais usar.
            raw_columns, final_columns = resolve_scenario(df, scenario_key)

            print(
                f"\n[alvo={target_name}] cenario={scenario_key} "
                f"({len(final_columns)} features apos engineering)"
            )
            print(flush=True)

            rows = evaluate_models_on_train(
                X_train[raw_columns],
                y_train,
                final_columns,
                cv=cv,
                scorers=scorers,
                scenario_key=scenario_key,
                target_name=target_name,
                stage="default",
                tune=False,
                n_jobs=n_jobs,
                verbose=True,
            )
            all_rows.extend(rows)

            if tuning_here:
                print("    -- ajuste de hiperparametros (F2) --", flush=True)
                tuned = evaluate_models_on_train(
                    X_train[raw_columns],
                    y_train,
                    final_columns,
                    cv=cv,
                    scorers=scorers,
                    scenario_key=scenario_key,
                    target_name=target_name,
                    stage="tuned",
                    tune=True,
                    n_jobs=n_jobs,
                    verbose=True,
                )
                all_rows.extend(tuned)

    frame = build_table(all_rows)
    save_table(frame, config.CV_COMPARISON_CSV)

    #: Nome do alvo principal **dentro da tabela** (a coluna ``target`` guarda
    #: o rotulo, nao o nome da coluna do DataFrame).
    main_target = "principal"
    main_rows = frame[frame["target"] == main_target]
    save_table(main_rows, config.TUNED_COMPARISON_CSV)

    # --- graficos ---------------------------------------------------------- #
    generated: list[Path] = []
    for metric in ("f2", "recall", "roc_auc"):
        for stage in ("default", "tuned"):
            if frame[frame["stage"] == stage].empty:
                continue
            for path in (
                plot_metric_by_model(
                    main_rows,
                    metric,
                    main_target,
                    config.FIGURES_DIR / f"phase2_{metric}_por_modelo_{stage}.png",
                    stage=stage,
                ),
                plot_scenario_heatmap(
                    main_rows,
                    metric,
                    main_target,
                    config.FIGURES_DIR / f"phase2_{metric}_heatmap_{stage}.png",
                    stage=stage,
                ),
            ):
                if path:
                    generated.append(path)

    gap_path = plot_train_vs_cv_gap(
        main_rows,
        "f2",
        main_target,
        config.FIGURES_DIR / "phase2_f2_treino_vs_cv.png",
        stage="tuned" if tune else "default",
    )
    if gap_path:
        generated.append(gap_path)

    report_path = write_report(
        frame,
        tuning_done=tune,
        targets=targets,
        split_info=split_info,
    )

    print()
    print("=" * 72)
    print("RESUMO (F2 na validacao cruzada, alvo principal)")
    print("=" * 72)
    stage = "tuned" if tune else "default"
    summary = _key_table(main_rows, main_target, stage)
    if not summary.empty:
        print(
            summary.head(10).to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}",
            )
        )
    print()
    print(f"Tabela completa : {config.CV_COMPARISON_CSV}")
    print(f"Relatorio       : {report_path}")
    print(f"Figuras         : {len(generated)} arquivo(s) em {config.FIGURES_DIR}")

    return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fase 2 - comparacao de modelos e cenarios."
    )
    parser.add_argument(
        "--scenarios",
        default=",".join(config.SCENARIOS),
        help="Lista de cenarios separados por virgula.",
    )
    parser.add_argument(
        "--no-tuning",
        action="store_true",
        help="Pula o ajuste de hiperparametros (somente parametros padrao).",
    )
    parser.add_argument(
        "--targets",
        default="main,sensitivity",
        help="Quais alvos avaliar: 'main', 'sensitivity' ou ambos.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=config.N_JOBS,
        help=(
            "Paralelismo (-1 = todos os nucleos). Padrao serial para evitar "
            "avisos de multiprocessing em ambientes restritos."
        ),
    )
    args = parser.parse_args()

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    unknown = [s for s in scenarios if s not in config.SCENARIOS]
    if unknown:
        parser.error(
            f"cenarios desconhecidos: {unknown}. "
            f"Disponiveis: {list(config.SCENARIOS)}"
        )

    include_sensitivity = "sensitivity" in args.targets
    run(
        scenarios=scenarios,
        tune=not args.no_tuning,
        include_sensitivity=include_sensitivity,
        n_jobs=args.jobs,
    )


if __name__ == "__main__":
    main()
