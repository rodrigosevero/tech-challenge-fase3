"""Metricas, agregacao de resultados de validacao cruzada e geracao de graficos.

A metrica de otimizacao e o **F2-score** (D8), que pondera o recall 4x mais que
a precision -- coerente com o objetivo de nao deixar evadir sem ser notado, sem
maximizar recall isoladamente.

A classe positiva e sempre ``1`` (evasao / ``Desistente``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sem interface grafica (scripts e CI)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate  # noqa: E402

from .config import (  # noqa: E402
    FIGURES_DIR,
    N_JOBS,
    POSITIVE_LABEL,
    REPORTED_METRICS,
    TABLES_DIR,
)

#: Colunas padronizadas das tabelas de resultado.
TABLE_COLUMNS: tuple[str, ...] = (
    "scenario",
    "scenario_label",
    "target",
    "stage",
    "model",
    "metric",
    "train_mean",
    "cv_mean",
    "cv_std",
    "gap",
)


def f2_score(y_true, y_pred) -> float:
    """Atalho para o F2-score com a classe positiva = evasao."""
    return float(
        fbeta_score(
            y_true,
            y_pred,
            beta=2,
            pos_label=POSITIVE_LABEL,
            zero_division=0,
        )
    )


def make_scorers() -> dict[str, object]:
    """Conjunto de scorers reportados, com a classe positiva = evasao."""
    return {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(
            precision_score, pos_label=POSITIVE_LABEL, zero_division=0
        ),
        "recall": make_scorer(
            recall_score, pos_label=POSITIVE_LABEL, zero_division=0
        ),
        "f1": make_scorer(
            f1_score, pos_label=POSITIVE_LABEL, zero_division=0
        ),
        "f2": make_scorer(
            fbeta_score,
            beta=2,
            pos_label=POSITIVE_LABEL,
            zero_division=0,
        ),
        "roc_auc": make_scorer(roc_auc_score, response_method="predict_proba"),
    }


def cross_validate_pipeline(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    scorers: Mapping[str, object] | None = None,
    n_jobs: int | None = None,
) -> dict:
    """Executa ``cross_validate`` guardando metricas de treino e de validacao.

    Como o ``pipeline`` completo entra como estimador, cada fold reajusta
    imputacao, escala e encoding apenas com os seus dados de treino -- sem
    vazamento entre folds.
    """
    scorers = dict(scorers or make_scorers())
    if n_jobs is None:
        n_jobs = N_JOBS
    return cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scorers,
        n_jobs=n_jobs,
        return_train_score=True,
        error_score="raise",
    )


def summarize_cv(
    raw_results: Mapping[str, object],
    metrics: Sequence[str] = REPORTED_METRICS,
) -> list[dict]:
    """Converte a saida de ``cross_validate`` em linhas metric x treino/CV."""
    rows: list[dict] = []
    for metric in metrics:
        test_key = f"test_{metric}"
        if test_key not in raw_results:
            continue

        cv_values = np.asarray(raw_results[test_key], dtype=float)
        row: dict[str, float] = {
            "cv_mean": float(cv_values.mean()),
            "cv_std": float(cv_values.std(ddof=0)),
        }

        train_key = f"train_{metric}"
        if train_key in raw_results:
            train_values = np.asarray(raw_results[train_key], dtype=float)
            row["train_mean"] = float(train_values.mean())
            row["gap"] = row["train_mean"] - row["cv_mean"]
        else:
            row["train_mean"] = math.nan
            row["gap"] = math.nan

        row["metric"] = metric
        rows.append(row)
    return rows


def build_table(rows: Sequence[dict]) -> pd.DataFrame:
    """Monta a tabela final na ordem padronizada de colunas."""
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return pd.DataFrame(columns=list(TABLE_COLUMNS))
    ordered = [column for column in TABLE_COLUMNS if column in frame.columns]
    remaining = [column for column in frame.columns if column not in ordered]
    return frame[ordered + remaining]


def save_table(frame: pd.DataFrame, path: Path) -> Path:
    """Salva uma tabela em CSV (criando o diretorio se necessario)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def pivot_metric(frame: pd.DataFrame, metric: str, target: str) -> pd.DataFrame:
    """Tabela cenarios (linhas) x modelos (colunas) para uma metrica."""
    subset = frame[(frame["metric"] == metric) & (frame["target"] == target)]
    if subset.empty:
        return pd.DataFrame()
    return subset.pivot_table(
        index="scenario_label", columns="model", values="cv_mean"
    )


def plot_metric_by_model(
    frame: pd.DataFrame,
    metric: str,
    target: str,
    path: Path,
    stage: str | None = None,
) -> Path | None:
    """Grafico de barras agrupadas: metrica por modelo, separado por cenario."""
    subset = frame[(frame["metric"] == metric) & (frame["target"] == target)]
    if stage is not None:
        subset = subset[subset["stage"] == stage]
    if subset.empty:
        return None

    order = (
        subset.groupby("model")["cv_mean"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=subset,
        x="model",
        y="cv_mean",
        hue="scenario_label",
        order=order,
        ax=axis,
    )
    axis.set_title(
        f"{metric.upper()} na validacao cruzada por modelo e cenario"
        f" (alvo: {target})"
    )
    axis.set_xlabel("Modelo")
    axis.set_ylabel(f"{metric.upper()} (media dos folds)")
    axis.set_ylim(0, 1)
    plt.setp(axis.get_xticklabels(), rotation=30, ha="right")
    axis.legend(title="Cenario", bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_scenario_heatmap(
    frame: pd.DataFrame,
    metric: str,
    target: str,
    path: Path,
    stage: str | None = None,
) -> Path | None:
    """Mapa de calor cenario x modelo para uma metrica."""
    subset = frame
    if stage is not None:
        subset = subset[subset["stage"] == stage]
    table = pivot_metric(subset, metric, target)
    if table.empty:
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(11, 5))
    sns.heatmap(
        table,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        ax=axis,
        cbar_kws={"label": f"{metric.upper()} (media dos folds)"},
    )
    axis.set_title(f"{metric.upper()} por cenario e modelo (alvo: {target})")
    axis.set_xlabel("Modelo")
    axis.set_ylabel("Cenario")
    plt.setp(axis.get_xticklabels(), rotation=30, ha="right")
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_train_vs_cv_gap(
    frame: pd.DataFrame,
    metric: str,
    target: str,
    path: Path,
    stage: str | None = None,
) -> Path | None:
    """Compara desempenho de treino e de validacao (indicador de overfitting)."""
    subset = frame[(frame["metric"] == metric) & (frame["target"] == target)].copy()
    if stage is not None:
        subset = subset[subset["stage"] == stage]
    if subset.empty:
        return None

    subset["label"] = subset["scenario_label"] + " | " + subset["model"]
    subset = subset.sort_values("gap", ascending=False).head(20)
    melted = subset.melt(
        id_vars="label",
        value_vars=["train_mean", "cv_mean"],
        var_name="conjunto",
        value_name="valor",
    )
    melted["conjunto"] = melted["conjunto"].map(
        {"train_mean": "Treino", "cv_mean": "Validacao cruzada"}
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(11, 7))
    sns.barplot(data=melted, y="label", x="valor", hue="conjunto", ax=axis)
    axis.set_title(
        f"{metric.upper()}: treino vs. validacao cruzada (maiores gaps no topo)"
    )
    axis.set_xlabel(f"{metric.upper()}")
    axis.set_ylabel("")
    axis.set_xlim(0, 1)
    axis.legend(title="Conjunto")
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def to_markdown(frame: pd.DataFrame, float_format: str = "%.4f") -> str:
    """Renderiza um DataFrame como tabela Markdown.

    Implementado manualmente para nao depender do pacote ``tabulate``.
    """
    if frame.empty:
        return "_(sem resultados)_\n"

    def _format(value: object, column: str) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            if math.isnan(value):
                return ""
            return float_format % value
        return str(value)

    subset = frame.copy()
    headers = [str(column) for column in subset.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for _, row in subset.iterrows():
        cells = [
            _format(row[column], str(column)) for column in subset.columns
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def ensure_output_dirs() -> None:
    """Garante que os diretorios de saida existam."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
