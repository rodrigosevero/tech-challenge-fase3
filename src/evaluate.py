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
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (  # noqa: E402
    cross_validate,
    learning_curve,
    validation_curve,
)

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


# --------------------------------------------------------------------------- #
# Fase 3: metricas por limiar, matriz de confusao e curvas
# --------------------------------------------------------------------------- #


def compute_threshold_metrics(
    y_true,
    y_proba,
    threshold: float,
) -> dict:
    """Metricas completas para um limiar de decisao especifico.

    ``roc_auc`` nao depende do limiar (mede o *ranking*), por isso e calculado
    uma unica vez a partir das probabilidades.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(
                y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0
            )
        ),
        "f1": float(
            f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        "f2": f2_score(y_true, y_pred),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def threshold_analysis_table(
    y_true,
    y_proba,
    thresholds: Sequence[float],
) -> pd.DataFrame:
    """Tabela de metricas para cada limiar avaliado (uso na validacao)."""
    rows = [compute_threshold_metrics(y_true, y_proba, t) for t in thresholds]
    return pd.DataFrame(rows)


def best_threshold_by(
    table: pd.DataFrame,
    metric: str = "f2",
) -> float:
    """Limiar que maximiza a metrica escolhida (empate -> o mais proximo de 0,5)."""
    if table.empty:
        return 0.5
    ordered = table.assign(
        _distance=(table["threshold"] - 0.5).abs()
    ).sort_values([metric, "_distance"], ascending=[False, True])
    return float(ordered.iloc[0]["threshold"])


def plot_confusion_matrix(
    y_true,
    y_pred,
    path: Path,
    title: str = "Matriz de confusao",
) -> Path:
    """Matriz de confusao em contagens e em proporcao por classe real."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    normalized = confusion_matrix(y_true, y_pred, labels=[0, 1], normalize="true")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = ["Nao evasao (0)", "Evasao (1)"]

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[0],
        cbar=False,
    )
    axes[0].set_title(f"{title} - contagens")
    axes[0].set_xlabel("Previsto")
    axes[0].set_ylabel("Real")

    sns.heatmap(
        normalized,
        annot=True,
        fmt=".1%",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[1],
        cbar=False,
        vmin=0,
        vmax=1,
    )
    axes[1].set_title(f"{title} - proporcao por classe real")
    axes[1].set_xlabel("Previsto")
    axes[1].set_ylabel("Real")

    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_roc_curve(y_true, y_proba, path: Path, title: str = "Curva ROC") -> Path:
    """Curva ROC com a area sob a curva (independente do limiar)."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    fpr, tpr, _ = roc_curve(y_true, y_proba, pos_label=POSITIVE_LABEL)
    auc_value = roc_auc_score(y_true, y_proba)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(6.5, 6))
    axis.plot(fpr, tpr, linewidth=2, label=f"Modelo (AUC = {auc_value:.3f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Aleatorio")
    axis.set_title(title)
    axis.set_xlabel("Taxa de falsos positivos")
    axis.set_ylabel("Taxa de verdadeiros positivos (recall)")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_precision_recall_curve(
    y_true,
    y_proba,
    path: Path,
    title: str = "Curva Precision-Recall",
) -> Path:
    """Curva precision-recall, mais informativa em bases desbalanceadas."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    precision, recall, _ = precision_recall_curve(
        y_true, y_proba, pos_label=POSITIVE_LABEL
    )
    average_precision = average_precision_score(
        y_true, y_proba, pos_label=POSITIVE_LABEL
    )
    baseline = float(np.mean(y_true == POSITIVE_LABEL))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(6.5, 6))
    axis.plot(recall, precision, linewidth=2, label=f"Modelo (AP = {average_precision:.3f})")
    axis.axhline(
        baseline,
        linestyle="--",
        color="grey",
        label=f"Baseline (prevalencia = {baseline:.3f})",
    )
    axis.set_title(title)
    axis.set_xlabel("Recall (evasao)")
    axis.set_ylabel("Precision (evasao)")
    axis.legend(loc="lower left")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_threshold_tradeoff(
    table: pd.DataFrame,
    path: Path,
    chosen_threshold: float | None = None,
    title: str = "Metricas x limiar de decisao",
) -> Path:
    """Mostra como precision, recall, F1 e F2 variam com o limiar (D8)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(9, 6))
    for metric in ("precision", "recall", "f1", "f2"):
        if metric in table.columns:
            axis.plot(
                table["threshold"], table[metric], marker="o", label=metric.upper()
            )
    if chosen_threshold is not None:
        axis.axvline(
            chosen_threshold,
            linestyle="--",
            color="black",
            label=f"Limiar escolhido = {chosen_threshold:.2f}",
        )
    axis.set_title(title)
    axis.set_xlabel("Limiar de decisao (probabilidade de evasao)")
    axis.set_ylabel("Valor da metrica")
    axis.set_ylim(0, 1)
    axis.legend(loc="best")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_learning_curve_analysis(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    path: Path,
    scoring,
    train_sizes: Sequence[float] = (0.15, 0.3, 0.45, 0.6, 0.8, 1.0),
    n_jobs: int = 1,
    title: str = "Curva de aprendizado",
) -> tuple[Path, pd.DataFrame]:
    """Curva de aprendizado (usa **apenas** o treino).

    Evidencia overfitting (gap que nao fecha com mais dados) e underfitting
    (ambas as curvas baixas e proximas).
    """
    sizes, train_scores, validation_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        scoring=scoring,
        train_sizes=list(train_sizes),
        n_jobs=n_jobs,
        error_score="raise",
        shuffle=False,
    )

    table = pd.DataFrame(
        {
            "train_size": sizes,
            "train_mean": train_scores.mean(axis=1),
            "train_std": train_scores.std(axis=1),
            "cv_mean": validation_scores.mean(axis=1),
            "cv_std": validation_scores.std(axis=1),
        }
    )
    table["gap"] = table["train_mean"] - table["cv_mean"]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(9, 6))
    axis.plot(table["train_size"], table["train_mean"], marker="o", label="Treino")
    axis.fill_between(
        table["train_size"],
        table["train_mean"] - table["train_std"],
        table["train_mean"] + table["train_std"],
        alpha=0.15,
    )
    axis.plot(
        table["train_size"], table["cv_mean"], marker="s", label="Validacao cruzada"
    )
    axis.fill_between(
        table["train_size"],
        table["cv_mean"] - table["cv_std"],
        table["cv_mean"] + table["cv_std"],
        alpha=0.15,
    )
    axis.set_title(title)
    axis.set_xlabel("Numero de exemplos de treino")
    axis.set_ylabel("F2 (evasao)")
    axis.set_ylim(0, 1)
    axis.legend(loc="lower right")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path, table


def plot_validation_curve_analysis(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    param_name: str,
    param_range: Sequence,
    cv,
    path: Path,
    scoring,
    n_jobs: int = 1,
    title: str = "Curva de validacao",
) -> tuple[Path, pd.DataFrame]:
    """Curva de validacao para um hiperparametro-chave (usa apenas o treino)."""
    train_scores, validation_scores = validation_curve(
        estimator,
        X,
        y,
        param_name=param_name,
        param_range=list(param_range),
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        error_score="raise",
    )

    table = pd.DataFrame(
        {
            "param_value": [str(value) for value in param_range],
            "train_mean": train_scores.mean(axis=1),
            "cv_mean": validation_scores.mean(axis=1),
            "cv_std": validation_scores.std(axis=1),
        }
    )
    table["gap"] = table["train_mean"] - table["cv_mean"]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    positions = np.arange(len(param_range))
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.plot(positions, table["train_mean"], marker="o", label="Treino")
    axis.plot(positions, table["cv_mean"], marker="s", label="Validacao cruzada")
    axis.fill_between(
        positions,
        table["cv_mean"] - table["cv_std"],
        table["cv_mean"] + table["cv_std"],
        alpha=0.15,
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(table["param_value"])
    axis.set_title(f"{title}: {param_name}")
    axis.set_xlabel(param_name)
    axis.set_ylabel("F2 (evasao)")
    axis.set_ylim(0, 1)
    axis.legend(loc="lower right")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path, table
