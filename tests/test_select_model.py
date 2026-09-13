"""Testes do diagnostico, da selecao de modelo e do limiar de decisao."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluate import best_threshold_by, threshold_analysis_table
from src.select_model import overfitting_diagnosis, select_final_candidate


# --------------------------------------------------------------------------- #
# Diagnostico de ajuste
# --------------------------------------------------------------------------- #


def test_diagnosis_detects_overfitting() -> None:
    """Gap grande entre treino e validacao indica overfitting."""
    result = overfitting_diagnosis(train_mean=0.95, cv_mean=0.77, test_value=0.78)

    assert result["diagnosis"] == "overfitting"
    assert result["gap_train_cv"] == pytest.approx(0.18)
    assert "memoriza" in result["explanation"]


def test_diagnosis_detects_underfitting() -> None:
    """Desempenho baixo e proximo nos dois niveis indica underfitting."""
    result = overfitting_diagnosis(train_mean=0.45, cv_mean=0.44, test_value=0.43)

    assert result["diagnosis"] == "underfitting"


def test_diagnosis_reports_adequate_fit() -> None:
    """Treino e validacao proximos e em nivel util = ajuste adequado."""
    result = overfitting_diagnosis(train_mean=0.79, cv_mean=0.78, test_value=0.77)

    assert result["diagnosis"] == "ajuste_adequado"
    assert result["gap_cv_test"] == pytest.approx(0.01)


# --------------------------------------------------------------------------- #
# Selecao do modelo final
# --------------------------------------------------------------------------- #


def _phase2_frame(rows: list[dict]) -> pd.DataFrame:
    """Monta um recorte da tabela da Fase 2 para os testes de selecao."""
    base = {
        "target": "principal",
        "stage": "tuned",
        "metric": "f2",
        "scenario": "early_warning",
        "scenario_label": "Early Warning",
    }
    return pd.DataFrame([{**base, **row} for row in rows])


def test_selection_prefers_simpler_model_within_margin() -> None:
    """Empate tecnico em F2 -> escolhe o modelo com menor gap treino-CV."""
    frame = _phase2_frame(
        [
            {"model": "random_forest_balanced", "cv_mean": 0.800, "gap": 0.200},
            {"model": "logistic_balanced", "cv_mean": 0.795, "gap": 0.010},
        ]
    )
    chosen, justification = select_final_candidate(frame, "early_warning", margin=0.01)

    assert chosen == "logistic_balanced"
    assert justification["best_by_f2"] == "random_forest_balanced"
    assert "Empate tecnico" in justification["reason"]


def test_selection_keeps_clear_winner() -> None:
    """Vantagem de F2 fora da margem -> mantem o melhor modelo."""
    frame = _phase2_frame(
        [
            {"model": "random_forest_balanced", "cv_mean": 0.900, "gap": 0.200},
            {"model": "logistic_balanced", "cv_mean": 0.500, "gap": 0.010},
        ]
    )
    chosen, _ = select_final_candidate(frame, "early_warning", margin=0.01)

    assert chosen == "random_forest_balanced"


def test_selection_ignores_dummy_baseline() -> None:
    """O baseline trivial nunca e escolhido, mesmo se liderar."""
    frame = _phase2_frame(
        [
            {"model": "dummy_prior", "cv_mean": 0.900, "gap": 0.000},
            {"model": "logistic_balanced", "cv_mean": 0.700, "gap": 0.010},
        ]
    )
    chosen, _ = select_final_candidate(frame, "early_warning")

    assert chosen == "logistic_balanced"


def test_selection_raises_for_unknown_scenario() -> None:
    """Cenario sem resultados levanta erro explicativo."""
    frame = _phase2_frame(
        [{"model": "logistic_balanced", "cv_mean": 0.700, "gap": 0.010}]
    )
    with pytest.raises(ValueError, match="Sem resultados"):
        select_final_candidate(frame, "cenario_inexistente")


# --------------------------------------------------------------------------- #
# Limiar de decisao
# --------------------------------------------------------------------------- #


def test_best_threshold_maximizes_metric() -> None:
    """O limiar escolhido e o que maximiza a metrica pedida."""
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_proba = np.array([0.05, 0.10, 0.35, 0.55, 0.90, 0.20, 0.75, 0.15])

    table = threshold_analysis_table(y_true, y_proba, [0.2, 0.4, 0.6, 0.8])
    chosen = best_threshold_by(table, metric="f2")

    best_row = table.sort_values("f2", ascending=False).iloc[0]
    assert chosen == pytest.approx(best_row["threshold"])


def test_best_threshold_breaks_tie_toward_half() -> None:
    """Em empate, prefere o limiar mais proximo de 0.5."""
    y_true = np.array([1, 1, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.2, 0.1])

    # 0.25 e 0.40 produzem a mesma classificacao (limiar irrelevante no intervalo)
    table = threshold_analysis_table(y_true, y_proba, [0.25, 0.40])
    chosen = best_threshold_by(table, metric="f2")

    assert chosen == pytest.approx(0.40)


def test_threshold_table_has_confusion_counts() -> None:
    """A tabela traz os quatro quadrantes da matriz de confusao."""
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.6, 0.7, 0.2])

    table = threshold_analysis_table(y_true, y_proba, [0.5])
    row = table.iloc[0]

    assert row["tp"] == 1  # 0.7 >= 0.5, real 1
    assert row["fp"] == 1  # 0.6 >= 0.5, real 0
    assert row["fn"] == 1  # 0.2 <  0.5, real 1
    assert row["tn"] == 1  # 0.1 <  0.5, real 0
    assert row["recall"] == pytest.approx(0.5)
