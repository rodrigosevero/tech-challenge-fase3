"""Testes da API de inferencia (usada pelo app Streamlit)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src import config
from src.predict import (
    ModelNotFoundError,
    classify_risk,
    load_metadata,
    load_model,
    predict_risk,
)
from src.preprocessing import build_pipeline


@pytest.fixture()
def trained_bundle():
    """Pipeline pequeno ja treinado, sem depender do artefato real."""
    rng = np.random.default_rng(1)
    n = 60
    frame = pd.DataFrame(
        {
            "NotaAdmissao": rng.normal(130, 12, n),
            "UnidadesCurriculares1SemestreInscrito": rng.integers(1, 8, n),
            "UnidadesCurriculares1SemestreAprovado": rng.integers(0, 7, n),
            "UnidadesCurriculares1SemestreAvaliacoes": rng.integers(0, 10, n),
            "UnidadesCurriculares1SemestreGrau": rng.uniform(8, 18, n),
            "UnidadesCurriculares1SemestreSemAvaliacoes": rng.integers(0, 3, n),
        }
    )
    y = pd.Series((frame["UnidadesCurriculares1SemestreAprovado"] < 2).astype(int))

    columns = [
        "NotaAdmissao",
        "UnidadesCurriculares1SemestreInscrito",
        "UnidadesCurriculares1SemestreAprovado",
        "UnidadesCurriculares1SemestreAvaliacoes",
        "UnidadesCurriculares1SemestreGrau",
        "UnidadesCurriculares1SemestreSemAvaliacoes",
        "taxa_aprovacao_1sem",
        "disciplinas_nao_aprovadas_1sem",
        "razao_avaliacoes_inscrito_1sem",
    ]
    pipeline = build_pipeline(columns, LogisticRegression(max_iter=1000))
    pipeline.fit(frame, y)

    metadata = {
        "input_columns": list(frame.columns),
        "decision_threshold": 0.5,
        "scenario": "early_warning",
        "model_name": "logistic_balanced",
    }
    return pipeline, metadata, frame


# --------------------------------------------------------------------------- #
# Faixas de risco
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("probability", "expected_band"),
    [
        (0.01, "Baixo"),
        (0.34, "Baixo"),
        (0.35, "Moderado"),
        (0.64, "Moderado"),
        (0.65, "Alto"),
        (0.99, "Alto"),
    ],
)
def test_classify_risk_bands(probability: float, expected_band: str) -> None:
    """As faixas de risco respeitam os limites definidos."""
    label, advice = classify_risk(probability)

    assert label == expected_band
    assert advice


def test_classify_risk_always_gives_advice() -> None:
    """Toda faixa traz uma recomendacao de acao nao vazia."""
    for probability in np.linspace(0, 1, 11):
        _, advice = classify_risk(float(probability))
        assert advice


# --------------------------------------------------------------------------- #
# Inferencia
# --------------------------------------------------------------------------- #


def test_predict_risk_returns_expected_fields(trained_bundle) -> None:
    """A previsao traz probabilidade, limiar, alerta e faixa de risco."""
    pipeline, metadata, frame = trained_bundle
    payload = frame.iloc[0].to_dict()

    result = predict_risk(payload, model=pipeline, metadata=metadata)

    assert 0.0 <= result["probability"] <= 1.0
    assert result["threshold"] == 0.5
    assert isinstance(result["is_dropout_risk"], bool)
    assert result["risk_band"] in {"Baixo", "Moderado", "Alto"}
    assert result["recommendation"]


def test_predict_risk_ignores_extra_columns(trained_bundle) -> None:
    """Campos extras enviados pelo formulario nao quebram a previsao."""
    pipeline, metadata, frame = trained_bundle
    payload = frame.iloc[0].to_dict()
    payload["campo_desconhecido"] = "qualquer coisa"

    result = predict_risk(payload, model=pipeline, metadata=metadata)

    assert 0.0 <= result["probability"] <= 1.0


def test_predict_risk_tolerates_missing_columns(trained_bundle) -> None:
    """Campos ausentes sao imputados pelo pipeline (nao geram erro)."""
    pipeline, metadata, frame = trained_bundle
    payload = frame.iloc[0].to_dict()
    payload.pop("NotaAdmissao")
    payload.pop("UnidadesCurriculares1SemestreGrau")

    result = predict_risk(payload, model=pipeline, metadata=metadata)

    assert 0.0 <= result["probability"] <= 1.0


def test_predict_risk_accepts_dataframe_batch(trained_bundle) -> None:
    """Aceita varios estudantes de uma vez (formato de lote)."""
    pipeline, metadata, frame = trained_bundle

    result = predict_risk(frame.head(3), model=pipeline, metadata=metadata)

    assert "results" in result
    assert len(result["results"]) == 3
    assert all(0.0 <= item["probability"] <= 1.0 for item in result["results"])


def test_predict_risk_respects_threshold(trained_bundle) -> None:
    """O limiar dos metadados define a classificacao binaria."""
    pipeline, metadata, frame = trained_bundle

    batch = predict_risk(frame, model=pipeline, metadata=metadata)["results"]
    probabilities = [item["probability"] for item in batch]
    # escolhe o estudante com probabilidade mais proxima de 0.5 para que
    # exista margem de limiar nos dois sentidos
    index = int(np.argmin([abs(value - 0.5) for value in probabilities]))
    payload = frame.iloc[index].to_dict()
    central = probabilities[index]

    base = {"input_columns": list(frame.columns)}
    strict = predict_risk(
        payload,
        model=pipeline,
        metadata={**base, "decision_threshold": (central + 1.0) / 2},
    )
    lenient = predict_risk(
        payload,
        model=pipeline,
        metadata={**base, "decision_threshold": central / 2},
    )

    assert strict["is_dropout_risk"] is False
    assert lenient["is_dropout_risk"] is True


# --------------------------------------------------------------------------- #
# Artefatos reais
# --------------------------------------------------------------------------- #


def test_load_model_raises_when_missing(tmp_path) -> None:
    """Mensagem clara quando o artefato nao existe."""
    with pytest.raises(ModelNotFoundError, match="nao encontrado"):
        load_model(tmp_path / "inexistente.joblib")


def test_load_metadata_returns_empty_when_missing(tmp_path) -> None:
    """Sem metadados, devolve dicionario vazio em vez de erro."""
    assert load_metadata(tmp_path / "inexistente.json") == {}


@pytest.mark.skipif(
    not config.MODEL_PATH.exists(),
    reason="Modelo da Fase 3 ainda nao gerado (rode python -m src.select_model).",
)
def test_real_model_is_loadable_and_predicts() -> None:
    """O artefato real carrega e produz probabilidades validas."""
    model = load_model()
    metadata = load_metadata()

    assert metadata["model_name"]
    assert 0.0 < float(metadata["decision_threshold"]) < 1.0

    payload = {column: 0 for column in metadata["input_columns"]}
    result = predict_risk(payload, model=model, metadata=metadata)

    assert 0.0 <= result["probability"] <= 1.0
