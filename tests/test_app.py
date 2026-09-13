"""Testes da aplicacao Streamlit (sem subir a interface).

O foco e garantir que os **exemplos de demonstracao** realmente funcionam com o
modelo entregue e que a aplicacao e importavel (sem efeitos colaterais no import).
"""

from __future__ import annotations

import pytest

from src import config
from src.form_spec import FIELD_SPECS, build_payload, columns_for_model, unknown_columns
from src.predict import load_metadata, load_model, predict_risk

pytestmark = pytest.mark.skipif(
    not config.MODEL_PATH.exists(),
    reason="Modelo da Fase 3 ainda nao gerado (rode python -m src.select_model).",
)


@pytest.fixture(scope="module")
def app_module():
    """Importa o modulo da aplicacao."""
    from app import streamlit_app

    return streamlit_app


@pytest.fixture(scope="module")
def bundle():
    """Modelo + metadados reais, carregados uma vez."""
    return load_model(), load_metadata()


def test_app_imports_without_side_effects(app_module) -> None:
    """O import nao executa a interface (tudo dentro de main())."""
    assert hasattr(app_module, "main")
    assert callable(app_module.main)


def test_examples_cover_all_model_fields(app_module, bundle) -> None:
    """Os exemplos de demonstracao preenchem todos os campos exigidos."""
    _, metadata = bundle
    expected = set(columns_for_model(metadata["input_columns"]))

    for example in (app_module.HIGH_RISK_EXAMPLE, app_module.LOW_RISK_EXAMPLE):
        assert expected.issubset(set(example))


def test_examples_use_valid_field_names(app_module, bundle) -> None:
    """Nenhum exemplo usa um campo desconhecido ou fora do modelo."""
    _, metadata = bundle
    expected = set(columns_for_model(metadata["input_columns"]))

    for example in (app_module.HIGH_RISK_EXAMPLE, app_module.LOW_RISK_EXAMPLE):
        assert set(example) <= expected
        assert unknown_columns(list(example)) == []


def test_high_risk_example_is_flagged(app_module, bundle) -> None:
    """O exemplo de risco alto aciona o alerta e a faixa 'Alto'."""
    model, metadata = bundle
    payload = build_payload(
        app_module.HIGH_RISK_EXAMPLE, columns_for_model(metadata["input_columns"])
    )

    result = predict_risk(payload, model=model, metadata=metadata)

    assert result["is_dropout_risk"] is True
    assert result["risk_band"] == "Alto"
    assert result["probability"] > 0.65


def test_low_risk_example_is_not_flagged(app_module, bundle) -> None:
    """O exemplo de risco baixo nao aciona o alerta."""
    model, metadata = bundle
    payload = build_payload(
        app_module.LOW_RISK_EXAMPLE, columns_for_model(metadata["input_columns"])
    )

    result = predict_risk(payload, model=model, metadata=metadata)

    assert result["is_dropout_risk"] is False
    assert result["risk_band"] == "Baixo"
    assert result["probability"] < 0.35


def test_examples_are_distinguishable(app_module, bundle) -> None:
    """Os dois exemplos produzem riscos bem diferentes (boa demonstracao)."""
    model, metadata = bundle
    columns = columns_for_model(metadata["input_columns"])

    alto = predict_risk(
        build_payload(app_module.HIGH_RISK_EXAMPLE, columns),
        model=model,
        metadata=metadata,
    )
    baixo = predict_risk(
        build_payload(app_module.LOW_RISK_EXAMPLE, columns),
        model=model,
        metadata=metadata,
    )

    assert alto["probability"] - baixo["probability"] > 0.5
