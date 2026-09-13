"""Testes da especificacao do formulario da aplicacao."""

from __future__ import annotations

import pandas as pd
import pytest

from src.form_spec import (
    FIELD_SPECS,
    GROUP_ORDER,
    build_payload,
    categorical_options,
    columns_for_model,
    default_payload,
    group_fields,
    unknown_columns,
)

#: Campos que o modelo entregue realmente usa (Early Warning).
MODEL_COLUMNS = [
    "EstadoCivil",
    "Curso",
    "QualificacaoAnterior",
    "Nacionalidade",
    "Genero",
    "NecessidadesEspeciais",
    "Devedor",
    "MensalidadesEmDia",
    "Bolsista",
    "International",
    "NotaAdmissao",
    "QualificacaoAnteriorGrau",
    "UnidadesCurriculares1SemestreCreditado",
    "UnidadesCurriculares1SemestreInscrito",
    "UnidadesCurriculares1SemestreAvaliacoes",
    "UnidadesCurriculares1SemestreAprovado",
    "UnidadesCurriculares1SemestreGrau",
    "UnidadesCurriculares1SemestreSemAvaliacoes",
    "TaxaDesemprego",
    "TaxaInflacao",
    "PIB",
]


def test_all_model_columns_have_specs() -> None:
    """Todo campo exigido pelo modelo tem especificacao de formulario."""
    assert unknown_columns(MODEL_COLUMNS) == []


def test_specs_cover_both_scenarios() -> None:
    """As colunas do 2o semestre tambem estao previstas (cenario Completo)."""
    for column in (
        "UnidadesCurriculares2SemestreInscrito",
        "UnidadesCurriculares2SemestreAprovado",
        "UnidadesCurriculares2SemestreGrau",
    ):
        assert column in FIELD_SPECS


def test_columns_for_model_keeps_order_and_filters() -> None:
    """A ordem do modelo e preservada e campos desconhecidos sao descartados."""
    assert columns_for_model(MODEL_COLUMNS) == MODEL_COLUMNS
    assert columns_for_model(["CampoInexistente", "PIB"]) == ["PIB"]


def test_default_payload_is_complete() -> None:
    """O perfil inicial preenche todos os campos do modelo."""
    payload = default_payload(MODEL_COLUMNS)

    assert set(payload) == set(MODEL_COLUMNS)


@pytest.mark.parametrize("column", MODEL_COLUMNS)
def test_defaults_respect_the_spec(column: str) -> None:
    """Cada valor padrao e valido para o seu tipo de campo."""
    spec = FIELD_SPECS[column]
    value = spec.default

    if spec.kind == "numeric":
        assert spec.min_value <= float(value) <= spec.max_value  # type: ignore[arg-type]
    elif spec.kind == "binary":
        assert value in (0, 1)
    else:
        assert isinstance(value, str) and value


def test_group_fields_follows_defined_order() -> None:
    """Os grupos aparecem na ordem definida em ``GROUP_ORDER``."""
    grouped = group_fields(MODEL_COLUMNS)
    positions = [GROUP_ORDER.index(group) for group in grouped]

    assert positions == sorted(positions)
    assert "Cadastro" in grouped
    assert "2º semestre" not in grouped  # cenario Early Warning


def test_group_fields_includes_semester2_when_asked() -> None:
    """No cenario Completo, o grupo do 2o semestre aparece."""
    columns = MODEL_COLUMNS + ["UnidadesCurriculares2SemestreGrau"]
    grouped = group_fields(columns)

    assert "2º semestre" in grouped


def test_build_payload_filters_unknown_fields() -> None:
    """O payload enviado ao modelo contem apenas os campos esperados."""
    values = {column: FIELD_SPECS[column].default for column in MODEL_COLUMNS}
    values["campo_intruso"] = "x"

    payload = build_payload(values, MODEL_COLUMNS)

    assert "campo_intruso" not in payload
    assert set(payload) == set(MODEL_COLUMNS)


def test_build_payload_ignores_missing_values() -> None:
    """Campos ausentes nao entram no payload."""
    payload = build_payload({"PIB": 1.74}, MODEL_COLUMNS)

    assert payload == {"PIB": 1.74}


# --------------------------------------------------------------------------- #
# Opcoes dos campos categoricos
# --------------------------------------------------------------------------- #


def test_categorical_options_prefers_dataset_values() -> None:
    """Usa os valores observados na base processada."""
    data = pd.DataFrame({"Curso": ["Zeta", "Alfa", "Alfa", "Beta"]})

    options = categorical_options(data, "Curso")

    assert options == ["Alfa", "Beta", "Zeta"]


def test_categorical_options_uses_fallback_without_data() -> None:
    """Sem a base, usa a lista de contingencia."""
    options = categorical_options(None, "Genero")

    assert options == ["Feminino", "Masculino"]


def test_categorical_options_returns_empty_for_unknown() -> None:
    """Campo sem lista de contingencia devolve lista vazia."""
    assert categorical_options(None, "CampoInexistente") == []
