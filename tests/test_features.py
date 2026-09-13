"""Testes do feature engineering (transformacoes linha a linha)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import ENGINEERED_FEATURES, ENGINEERED_SEMESTER1_FEATURES
from src.features import (
    FeatureEngineer,
    add_engineered_features,
    engineered_feature_columns,
    resolve_scenario_columns,
)


def _base_frame() -> pd.DataFrame:
    """DataFrame minimo com as colunas de origem das features derivadas."""
    return pd.DataFrame(
        {
            "UnidadesCurriculares1SemestreInscrito": [6, 0, 4],
            "UnidadesCurriculares1SemestreAprovado": [3, 0, 4],
            "UnidadesCurriculares1SemestreAvaliacoes": [9, 0, 8],
            "UnidadesCurriculares2SemestreInscrito": [5, 0, 3],
            "UnidadesCurriculares2SemestreAprovado": [2, 0, 3],
            "UnidadesCurriculares1SemestreGrau": [12.0, 0.0, 14.0],
            "UnidadesCurriculares2SemestreGrau": [13.0, 0.0, 15.0],
            "UnidadesCurriculares1SemestreSemAvaliacoes": [0, 0, 1],
            "UnidadesCurriculares2SemestreSemAvaliacoes": [0, 0, 1],
        }
    )


def test_creates_all_engineered_features() -> None:
    """Todas as features derivadas sao criadas quando as origens existem."""
    engineered = add_engineered_features(_base_frame())

    for column in ENGINEERED_FEATURES:
        assert column in engineered.columns


def test_taxa_aprovacao_values() -> None:
    """Taxa de aprovacao = aprovadas / inscritas."""
    engineered = add_engineered_features(_base_frame())

    assert engineered["taxa_aprovacao_1sem"].tolist() == pytest.approx(
        [0.5, 0.0, 1.0]
    )
    assert engineered["taxa_aprovacao_2sem"].tolist() == pytest.approx(
        [0.4, 0.0, 1.0]
    )


def test_division_by_zero_is_safe() -> None:
    """Sem disciplinas inscritas, a taxa e zero (nao NaN nem erro)."""
    engineered = add_engineered_features(_base_frame())

    assert engineered["taxa_aprovacao_1sem"].iloc[1] == 0.0
    assert engineered["razao_avaliacoes_inscrito_1sem"].iloc[1] == 0.0
    assert not engineered["taxa_aprovacao_1sem"].isna().any()


def test_disciplinas_nao_aprovadas_never_negative() -> None:
    """Nao aprovadas e truncado em zero (nunca negativo)."""
    df = pd.DataFrame(
        {
            "UnidadesCurriculares1SemestreInscrito": [10, 2],
            "UnidadesCurriculares1SemestreAprovado": [4, 5],
        }
    )
    engineered = add_engineered_features(df)

    assert engineered["disciplinas_nao_aprovadas_1sem"].tolist() == [6, 0]


def test_delta_features() -> None:
    """Deltas entre semestres sao calculados corretamente."""
    engineered = add_engineered_features(_base_frame())

    assert engineered["delta_grau"].tolist() == pytest.approx([1.0, 0.0, 1.0])
    assert engineered["delta_taxa_aprovacao"].tolist() == pytest.approx(
        [-0.1, 0.0, 0.0]
    )
    assert engineered["total_sem_avaliacoes"].tolist() == [0, 0, 2]


def test_early_warning_frame_does_not_create_semester2_features() -> None:
    """Sem colunas do 2o semestre, as features dependentes nao sao criadas."""
    frame = _base_frame().drop(
        columns=[
            "UnidadesCurriculares2SemestreInscrito",
            "UnidadesCurriculares2SemestreAprovado",
            "UnidadesCurriculares2SemestreGrau",
            "UnidadesCurriculares2SemestreSemAvaliacoes",
        ]
    )
    engineered = add_engineered_features(frame)

    for column in ENGINEERED_SEMESTER1_FEATURES:
        assert column in engineered.columns
    assert "taxa_aprovacao_2sem" not in engineered.columns
    assert "delta_grau" not in engineered.columns


def test_does_not_mutate_input() -> None:
    """A funcao devolve copia e preserva o DataFrame original."""
    original = _base_frame()
    snapshot = original.copy(deep=True)
    add_engineered_features(original)

    pd.testing.assert_frame_equal(original, snapshot)


def test_engineered_feature_columns_matches_transform() -> None:
    """A descoberta de nomes (0 linhas) coincide com o transform real."""
    frame = _base_frame()
    expected = list(add_engineered_features(frame).columns)

    assert engineered_feature_columns(frame.columns) == expected


def test_resolve_scenario_columns_filters_unknown() -> None:
    """Colunas indisponiveis sao descartadas."""
    available = ["a", "b", "c"]
    assert resolve_scenario_columns(available, ["b", "z", "a"]) == ["b", "a"]


def test_feature_engineer_is_sklearn_compatible() -> None:
    """O transformer funciona dentro de um Pipeline (fit/transform)."""
    from sklearn.pipeline import Pipeline  # import local: clareza do escopo

    frame = _base_frame()
    pipeline = Pipeline([("engineer", FeatureEngineer())])

    transformed = pipeline.fit_transform(frame)
    assert "taxa_aprovacao_1sem" in transformed.columns

    names = pipeline.named_steps["engineer"].get_feature_names_out()
    assert "taxa_aprovacao_1sem" in list(names)


def test_feature_engineer_rejects_non_dataframe() -> None:
    """Entrada que nao seja DataFrame falha com mensagem clara."""
    with pytest.raises(TypeError):
        FeatureEngineer().fit(np.zeros((3, 2)))
