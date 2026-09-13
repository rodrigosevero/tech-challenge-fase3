"""Testes do pre-processamento (ColumnTransformer e Pipeline)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src import config
from src.features import add_engineered_features, engineered_feature_columns
from src.preprocessing import (
    build_pipeline,
    build_preprocessor,
    split_feature_groups,
)

_RANDOM_STATE = 0


def _synthetic(include_semester2: bool = True, n: int = 80) -> pd.DataFrame:
    """Base sintetica com as colunas usadas pelos cenarios."""
    rng = np.random.default_rng(_RANDOM_STATE)
    data: dict[str, object] = {
        "EstadoCivil": (["Solteiro", "Casado"] * (n // 2))[:n],
        "Curso": (["Enfermagem", "Gestao"] * (n // 2))[:n],
        "QualificacaoAnterior": ["Ensino Secundario"] * n,
        "Nacionalidade": ["Portugues"] * n,
        "Genero": (["Feminino", "Masculino"] * (n // 2))[:n],
        "NecessidadesEspeciais": np.zeros(n, dtype=int),
        "Devedor": rng.integers(0, 2, n),
        "MensalidadesEmDia": rng.integers(0, 2, n),
        "Bolsista": rng.integers(0, 2, n),
        "International": np.zeros(n, dtype=int),
        "NotaAdmissao": rng.normal(127, 10, n),
        "QualificacaoAnteriorGrau": rng.normal(133, 10, n),
        "UnidadesCurriculares1SemestreCreditado": rng.integers(0, 10, n),
        "UnidadesCurriculares1SemestreInscrito": rng.integers(1, 10, n),
        "UnidadesCurriculares1SemestreAvaliacoes": rng.integers(0, 12, n),
        "UnidadesCurriculares1SemestreAprovado": rng.integers(0, 8, n),
        "UnidadesCurriculares1SemestreGrau": rng.uniform(10, 16, n),
        "UnidadesCurriculares1SemestreSemAvaliacoes": rng.integers(0, 3, n),
        "TaxaDesemprego": rng.uniform(7, 16, n),
        "TaxaInflacao": rng.uniform(-1, 4, n),
        "PIB": rng.uniform(-4, 3, n),
    }
    if include_semester2:
        data.update(
            {
                "UnidadesCurriculares2SemestreCreditado": rng.integers(0, 10, n),
                "UnidadesCurriculares2SemestreInscrito": rng.integers(1, 10, n),
                "UnidadesCurriculares2SemestreAvaliacoes": rng.integers(0, 12, n),
                "UnidadesCurriculares2SemestreAprovado": rng.integers(0, 8, n),
                "UnidadesCurriculares2SemestreGrau": rng.uniform(10, 16, n),
                "UnidadesCurriculares2SemestreSemAvaliacoes": rng.integers(
                    0, 3, n
                ),
            }
        )
    return pd.DataFrame(data)


def _scenario_columns(
    df: pd.DataFrame, scenario_columns: tuple[str, ...]
) -> tuple[list[str], list[str]]:
    """(colunas de entrada, colunas apos engineering) para um cenario."""
    engineered = set(config.ENGINEERED_FEATURES)
    raw = [
        column
        for column in scenario_columns
        if column not in engineered and column in df.columns
    ]
    available = engineered_feature_columns(raw)
    final = [column for column in scenario_columns if column in available]
    return raw, final


# --------------------------------------------------------------------------- #
# split_feature_groups
# --------------------------------------------------------------------------- #


def test_split_feature_groups_classifies_correctly() -> None:
    """Categorias, flags e numericas vao para grupos distintos."""
    numeric, binary, categorical = split_feature_groups(
        [
            "NotaAdmissao",
            "Devedor",
            "Curso",
            "taxa_aprovacao_1sem",
        ]
    )

    assert set(categorical) == {"Curso"}
    assert set(binary) == {"Devedor"}
    assert set(numeric) == {"NotaAdmissao", "taxa_aprovacao_1sem"}


# --------------------------------------------------------------------------- #
# build_preprocessor
# --------------------------------------------------------------------------- #


def test_preprocessor_produces_no_missing_values() -> None:
    """Imputacao cobre valores ausentes nas colunas usadas."""
    df = _synthetic()
    raw, final = _scenario_columns(df, config.EARLY_WARNING_FEATURES)

    df_with_nan = df.copy()
    df_with_nan.loc[0, "NotaAdmissao"] = np.nan
    df_with_nan.loc[1, "Curso"] = None

    # o pre-processador opera **apos** o feature engineering
    engineered = add_engineered_features(df_with_nan[raw])

    preprocessor = build_preprocessor(final)
    matrix = preprocessor.fit_transform(engineered[final])

    assert matrix.shape[0] == df.shape[0]
    assert not matrix.isna().any().any(), "restaram valores ausentes"


def test_preprocessor_groups_rare_categories() -> None:
    """Categorias raras sao agrupadas (min_frequency), reduzindo dimensao."""
    df = _synthetic()
    df.loc[0, "Curso"] = "CursoRarissimo"

    _, _, categorical = split_feature_groups(["Curso"])
    assert categorical == ["Curso"]

    preprocessor = build_preprocessor(["Curso"])
    matrix = preprocessor.fit_transform(df[["Curso"]])

    # apenas 2 categorias relevantes -> no maximo 3 colunas (2 + infrequentes)
    assert matrix.shape[1] <= 3


# --------------------------------------------------------------------------- #
# build_pipeline (integracao)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("include_semester2", [True, False])
def test_pipeline_fits_and_predicts(include_semester2: bool) -> None:
    """Pipeline completo funciona nos cenarios Completo e Early Warning."""
    df = _synthetic(include_semester2=include_semester2)
    scenario = (
        config.FULL_FEATURES
        if include_semester2
        else config.EARLY_WARNING_FEATURES
    )
    raw, final = _scenario_columns(df, scenario)
    y = pd.Series([0, 1] * (len(df) // 2))

    pipeline = build_pipeline(final, LogisticRegression(max_iter=1000))
    pipeline.fit(df[raw], y)
    probabilities = pipeline.predict_proba(df[raw])[:, 1]

    assert probabilities.shape == (len(df),)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_pipeline_handles_unseen_categories() -> None:
    """Categorias novas nao quebram a predicao (handle_unknown='ignore')."""
    df = _synthetic()
    raw, final = _scenario_columns(df, config.EARLY_WARNING_FEATURES)
    y = pd.Series([0, 1] * (len(df) // 2))

    pipeline = build_pipeline(final, LogisticRegression(max_iter=1000))
    pipeline.fit(df[raw], y)

    unseen = df[raw].copy()
    unseen.loc[0, "Curso"] = "CursoQueNaoExisteNoTreino"
    unseen.loc[1, "EstadoCivil"] = "EstadoCivilNovo"

    probabilities = pipeline.predict_proba(unseen)[:, 1]
    assert probabilities.shape == (len(df),)


def test_pipeline_engineering_is_inside_the_artifact() -> None:
    """O feature engineering faz parte da pipeline (vai para o joblib)."""
    df = _synthetic()
    raw, final = _scenario_columns(df, config.EARLY_WARNING_FEATURES)
    y = pd.Series([0, 1] * (len(df) // 2))

    pipeline = build_pipeline(final, LogisticRegression(max_iter=1000))
    pipeline.fit(df[raw], y)

    assert "engineer" in pipeline.named_steps
    engineered = pipeline.named_steps["engineer"].transform(df[raw])
    assert "taxa_aprovacao_1sem" in engineered.columns


def test_preprocessor_only_uses_training_statistics() -> None:
    """A padronizacao e ajustada so no treino (sem vazamento).

    Depois de ajustar no treino, a media das colunas numericas transformadas no
    proprio treino fica proxima de zero -- e nao muda ao transformar o teste.
    """
    df = _synthetic()
    raw, final = _scenario_columns(df, config.EARLY_WARNING_FEATURES)

    engineered = add_engineered_features(df[raw])
    train = engineered[final].iloc[:60]
    test = engineered[final].iloc[60:]

    preprocessor = build_preprocessor(final)
    preprocessor.fit(train)

    train_matrix = preprocessor.transform(train)
    test_matrix = preprocessor.transform(test)

    # o scaler aprendeu com o treino: media ~0 nas colunas com variancia
    numeric_columns = [c for c in train_matrix.columns if "Grau" in c or c == "NotaAdmissao"]
    for column in numeric_columns:
        assert abs(float(train_matrix[column].mean())) < 0.2

    # transformar o teste nao deve alterar o que foi aprendido no treino
    repeated = preprocessor.transform(train)
    pd.testing.assert_frame_equal(train_matrix, repeated)
    assert test_matrix.shape[1] == train_matrix.shape[1]
