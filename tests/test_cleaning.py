"""Testes da etapa de limpeza: correcao de notas, duplicatas e alvo."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.cleaning import (
    correct_grade_columns,
    create_target,
    describe_target,
    remove_exact_duplicates,
    summarize_missing_values,
    validate_correction_by_distribution,
    validate_grade_consistency,
)
from src.config import (
    CLASS_DROPOUT,
    CLASS_ENROLLED,
    CLASS_GRADUATE,
    GRADE_COLUMNS,
    GRADE_CONSISTENCY_COLUMN,
    TARGET_CLASS_COLUMN,
    TARGET_COLUMN,
    TARGET_RAW,
    TARGET_SENSITIVITY_COLUMN,
)


@pytest.fixture()
def grade_df() -> pd.DataFrame:
    """DataFrame minimo com as duas colunas de nota afetadas."""
    return pd.DataFrame(
        {
            "UnidadesCurriculares1SemestreGrau": [
                0.0,
                14.0,
                1.34285714285714e16,
                10375.0,
            ],
            "UnidadesCurriculares2SemestreGrau": [
                12.4,
                0.0,
                1.73333333333333e16,
                13875.0,
            ],
            GRADE_CONSISTENCY_COLUMN: [0, 1, 1, 1],
        }
    )


def test_correct_grade_columns_reports_statistics(grade_df: pd.DataFrame) -> None:
    """Registra quantos valores foram corrigidos e a faixa antes/depois."""
    corrected, stats = correct_grade_columns(grade_df)

    for column in GRADE_COLUMNS:
        assert column in stats
        assert stats[column]["n_values"] == 4
        # cada coluna da fixture tem 2 valores corrompidos:
        # um por potencia grande (~1e16) e um por fator 1000 (10375 / 13875)
        assert stats[column]["n_corrected"] == 2

    assert stats[GRADE_COLUMNS[0]]["max_before"] == pytest.approx(
        1.34285714285714e16
    )
    assert stats[GRADE_COLUMNS[1]]["max_before"] == pytest.approx(
        1.73333333333333e16
    )


def test_correct_grade_columns_within_scale(grade_df: pd.DataFrame) -> None:
    """Todos os valores corrigidos ficam na faixa 0-20."""
    corrected, _ = correct_grade_columns(grade_df)

    for column in GRADE_COLUMNS:
        assert corrected[column].max() <= 20.0
        assert corrected[column].min() >= 0.0


def test_correct_grade_columns_keeps_representative_examples(
    grade_df: pd.DataFrame,
) -> None:
    """Guarda exemplos 'antes -> depois' para o relatorio (D4)."""
    _, stats = correct_grade_columns(grade_df)

    examples = stats[GRADE_COLUMNS[0]]["examples"]
    assert examples
    assert examples[0]["before"] == pytest.approx(1.34285714285714e16)
    assert examples[0]["after"] == pytest.approx(13.4285714285714, rel=1e-9)


def test_correct_grade_columns_does_not_mutate_input(grade_df: pd.DataFrame) -> None:
    """A funcao devolve copia e preserva o DataFrame original."""
    original = grade_df.copy(deep=True)
    correct_grade_columns(grade_df)

    pd.testing.assert_frame_equal(grade_df, original)


def test_correct_grade_columns_normalizes_text(
    grade_df: pd.DataFrame,
) -> None:
    """Valores textuais tambem sao corrigidos."""
    df = pd.DataFrame({GRADE_COLUMNS[0]: ["13875", "14", ""]})
    corrected, stats = correct_grade_columns(df, columns=(GRADE_COLUMNS[0],))

    assert corrected[GRADE_COLUMNS[0]].iloc[0] == pytest.approx(13.875)
    assert corrected[GRADE_COLUMNS[0]].iloc[1] == pytest.approx(14.0)
    assert stats[GRADE_COLUMNS[0]]["n_corrected"] == 1


def test_validate_grade_consistency_detects_violation() -> None:
    """Sinaliza linhas com Aprovado == 0 e Grau > 0."""
    df = pd.DataFrame(
        {
            GRADE_COLUMNS[0]: [0.0, 13.5],
            GRADE_CONSISTENCY_COLUMN: [0, 0],
        }
    )
    result = validate_grade_consistency(df, grade_column=GRADE_COLUMNS[0])

    assert result["ok"] is False
    assert result["n_violations"] == 1


def test_validate_grade_consistency_ok() -> None:
    """Sem violacoes quando todos os aprovados sao coerentes com a nota."""
    df = pd.DataFrame(
        {
            GRADE_COLUMNS[0]: [0.0, 13.5],
            GRADE_CONSISTENCY_COLUMN: [0, 1],
        }
    )
    result = validate_grade_consistency(df, grade_column=GRADE_COLUMNS[0])

    assert result["ok"] is True
    assert result["n_violations"] == 0


def test_validate_grade_consistency_pairs_same_semester() -> None:
    """Cada nota e comparada com o `Aprovado` do MESMO semestre.

    Regressao: usar o aprovado do 1o semestre para validar a nota do 2o gerava
    falsas violacoes, pois um aluno pode nao ter aprovacao no 1o semestre e
    ainda assim ter nota no 2o.
    """
    df = pd.DataFrame(
        {
            "UnidadesCurriculares1SemestreGrau": [0.0],
            "UnidadesCurriculares1SemestreAprovado": [0],
            "UnidadesCurriculares2SemestreGrau": [12.0],
            "UnidadesCurriculares2SemestreAprovado": [5],
        }
    )
    result = validate_grade_consistency(
        df, grade_column="UnidadesCurriculares2SemestreGrau"
    )

    assert result["approved_column"] == "UnidadesCurriculares2SemestreAprovado"
    assert result["n_violations"] == 0
    assert result["ok"] is True


def test_validate_grade_consistency_accepts_explicit_approved_column() -> None:
    """O pareamento pode ser sobrescrito explicitamente."""
    df = pd.DataFrame(
        {
            GRADE_COLUMNS[0]: [13.5],
            GRADE_CONSISTENCY_COLUMN: [0],
        }
    )
    result = validate_grade_consistency(
        df, grade_column=GRADE_COLUMNS[0], approved_column=GRADE_CONSISTENCY_COLUMN
    )

    assert result["n_violations"] == 1
    assert result["ok"] is False


def test_validate_correction_by_distribution_reports_groups() -> None:
    """Compara valores nunca corrompidos com valores corrigidos."""
    df = pd.DataFrame(
        {
            GRADE_COLUMNS[0]: [
                10.0,
                12.0,
                13875.0,
                1.34285714285714e16,
            ]
        }
    )
    result = validate_correction_by_distribution(df, GRADE_COLUMNS[0])

    assert result["untouched"]["n"] == 2
    assert result["corrected"]["n"] == 2
    assert result["n_corrupted"] == 2
    assert result["n_corrupted_integers"] == 2
    assert result["pct_corrupted_integers"] == 100.0
    assert result["all_corrected_within_scale"] is True
    assert result["ok"] is True


def test_validate_correction_by_distribution_missing_column() -> None:
    """Coluna ausente e sinalizada sem levantar excecao."""
    result = validate_correction_by_distribution(
        pd.DataFrame({"outra": [1]}), GRADE_COLUMNS[0]
    )

    assert result["ok"] is False
    assert "ausente" in result["detail"]


def test_remove_exact_duplicates_removes_one_row() -> None:
    """Remove apenas as repeticoes, preservando a primeira ocorrencia (D9)."""
    df = pd.DataFrame({"a": [1, 2, 1], "b": ["x", "y", "x"]})
    deduplicated, report = remove_exact_duplicates(df)

    assert report["n_removed"] == 1
    assert report["n_duplicate_groups"] == 1
    assert report["n_rows_in_duplicate_groups"] == 2
    assert report["n_rows_before"] == 3
    assert report["n_rows_after"] == 2
    assert deduplicated.shape[0] == 2


def test_remove_exact_duplicates_no_duplicates() -> None:
    """Nao remove nada quando todas as linhas sao distintas."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    deduplicated, report = remove_exact_duplicates(df)

    assert report["n_removed"] == 0
    assert deduplicated.shape[0] == 3


def test_create_target_principal_and_sensitivity() -> None:
    """Cria as duas binarizacoes definidas em D1."""
    df = pd.DataFrame(
        {TARGET_RAW: [CLASS_DROPOUT, CLASS_GRADUATE, CLASS_ENROLLED, CLASS_DROPOUT]}
    )
    targeted = create_target(df)

    assert targeted[TARGET_CLASS_COLUMN].tolist() == [
        CLASS_DROPOUT,
        CLASS_GRADUATE,
        CLASS_ENROLLED,
        CLASS_DROPOUT,
    ]

    # alvo principal: Desistente = 1; Graduado e Matriculado = 0
    assert targeted[TARGET_COLUMN].tolist() == [1, 0, 0, 1]

    # sensibilidade: Matriculado fica ausente
    sensitivity = targeted[TARGET_SENSITIVITY_COLUMN]
    assert int(sensitivity.iloc[0]) == 1
    assert int(sensitivity.iloc[1]) == 0
    assert pd.isna(sensitivity.iloc[2])
    assert int(sensitivity.iloc[3]) == 1


def test_create_target_does_not_mutate_input() -> None:
    """O alvo e adicionado em uma copia."""
    df = pd.DataFrame({TARGET_RAW: [CLASS_DROPOUT]})
    create_target(df)
    assert TARGET_COLUMN not in df.columns


def test_create_target_raises_when_column_missing() -> None:
    """Falha com mensagem clara quando a coluna-alvo nao existe."""
    with pytest.raises(KeyError):
        create_target(pd.DataFrame({"outra": [1]}))


def test_describe_target_reports_both_definitions() -> None:
    """Resume classes originais, alvo principal e sensibilidade."""
    df = pd.DataFrame(
        {
            TARGET_RAW: [CLASS_DROPOUT] * 2
            + [CLASS_GRADUATE] * 3
            + [CLASS_ENROLLED] * 5
        }
    )
    description = describe_target(create_target(df))

    assert description["classes_originais"][CLASS_DROPOUT] == 2
    assert description["target_principal"]["desistente_1"] == 2
    assert description["target_principal"]["outros_0"] == 8
    assert description["target_principal"]["proporcao_positivos"] == 0.2
    assert description["target_sensibilidade"]["matriculado_excluido"] == 5
    assert description["target_sensibilidade"]["proporcao_positivos"] == 0.4


def test_summarize_missing_values() -> None:
    """Conta apenas colunas com ausencias."""
    df = pd.DataFrame({"a": [1, np.nan, 3], "b": ["x", "y", "z"]})
    missing = summarize_missing_values(df)

    assert missing == {"a": 1}
