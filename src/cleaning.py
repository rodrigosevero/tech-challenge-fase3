"""Limpieza controlada da base: correcao de notas, duplicatas e alvo.

Todas as funcoes deste modulo sao **puras com relacao a entrada**: recebem um
``DataFrame`` e devolvem uma copia transformada, sem nunca alterar o objeto
original nem o arquivo bruto em disco (D4/D5).
"""

from __future__ import annotations

import pandas as pd

from .config import (
    CLASS_DROPOUT,
    CLASS_GRADUATE,
    GRADE_APPROVED_COLUMNS,
    GRADE_COLUMNS,
    GRADE_CONSISTENCY_COLUMN,
    GRADE_MAX_VALID,
    TARGET_CLASS_COLUMN,
    TARGET_COLUMN,
    TARGET_RAW,
    TARGET_SENSITIVITY_COLUMN,
)
from .grade_scale import correct_grade_value, needs_correction

# --- valores usados na binarizacao da analise de sensibilidade (D1) --------- #
_SENSITIVITY_MAPPING: dict[str, int] = {
    CLASS_DROPOUT: 1,
    CLASS_GRADUATE: 0,
}


def summarize_missing_values(df: pd.DataFrame) -> dict[str, int]:
    """Retorna a contagem de valores ausentes por coluna (apenas > 0)."""
    missing = df.isna().sum()
    return {
        str(column): int(count)
        for column, count in missing.items()
        if int(count) > 0
    }


def correct_grade_columns(
    df: pd.DataFrame,
    columns: tuple[str, ...] = GRADE_COLUMNS,
    max_valid: float = GRADE_MAX_VALID,
    n_examples: int = 5,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Corrige a escala das colunas de nota ``...SemestreGrau`` (D4).

    Parameters
    ----------
    df:
        Base bruta.
    columns:
        Colunas a corrigir.
    max_valid:
        Limite superior da escala valida (20.0).
    n_examples:
        Quantos exemplos ``antes -> depois`` registrar por coluna.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        Copia corrigida e estatisticas detalhadas por coluna.
    """
    corrected_df = df.copy()
    stats: dict[str, dict] = {}

    for column in columns:
        if column not in corrected_df.columns:
            continue

        original = pd.to_numeric(corrected_df[column], errors="coerce")
        corrected = original.map(
            lambda value: correct_grade_value(value, max_valid=max_valid)
        )
        mask = original.map(
            lambda value: needs_correction(value, max_valid=max_valid)
        )

        n_values = int(original.notna().sum())
        n_corrected = int(mask.sum())

        examples = [
            {"before": float(before), "after": float(after)}
            for before, after in zip(
                original[mask].head(n_examples),
                corrected[mask].head(n_examples),
            )
        ]

        stats[column] = {
            "n_values": n_values,
            "n_corrected": n_corrected,
            "pct_corrected": round(
                100.0 * n_corrected / n_values if n_values else 0.0, 3
            ),
            "min_before": float(original.min()) if n_values else None,
            "max_before": float(original.max()) if n_values else None,
            "min_after": float(corrected.min()) if n_values else None,
            "max_after": float(corrected.max()) if n_values else None,
            "examples": examples,
        }

        corrected_df[column] = corrected

    return corrected_df, stats


def validate_grade_consistency(
    df: pd.DataFrame,
    grade_column: str = GRADE_COLUMNS[0],
    approved_column: str | None = None,
) -> dict:
    """Teste de consistencia: ``Aprovado == 0`` deve implicar ``Grau == 0``.

    O pareamento e feito **por semestre** (``GRADE_APPROVED_COLUMNS``): a nota do
    1o semestre e comparada com o aprovado do 1o semestre, e a do 2o com o do 2o.
    Comparar semestres diferentes produziria falsas violacoes, pois um aluno pode
    nao ter nenhuma aprovacao no 1o semestre e ainda assim ter nota no 2o.
    """
    if approved_column is None:
        approved_column = GRADE_APPROVED_COLUMNS.get(
            grade_column, GRADE_CONSISTENCY_COLUMN
        )

    if grade_column not in df.columns or approved_column not in df.columns:
        return {
            "grade_column": grade_column,
            "approved_column": approved_column,
            "ok": False,
            "n_violations": None,
            "detail": "colunas ausentes",
        }

    violations = df[approved_column].eq(0) & df[grade_column].gt(0)
    n_violations = int(violations.sum())

    return {
        "grade_column": grade_column,
        "approved_column": approved_column,
        "ok": n_violations == 0,
        "n_violations": n_violations,
        "detail": "linhas com Aprovado == 0 e Grau > 0",
    }


def validate_correction_by_distribution(
    df: pd.DataFrame,
    column: str,
    max_valid: float = GRADE_MAX_VALID,
) -> dict:
    """Valida a correcao comparando distribuicoes (evidencia principal - D4).

    Compara a distribuicao dos valores **nunca corrompidos** com a dos valores
    **corrigidos**. Se a correcao estiver certa, as duas distribuicoes devem ser
    compativeis (mesma faixa plausivel de notas).

    Tambem registra quantos valores corrompidos sao inteiros exatos, o que e
    consistente com a hipotese de remocao do separador decimal.
    """
    if column not in df.columns:
        return {"column": column, "ok": False, "detail": "coluna ausente"}

    raw = pd.to_numeric(df[column], errors="coerce")
    corrupted_mask = raw.map(
        lambda value: needs_correction(value, max_valid=max_valid)
    )
    corrected = raw.map(
        lambda value: correct_grade_value(value, max_valid=max_valid)
    )

    untouched = raw.loc[(~corrupted_mask) & raw.gt(0)]
    fixed = corrected.loc[corrupted_mask]
    corrupted_raw = raw.loc[corrupted_mask]

    def _summary(series: pd.Series) -> dict:
        if series.empty:
            return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
        return {
            "n": int(series.shape[0]),
            "mean": round(float(series.mean()), 4),
            "median": round(float(series.median()), 4),
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
        }

    n_corrupted = int(corrupted_raw.shape[0])
    n_integer = int(
        corrupted_raw.map(lambda value: float(value).is_integer()).sum()
    )

    return {
        "column": column,
        "untouched": _summary(untouched),
        "corrected": _summary(fixed),
        "mean_difference": (
            round(float(fixed.mean() - untouched.mean()), 4)
            if not fixed.empty and not untouched.empty
            else None
        ),
        "all_corrected_within_scale": bool(
            fixed.between(0.0, max_valid).all() if not fixed.empty else True
        ),
        "n_corrupted_integers": n_integer,
        "n_corrupted": n_corrupted,
        "pct_corrupted_integers": (
            round(100.0 * n_integer / n_corrupted, 3) if n_corrupted else None
        ),
        "ok": bool(
            not fixed.empty
            and fixed.between(0.0, max_valid).all()
            and n_integer == n_corrupted
        ),
    }


def remove_exact_duplicates(
    df: pd.DataFrame,
    subset: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Remove linhas duplicadas exatas, preservando a primeira ocorrencia (D9).

    Deve ser chamada **apos** a separacao conceitual entre bruto e processado,
    ou seja, no estagio de construcao do dataset processado.
    """
    duplicated_mask = df.duplicated(subset=subset, keep=False)
    n_rows_in_groups = int(duplicated_mask.sum())

    drop_mask = df.duplicated(subset=subset, keep="first")
    n_removed = int(drop_mask.sum())

    key_columns = list(df.columns) if subset is None else subset
    n_groups = int(
        df.loc[duplicated_mask, key_columns].drop_duplicates().shape[0]
    )

    deduplicated = df.loc[~drop_mask].reset_index(drop=True)

    report = {
        "n_rows_before": int(df.shape[0]),
        "n_rows_after": int(deduplicated.shape[0]),
        "n_rows_in_duplicate_groups": n_rows_in_groups,
        "n_duplicate_groups": n_groups,
        "n_removed": n_removed,
    }
    return deduplicated, report


def create_target(
    df: pd.DataFrame,
    raw_target: str = TARGET_RAW,
) -> pd.DataFrame:
    """Cria as colunas de alvo binario a partir da coluna ``Target`` (D1).

    Gera tres colunas:

    * ``target_class``: rotulo original de 3 classes (rastreabilidade);
    * ``target``: binarizacao **principal** -> ``Desistente = 1``;
      ``Graduado`` e ``Matriculado`` = 0;
    * ``target_excl_enrolled``: binarizacao da **analise de sensibilidade** ->
      ``Desistente = 1``, ``Graduado = 0`` e ``Matriculado`` = ausente (NA).
    """
    if raw_target not in df.columns:
        raise KeyError(f"Coluna-alvo '{raw_target}' nao encontrada na base.")

    targeted = df.copy()
    targeted[TARGET_CLASS_COLUMN] = targeted[raw_target].astype("string")

    # D1 - alvo principal: Desistente vs. (Graduado + Matriculado)
    targeted[TARGET_COLUMN] = (
        targeted[TARGET_CLASS_COLUMN].eq(CLASS_DROPOUT).astype("int64")
    )

    # D1 - sensibilidade: Desistente vs. Graduado (Matriculado excluido)
    sensitivity = targeted[TARGET_CLASS_COLUMN].map(_SENSITIVITY_MAPPING)
    targeted[TARGET_SENSITIVITY_COLUMN] = sensitivity.astype("Int64")

    return targeted


def describe_target(df: pd.DataFrame) -> dict:
    """Resume a distribuicao do alvo nas duas definicoes de binarizacao (D1)."""
    description: dict = {}

    if TARGET_CLASS_COLUMN in df.columns:
        counts = df[TARGET_CLASS_COLUMN].value_counts(dropna=False)
        description["classes_originais"] = {
            str(key): int(value) for key, value in counts.items()
        }

    if TARGET_COLUMN in df.columns:
        counts = df[TARGET_COLUMN].value_counts(dropna=False)
        total = int(df[TARGET_COLUMN].notna().sum())
        positives = int(df[TARGET_COLUMN].eq(1).sum())
        description["target_principal"] = {
            "desistente_1": positives,
            "outros_0": int(df[TARGET_COLUMN].eq(0).sum()),
            "total": total,
            "proporcao_positivos": round(positives / total, 4) if total else None,
        }

    if TARGET_SENSITIVITY_COLUMN in df.columns:
        sensitivity = df[TARGET_SENSITIVITY_COLUMN].dropna()
        total = int(sensitivity.shape[0])
        positives = int(sensitivity.eq(1).sum())
        description["target_sensibilidade"] = {
            "desistente_1": positives,
            "graduado_0": int(sensitivity.eq(0).sum()),
            "matriculado_excluido": int(
                df[TARGET_SENSITIVITY_COLUMN].isna().sum()
            ),
            "total": total,
            "proporcao_positivos": round(positives / total, 4) if total else None,
        }

    return description
