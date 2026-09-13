"""Testes da validacao de schema da base bruta."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import CLASS_DROPOUT, CLASS_GRADUATE, TARGET_RAW
from src.schema import EXPECTED_COLUMNS, SchemaError, validate_schema


def _valid_dataframe() -> pd.DataFrame:
    """DataFrame com exatamente as colunas esperadas."""
    data: dict[str, list[object]] = {
        column: ["valor"] for column in EXPECTED_COLUMNS
    }
    data[TARGET_RAW] = [CLASS_DROPOUT, CLASS_GRADUATE]
    for column in EXPECTED_COLUMNS:
        if column != TARGET_RAW:
            data[column] = data[column] * 2
    return pd.DataFrame(data)


def test_expected_columns_has_28_entries() -> None:
    """A base possui 28 colunas conhecidas."""
    assert len(EXPECTED_COLUMNS) == 28


def test_validate_schema_ok() -> None:
    """Schema valido quando colunas e classes conferem."""
    report = validate_schema(_valid_dataframe())

    assert report.ok is True
    assert report.n_rows == 2
    assert report.n_cols == 28
    assert report.missing_columns == []
    assert report.unexpected_columns == []


def test_validate_schema_detects_missing_column() -> None:
    """Coluna obrigatoria ausente levanta SchemaError."""
    df = _valid_dataframe().drop(columns=["Curso"])

    with pytest.raises(SchemaError, match="Curso"):
        validate_schema(df)

    report = validate_schema(df, raise_on_missing=False)
    assert report.ok is False
    assert "Curso" in report.missing_columns


def test_validate_schema_flags_unexpected_column() -> None:
    """Coluna extra e sinalizada no relatorio."""
    df = _valid_dataframe()
    df["coluna_extra"] = 1

    report = validate_schema(df, raise_on_missing=False)
    assert "coluna_extra" in report.unexpected_columns


def test_validate_schema_detects_unexpected_target_class() -> None:
    """Classe desconhecida na coluna-alvo levanta erro."""
    df = _valid_dataframe()
    df.loc[0, TARGET_RAW] = "ClasseNova"

    with pytest.raises(SchemaError, match="ClasseNova"):
        validate_schema(df)


def test_schema_report_serializes() -> None:
    """O relatorio pode ser convertido em dicionario."""
    report = validate_schema(_valid_dataframe())
    payload = report.as_dict()

    assert payload["ok"] is True
    assert isinstance(payload["dtypes"], dict)
