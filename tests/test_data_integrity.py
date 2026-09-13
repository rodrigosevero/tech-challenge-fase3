"""Teste de integridade da base bruta (D4/D5).

Garante que o arquivo em ``data/raw`` nunca seja alterado pelo pipeline.
"""

from __future__ import annotations

import pytest

from src.config import LEGACY_RAW_PATH, RAW_DATA_PATH, RAW_FILE_SHA256
from src.data_loader import load_raw_data, verify_raw_integrity
from src.schema import validate_schema

pytestmark = pytest.mark.skipif(
    not RAW_DATA_PATH.exists(),
    reason="Base bruta ausente em data/raw - teste de integridade ignorado.",
)


def test_raw_file_sha256_matches_expected() -> None:
    """A copia de trabalho tem o hash registrado (arquivo intacto)."""
    result = verify_raw_integrity()

    assert result["ok"] is True, (
        "O arquivo bruto foi modificado. "
        f"Esperado {RAW_FILE_SHA256}, obtido {result['actual_sha256']}."
    )


def test_original_delivered_file_was_preserved() -> None:
    """O arquivo entregue originalmente continua no lugar."""
    assert LEGACY_RAW_PATH.exists(), (
        f"O arquivo original '{LEGACY_RAW_PATH}' nao foi preservado."
    )


def test_raw_data_matches_expected_shape() -> None:
    """A base carregada corresponde ao schema e ao volume esperados."""
    df = load_raw_data()
    report = validate_schema(df)

    assert report.ok is True
    assert report.n_rows == 4424
    assert report.n_cols == 28
