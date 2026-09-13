"""Carregamento da base bruta e do dataset processado.

O arquivo bruto em ``data/raw`` e tratado como **somente leitura**: o pipeline
nunca escreve nele (D5).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .config import PROCESSED_DATASET_PATH, RAW_DATA_PATH, RAW_FILE_SHA256


class DataNotFoundError(FileNotFoundError):
    """Erro levantado quando a base esperada nao existe no caminho informado."""


def compute_sha256(path: str | Path) -> str:
    """Calcula o SHA-256 de um arquivo em blocos (sem carrega-lo inteiro)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_raw_integrity(
    path: str | Path = RAW_DATA_PATH,
    expected_sha256: str = RAW_FILE_SHA256,
) -> dict:
    """Confere se o arquivo bruto permanece intacto (D4/D5)."""
    file_path = Path(path)
    if not file_path.exists():
        raise DataNotFoundError(f"Arquivo bruto nao encontrado: {file_path}")

    actual = compute_sha256(file_path)
    return {
        "path": str(file_path),
        "expected_sha256": expected_sha256,
        "actual_sha256": actual,
        "ok": actual == expected_sha256,
    }


def load_raw_data(
    path: str | Path = RAW_DATA_PATH,
    sheet_name: int | str = 0,
    verify_integrity: bool = False,
) -> pd.DataFrame:
    """Carrega a base bruta a partir do ``.xlsx``.

    Parameters
    ----------
    path:
        Caminho do arquivo Excel.
    sheet_name:
        Aba a ser lida (padrao: primeira aba).
    verify_integrity:
        Se ``True``, valida o SHA-256 antes de carregar.

    Returns
    -------
    pandas.DataFrame
        Base bruta, sem qualquer transformacao.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise DataNotFoundError(
            f"Base bruta nao encontrada em '{file_path}'. "
            "Confirme se o arquivo esta em data/raw/."
        )

    if verify_integrity:
        integrity = verify_raw_integrity(file_path)
        if not integrity["ok"]:
            raise ValueError(
                "O arquivo bruto foi alterado (SHA-256 divergente). "
                "Restaure a copia original antes de continuar."
            )

    return pd.read_excel(file_path, sheet_name=sheet_name)


def load_processed_data(
    path: str | Path = PROCESSED_DATASET_PATH,
) -> pd.DataFrame:
    """Carrega o dataset processado salvo em ``data/processed``."""
    file_path = Path(path)
    if not file_path.exists():
        raise DataNotFoundError(
            f"Dataset processado nao encontrado em '{file_path}'. "
            "Execute 'python -m src.build_dataset' primeiro."
        )
    return pd.read_csv(file_path)
