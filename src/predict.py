"""API de inferencia: carrega o modelo serializado e estima o risco de evasao.

Esta e a **unica porta de entrada** para previsoes, usada tanto pela aplicacao
Streamlit quanto pelos testes. Assim nao existe logica duplicada entre treino e
producao.

O artefato salvo e o **Pipeline completo** (feature engineering +
pre-processamento + modelo), portanto a aplicacao pode receber os dados crus do
formulario sem reimplementar nenhuma transformacao.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config

#: Faixas de risco usadas para comunicar o resultado ao usuario.
RISK_BANDS: tuple[tuple[float, str, str], ...] = (
    (0.00, "Baixo", "Manter acompanhamento padrao."),
    (0.35, "Moderado", "Vale acompanhar de perto e oferecer apoio."),
    (0.65, "Alto", "Priorizar contato e acao de retencao."),
)


class ModelNotFoundError(FileNotFoundError):
    """Erro levantado quando o artefato do modelo nao existe."""


def load_model(path: str | Path | None = None):
    """Carrega o pipeline serializado em ``models/model.joblib``."""
    file_path = Path(path or config.MODEL_PATH)
    if not file_path.exists():
        raise ModelNotFoundError(
            f"Modelo nao encontrado em '{file_path}'. "
            "Execute 'python -m src.select_model' para gera-lo."
        )
    return joblib.load(file_path)


def load_metadata(path: str | Path | None = None) -> dict:
    """Carrega os metadados do modelo (limiar, features, metricas)."""
    file_path = Path(path or config.MODEL_METADATA_PATH)
    if not file_path.exists():
        return {}
    return json.loads(file_path.read_text(encoding="utf-8"))


def classify_risk(probability: float) -> tuple[str, str]:
    """Traduz a probabilidade em faixa textual + recomendacao de acao.

    A saida e **orientacao de risco**, nao uma decisao automatica sobre o
    estudante -- o que o app deixa explicito ao usuario (D10).
    """
    label, advice = RISK_BANDS[0][1], RISK_BANDS[0][2]
    for minimum, band_label, band_advice in RISK_BANDS:
        if probability >= minimum:
            label, advice = band_label, band_advice
    return label, advice


def _to_frame(
    payload: dict | pd.DataFrame,
    expected_columns: list[str] | None,
) -> pd.DataFrame:
    """Normaliza a entrada (dict ou DataFrame) para uma linha valida.

    Colunas ausentes sao criadas com ``NaN`` e tratadas pela imputacao do
    proprio pipeline -- o que mantem o comportamento igual ao do treino.
    """
    if isinstance(payload, pd.DataFrame):
        frame = payload.copy()
    else:
        frame = pd.DataFrame([payload])

    if expected_columns:
        missing = [column for column in expected_columns if column not in frame]
        for column in missing:
            # ``np.nan`` (e nao ``pd.NA``) para preservar o dtype numerico e
            # permitir que a imputacao do pipeline trate o campo ausente.
            frame[column] = np.nan
        frame = frame[expected_columns]
    return frame


def predict_risk(
    payload: dict | pd.DataFrame,
    model=None,
    metadata: dict | None = None,
) -> dict:
    """Estima o risco de evasao para um ou mais estudantes.

    Parameters
    ----------
    payload:
        Dicionario (um estudante) ou DataFrame (varios) com os campos do
        formulario, nos valores originais (sem tratamento).
    model:
        Pipeline ja carregado (opcional; evita recarregar em lote).
    metadata:
        Metadados do modelo (opcional). Se ausente, e carregado do disco.

    Returns
    -------
    dict
        Probabilidades, limiar aplicado, classificacao binaria e faixa de risco.
    """
    metadata = load_metadata() if metadata is None else metadata
    model = load_model() if model is None else model

    expected_columns = metadata.get("input_columns")
    threshold = float(metadata.get("decision_threshold", 0.5))

    frame = _to_frame(payload, expected_columns)
    probabilities = model.predict_proba(frame)[:, 1]

    results = []
    for probability in probabilities:
        probability = float(probability)
        label, advice = classify_risk(probability)
        results.append(
            {
                "probability": probability,
                "threshold": threshold,
                "is_dropout_risk": bool(probability >= threshold),
                "risk_band": label,
                "recommendation": advice,
            }
        )

    single = not isinstance(payload, pd.DataFrame) and len(results) == 1
    return results[0] if single else {"results": results}
