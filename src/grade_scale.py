"""Nucleo numerico puro da correcao de escala das colunas ``...SemestreGrau``.

Contexto (D4)
-------------
As colunas ``UnidadesCurriculares1SemestreGrau`` e
``UnidadesCurriculares2SemestreGrau`` contem valores muito acima da escala
portuguesa de avaliacao (0 a 20), chegando a ~1,7e16.

O padrao observado e **consistente com a perda do separador decimal**: o valor
original foi multiplicado por uma potencia de 10 e armazenado como numero
grande. Exemplos verificados nos dados:

    13.4285714285714  ->  1.34285714285714e+16
    10.375            ->  10375
    12.48375          ->  1248375

Nao afirmamos que essa seja a **causa comprovada** - apenas que os dados sao
consistentes com esse mecanismo, e que a correcao abaixo recupera 100% dos
valores para a faixa [0, 20].

Este modulo foi mantido **sem dependencias de terceiros** de proposito, para
que o nucleo numerico possa ser testado de forma isolada e rapida.
"""

from __future__ import annotations

import math

from .config import GRADE_MAX_VALID, GRADE_SCALE_BASE


def to_float(value: object) -> float | None:
    """Converte ``value`` para ``float`` de forma tolerante.

    Retorna ``None`` para ``None``, string vazia, valores nao numericos e
    ``NaN``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def needs_correction(
    value: object,
    max_valid: float = GRADE_MAX_VALID,
) -> bool:
    """Indica se ``value`` esta fora da escala valida de notas."""
    number = to_float(value)
    if number is None:
        return False
    return abs(number) > max_valid


def correct_grade_value(
    value: object,
    max_valid: float = GRADE_MAX_VALID,
    base: float = GRADE_SCALE_BASE,
) -> float:
    """Recupera uma nota da escala 0-20 a partir de um valor corrompido.

    Divide sucessivamente por ``base`` (10) enquanto o valor absoluto
    permanecer acima de ``max_valid``. Valores ja validos sao devolvidos
    inalterados. Valores ausentes/invalidos viram ``NaN``.

    Parameters
    ----------
    value:
        Valor bruto (pode ser ``str``, ``int``, ``float`` ou ``None``).
    max_valid:
        Limite superior da escala valida (padrao: 20.0).
    base:
        Base do fator de escala aplicado a cada passo (padrao: 10).

    Returns
    -------
    float
        Nota na faixa ``[0, max_valid]`` ou ``NaN`` quando o valor e invalido.
    """
    number = to_float(value)
    if number is None:
        return math.nan

    result = number
    # Guarda de seguranca contra loop infinito em valores exotcos.
    for _ in range(64):
        if abs(result) <= max_valid:
            break
        result /= base
    return result


def corrected_values_match_scale(
    values: object,
    max_valid: float = GRADE_MAX_VALID,
) -> bool:
    """Verifica se todos os valores corrigidos respeitam a escala valida."""
    for value in values:  # type: ignore[union-attr]
        number = to_float(value)
        if number is None:
            continue
        if not (0.0 <= number <= max_valid):
            return False
    return True
