"""Testes do nucleo numerico de correcao de escala (D4).

Este arquivo nao depende de pandas nem de scikit-learn, por isso roda de forma
isolada e rapida.
"""

from __future__ import annotations

import math

import pytest

from src.grade_scale import (
    correct_grade_value,
    corrected_values_match_scale,
    needs_correction,
    to_float,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # valores ja validos permanecem inalterados
        (0, 0.0),
        (0.0, 0.0),
        (10.375, 10.375),
        (14.0, 14.0),
        (20.0, 20.0),
        # valores corrompidos por multiplo de 10
        (10375.0, 10.375),
        (13875.0, 13.875),
        (14345.0, 14.345),
        (12288.0, 12.288),
        (1248375.0, 12.48375),
        # valores corrompidos por potencia grande (caso mais frequente)
        (1.34285714285714e16, 13.4285714285714),
        (1.73333333333333e16, 17.3333333333333),
        # entradas textuais
        ("14.0", 14.0),
        ("13875", 13.875),
    ],
)
def test_correct_grade_value(value: object, expected: float) -> None:
    """Recupera a nota original na escala 0-20."""
    assert correct_grade_value(value) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize(
    "value",
    [None, "", "   ", "abc", float("nan")],
)
def test_correct_grade_value_handles_invalid_input(value: object) -> None:
    """Valores ausentes ou invalidos viram NaN, sem levantar excecao."""
    assert math.isnan(correct_grade_value(value))


def test_correction_is_idempotent() -> None:
    """Aplicar a correcao duas vezes nao altera o resultado."""
    once = correct_grade_value(1.34285714285714e16)
    twice = correct_grade_value(once)
    assert twice == pytest.approx(once, rel=1e-12)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, False),
        (20.0, False),
        (20.5, True),
        (10375.0, True),
        (1.34285714285714e16, True),
        (None, False),
        (float("nan"), False),
    ],
)
def test_needs_correction(value: object, expected: bool) -> None:
    """Detecta corretamente quais valores exigem correcao."""
    assert needs_correction(value) is expected


@pytest.mark.parametrize("value", [0.0, 5.5, 10.375, 18.875, 20.0])
def test_values_stay_within_scale(value: float) -> None:
    """Nenhum valor corrigido ultrapassa a escala valida."""
    corrected = correct_grade_value(value)
    assert 0.0 <= corrected <= 20.0


def test_all_corrupted_values_land_in_scale() -> None:
    """Lote representativo de valores corrompidos cai em [0, 20]."""
    corrupted = [
        10375.0,
        13875.0,
        12288.0,
        1248375.0,
        1.34285714285714e16,
        1.73333333333333e16,
        1.68857142857142e16,
        0.0,
        14.0,
    ]
    corrected = [correct_grade_value(value) for value in corrupted]
    assert corrected_values_match_scale(corrected)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (10.5, 10.5),
        (" 7.25 ", 7.25),
        ("0", 0.0),
        (None, None),
        ("", None),
        ("abc", None),
    ],
)
def test_to_float(value: object, expected: float | None) -> None:
    """Conversao tolerante para float."""
    result = to_float(value)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


def test_corrected_values_match_scale_detects_out_of_range() -> None:
    """A validacao de faixa sinaliza valores fora da escala."""
    assert corrected_values_match_scale([0.0, 20.0, 13.5]) is True
    assert corrected_values_match_scale([0.0, 25.0, 13.5]) is False
    assert corrected_values_match_scale([-1.0, 13.5]) is False
