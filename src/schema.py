"""Validacao de schema do arquivo bruto.

Garante que a base carregada corresponde ao que o pipeline espera antes de
qualquer transformacao. Falha cedo e com mensagem clara se algo mudar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import EXPECTED_TARGET_CLASSES, TARGET_RAW

#: As 28 colunas esperadas no arquivo ``StudentsPrepared.xlsx``.
EXPECTED_COLUMNS: tuple[str, ...] = (
    "EstadoCivil",
    "Curso",
    "QualificacaoAnterior",
    "QualificacaoAnteriorGrau",
    "Nacionalidade",
    "NotaAdmissao",
    "NecessidadesEspeciais",
    "Devedor",
    "MensalidadesEmDia",
    "Genero",
    "Bolsista",
    "International",
    "UnidadesCurriculares1SemestreCreditado",
    "UnidadesCurriculares1SemestreInscrito",
    "UnidadesCurriculares1SemestreAvaliacoes",
    "UnidadesCurriculares1SemestreAprovado",
    "UnidadesCurriculares1SemestreGrau",
    "UnidadesCurriculares1SemestreSemAvaliacoes",
    "UnidadesCurriculares2SemestreCreditado",
    "UnidadesCurriculares2SemestreInscrito",
    "UnidadesCurriculares2SemestreAvaliacoes",
    "UnidadesCurriculares2SemestreAprovado",
    "UnidadesCurriculares2SemestreGrau",
    "UnidadesCurriculares2SemestreSemAvaliacoes",
    "TaxaDesemprego",
    "TaxaInflacao",
    "PIB",
    "Target",
)


class SchemaError(ValueError):
    """Erro levantado quando a base nao corresponde ao schema esperado."""


@dataclass
class SchemaReport:
    """Resultado da validacao de schema."""

    ok: bool
    n_rows: int
    n_cols: int
    missing_columns: list[str] = field(default_factory=list)
    unexpected_columns: list[str] = field(default_factory=list)
    target_classes: dict[str, int] = field(default_factory=dict)
    unexpected_target_classes: list[str] = field(default_factory=list)
    dtypes: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        """Serializa o relatorio para dicionario (JSON-friendly)."""
        return {
            "ok": self.ok,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "missing_columns": self.missing_columns,
            "unexpected_columns": self.unexpected_columns,
            "target_classes": self.target_classes,
            "unexpected_target_classes": self.unexpected_target_classes,
            "dtypes": self.dtypes,
        }


def validate_schema(
    df: pd.DataFrame,
    expected_columns: tuple[str, ...] = EXPECTED_COLUMNS,
    raise_on_missing: bool = True,
) -> SchemaReport:
    """Valida colunas, contagem de linhas e classes da variavel-alvo.

    Parameters
    ----------
    df:
        DataFrame bruto recem-carregado.
    expected_columns:
        Colunas obrigatorias.
    raise_on_missing:
        Se ``True``, levanta :class:`SchemaError` quando faltarem colunas.

    Returns
    -------
    SchemaReport
    """
    missing = [column for column in expected_columns if column not in df.columns]
    unexpected = [column for column in df.columns if column not in expected_columns]

    target_classes: dict[str, int] = {}
    unexpected_target_classes: list[str] = []
    if TARGET_RAW in df.columns:
        counts = df[TARGET_RAW].value_counts(dropna=False)
        target_classes = {str(key): int(value) for key, value in counts.items()}
        unexpected_target_classes = [
            label for label in target_classes if label not in EXPECTED_TARGET_CLASSES
        ]

    ok = not missing and not unexpected and not unexpected_target_classes

    report = SchemaReport(
        ok=ok,
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        missing_columns=missing,
        unexpected_columns=unexpected,
        target_classes=target_classes,
        unexpected_target_classes=unexpected_target_classes,
        dtypes={column: str(dtype) for column, dtype in df.dtypes.items()},
    )

    if raise_on_missing and missing:
        raise SchemaError(
            "Colunas obrigatorias ausentes na base: " + ", ".join(missing)
        )
    if raise_on_missing and unexpected_target_classes:
        raise SchemaError(
            "Classes inesperadas em 'Target': "
            + ", ".join(unexpected_target_classes)
        )

    return report
