"""Feature engineering: transformacoes **linha a linha**.

Todas as features criadas aqui dependem apenas da propria linha do estudante.
Por isso **nao introduzem vazamento** entre treino e teste: nao existe nenhuma
estatistica calculada sobre o conjunto (media, mediana, contagem), apenas
aritmetica entre colunas da mesma linha.

Mesmo assim, a classe :class:`FeatureEngineer` permite embutir essa etapa dentro
do ``Pipeline``, garantindo que o artefato serializado aplique exatamente a mesma
transformacao na aplicacao Streamlit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# --- nomes base usados nas formulas ---------------------------------------- #
_ENROLLED_1 = "UnidadesCurriculares1SemestreInscrito"
_APPROVED_1 = "UnidadesCurriculares1SemestreAprovado"
_EVALUATIONS_1 = "UnidadesCurriculares1SemestreAvaliacoes"
_GRADE_1 = "UnidadesCurriculares1SemestreGrau"
_NO_EVAL_1 = "UnidadesCurriculares1SemestreSemAvaliacoes"

_ENROLLED_2 = "UnidadesCurriculares2SemestreInscrito"
_APPROVED_2 = "UnidadesCurriculares2SemestreAprovado"
_GRADE_2 = "UnidadesCurriculares2SemestreGrau"
_NO_EVAL_2 = "UnidadesCurriculares2SemestreSemAvaliacoes"


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divisao protegida contra divisao por zero (retorna 0 nesses casos)."""
    numerator = pd.to_numeric(numerator, errors="coerce").astype(float)
    denominator = pd.to_numeric(denominator, errors="coerce").astype(float)
    safe_denominator = denominator.where(denominator != 0)
    return (numerator / safe_denominator).fillna(0.0)


def _non_approved(approved: pd.Series, enrolled: pd.Series) -> pd.Series:
    """Disciplinas nao aprovadas = inscritas - aprovadas (nunca negativo)."""
    approved = pd.to_numeric(approved, errors="coerce")
    enrolled = pd.to_numeric(enrolled, errors="coerce")
    return (enrolled - approved).clip(lower=0).fillna(0.0)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Devolve uma copia do DataFrame com as features derivadas adicionadas.

    As features so sao criadas quando as colunas de origem existem, o que
    permite usar a mesma funcao nos cenarios *Early Warning* (1o semestre) e
    *Completo* (1o + 2o semestres).
    """
    out = df.copy()
    columns = set(out.columns)

    # --- 1o semestre ------------------------------------------------------- #
    if {_ENROLLED_1, _APPROVED_1} <= columns:
        out["taxa_aprovacao_1sem"] = _safe_ratio(
            out[_APPROVED_1], out[_ENROLLED_1]
        )
        out["disciplinas_nao_aprovadas_1sem"] = _non_approved(
            out[_APPROVED_1], out[_ENROLLED_1]
        )
    if {_ENROLLED_1, _EVALUATIONS_1} <= columns:
        out["razao_avaliacoes_inscrito_1sem"] = _safe_ratio(
            out[_EVALUATIONS_1], out[_ENROLLED_1]
        )

    # --- 2o semestre ------------------------------------------------------- #
    if {_ENROLLED_2, _APPROVED_2} <= columns:
        out["taxa_aprovacao_2sem"] = _safe_ratio(
            out[_APPROVED_2], out[_ENROLLED_2]
        )
        out["disciplinas_nao_aprovadas_2sem"] = _non_approved(
            out[_APPROVED_2], out[_ENROLLED_2]
        )

    # --- evolucao entre semestres ------------------------------------------ #
    if {"taxa_aprovacao_1sem", "taxa_aprovacao_2sem"} <= set(out.columns):
        out["delta_taxa_aprovacao"] = (
            out["taxa_aprovacao_2sem"] - out["taxa_aprovacao_1sem"]
        )
    if {_GRADE_1, _GRADE_2} <= columns:
        grade_1 = pd.to_numeric(out[_GRADE_1], errors="coerce")
        grade_2 = pd.to_numeric(out[_GRADE_2], errors="coerce")
        out["delta_grau"] = (grade_2 - grade_1).fillna(0.0)
    if {_NO_EVAL_1, _NO_EVAL_2} <= columns:
        no_eval_1 = pd.to_numeric(out[_NO_EVAL_1], errors="coerce").fillna(0.0)
        no_eval_2 = pd.to_numeric(out[_NO_EVAL_2], errors="coerce").fillna(0.0)
        out["total_sem_avaliacoes"] = no_eval_1 + no_eval_2

    return out


def engineered_feature_columns(base_columns) -> list[str]:
    """Lista das colunas resultantes apos o feature engineering.

    Usa um DataFrame vazio (0 linhas) para descobrir os nomes sem custo.
    """
    empty = pd.DataFrame(columns=list(base_columns))
    return list(add_engineered_features(empty).columns)


def resolve_scenario_columns(
    available_columns,
    scenario_columns,
) -> list[str]:
    """Mantem apenas as colunas do cenario que existem apos o engineering."""
    available = set(available_columns)
    return [column for column in scenario_columns if column in available]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Transformer *stateless* que aplica :func:`add_engineered_features`.

    Pensado para ser a primeira etapa de um ``Pipeline``, garantindo que o
    artefato salvo contenha tambem a logica de feature engineering.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        self._validate(X)
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate(X)
        return add_engineered_features(X)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        if input_features is None:
            raise ValueError(
                "get_feature_names_out requer input_features ou fit previo."
            )
        return np.asarray(
            engineered_feature_columns(input_features), dtype=object
        )

    @staticmethod
    def _validate(X: object) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "FeatureEngineer espera um pandas.DataFrame como entrada."
            )
