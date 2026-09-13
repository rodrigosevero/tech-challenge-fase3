"""Pre-processamento: ``ColumnTransformer`` + ``Pipeline``.

Objetivo central (requisito do trabalho): **evitar vazamento entre treino e
teste**. Toda transformacao que "aprende" parametros (mediana da imputacao,
media/desvio da padronizacao, categorias do one-hot) fica encapsulada em
``Pipeline``/``ColumnTransformer`` e e ajustada **somente** com os dados de
treino de cada fold da validacao cruzada.
"""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import BINARY_FEATURES, CATEGORICAL_FEATURES
from .features import FeatureEngineer

#: Categorias com menos ocorrencias que este limiar sao agrupadas em uma unica
#: categoria "infrequente". Resolve a cauda longa de ``Nacionalidade``,
#: ``QualificacaoAnterior`` e ``EstadoCivil``.
MIN_FREQUENCY: int = 10


def split_feature_groups(
    columns: Sequence[str],
) -> tuple[list[str], list[str], list[str]]:
    """Separa as colunas do cenario em (numericas, binarias, categoricas).

    As binarias (flags 0/1) nao passam por padronizacao, para permanecerem
    interpretaveis. As derivadas (taxas, deltas) sao tratadas como numericas.
    """
    categorical = [column for column in columns if column in CATEGORICAL_FEATURES]
    binary = [column for column in columns if column in BINARY_FEATURES]
    numeric = [
        column
        for column in columns
        if column not in CATEGORICAL_FEATURES and column not in BINARY_FEATURES
    ]
    return numeric, binary, categorical


def build_preprocessor(
    columns: Sequence[str],
    min_frequency: int = MIN_FREQUENCY,
) -> ColumnTransformer:
    """Monta o ``ColumnTransformer`` adequado ao conjunto de colunas do cenario.

    Parameters
    ----------
    columns:
        Colunas de entrada do cenario (ja apos o feature engineering).
    min_frequency:
        Limiar para agrupar categorias raras no one-hot encoding.

    Returns
    -------
    sklearn.compose.ColumnTransformer
    """
    numeric, binary, categorical = split_feature_groups(columns)
    transformers: list[tuple[str, object, list[str]]] = []

    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )

    if binary:
        transformers.append(
            ("bin", SimpleImputer(strategy="most_frequent"), binary)
        )

    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=min_frequency,
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor


def build_pipeline(
    columns: Sequence[str],
    model,
    min_frequency: int = MIN_FREQUENCY,
) -> Pipeline:
    """Pipeline completo: feature engineering -> pre-processamento -> modelo.

    Parameters
    ----------
    columns:
        Colunas do cenario **antes** do feature engineering (as derivadas sao
        criadas pela propria pipeline).
    model:
        Classificador scikit-learn.
    min_frequency:
        Repassado para :func:`build_preprocessor`.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline pronto para ``fit``/``predict``. Como o modelo final e
        serializado junto, a aplicacao nao precisa reimplementar nenhuma etapa.
    """
    return Pipeline(
        steps=[
            ("engineer", FeatureEngineer()),
            ("preprocessor", build_preprocessor(columns, min_frequency=min_frequency)),
            ("model", model),
        ]
    )
