"""Configuracao central do projeto (Tech Challenge - Fase 3, FIAP MLET).

Este modulo e a **unica fonte de verdade** para:

* ``RANDOM_STATE`` (reprodutibilidade);
* caminhos de dados, modelos e relatorios;
* nomes de colunas e agrupamentos de features;
* definicao da variavel-alvo e das duas binarizacoes (D1);
* definicao dos cenarios ``EARLY_WARNING`` e ``FULL`` (D2);
* grupos de ablação: financeiras (D3) e macroeconomicas (D7).

Nenhum outro modulo deve declarar literais de coluna ou caminhos soltos.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Reprodutibilidade
# --------------------------------------------------------------------------- #

#: Semente unica usada em train_test_split, StratifiedKFold, modelos, etc.
RANDOM_STATE: int = 42

# --------------------------------------------------------------------------- #
# Caminhos
# --------------------------------------------------------------------------- #

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
TABLES_DIR: Path = REPORTS_DIR / "tables"
MODELS_DIR: Path = PROJECT_ROOT / "models"

#: Nome do arquivo bruto (mantido intacto, conforme D5).
RAW_FILENAME: str = "StudentsPrepared.xlsx"

#: Copia de trabalho usada pelo pipeline. O original NUNCA e sobrescrito.
RAW_DATA_PATH: Path = DATA_RAW_DIR / RAW_FILENAME

#: Local onde o arquivo foi entregue originalmente (preservado, somente leitura).
LEGACY_RAW_PATH: Path = DATA_DIR / RAW_FILENAME

#: SHA-256 do arquivo bruto, usado no teste de integridade (D4/D5).
RAW_FILE_SHA256: str = (
    "10687fd56d075289d7cc112592d977d56c3c88df90d6a2eb6deeb7f6be1206ba"
)

PROCESSED_DATASET_PATH: Path = DATA_PROCESSED_DIR / "students_processed.csv"
QUALITY_REPORT_JSON: Path = REPORTS_DIR / "data_quality_report.json"
QUALITY_REPORT_MD: Path = REPORTS_DIR / "data_quality_report.md"

#: Indices da divisao treino/teste, congelados para reuso na Fase 3.
SPLIT_INDICES_PATH: Path = DATA_PROCESSED_DIR / "train_test_split.json"

#: Saidas da Fase 2 (comparacao de modelos e cenarios).
CV_COMPARISON_CSV: Path = TABLES_DIR / "phase2_cv_comparison.csv"
TUNED_COMPARISON_CSV: Path = TABLES_DIR / "phase2_tuned_comparison.csv"
PHASE2_REPORT_MD: Path = REPORTS_DIR / "phase2_model_comparison.md"

#: Saida da Fase 3 (modelo final, teste unico e diagnostico).
PHASE3_REPORT_MD: Path = REPORTS_DIR / "phase3_final_model.md"

MODEL_PATH: Path = MODELS_DIR / "model.joblib"
MODEL_METADATA_PATH: Path = MODELS_DIR / "model_metadata.json"

# --------------------------------------------------------------------------- #
# Variavel-alvo
# --------------------------------------------------------------------------- #

#: Nome da coluna-alvo no arquivo bruto.
TARGET_RAW: str = "Target"

#: Nome da coluna alvo binaria principal no dataset processado (D1).
TARGET_COLUMN: str = "target"

#: Nome da coluna com o rotulo original de 3 classes (rastreabilidade).
TARGET_CLASS_COLUMN: str = "target_class"

#: Nome da coluna alvo da analise de sensibilidade (D1, exclui "Matriculado").
TARGET_SENSITIVITY_COLUMN: str = "target_excl_enrolled"

CLASS_DROPOUT: str = "Desistente"
CLASS_GRADUATE: str = "Graduado"
CLASS_ENROLLED: str = "Matriculado"

#: Classes esperadas na coluna ``Target`` bruta.
EXPECTED_TARGET_CLASSES: tuple[str, ...] = (
    CLASS_DROPOUT,
    CLASS_GRADUATE,
    CLASS_ENROLLED,
)

# --------------------------------------------------------------------------- #
# Correcao de escala das notas (D4)
# --------------------------------------------------------------------------- #

#: Colunas afetadas pelo artefato de escala.
GRADE_COLUMNS: tuple[str, ...] = (
    "UnidadesCurriculares1SemestreGrau",
    "UnidadesCurriculares2SemestreGrau",
)

#: Escala de avaliacao portuguesa: 0 a 20.
GRADE_MAX_VALID: float = 20.0

#: Fator aplicado a cada passo da correcao.
GRADE_SCALE_BASE: float = 10.0

#: Coluna usada no teste de consistencia (Aprovado == 0 => Grau == 0).
#: Mantida para compatibilidade; o par correto e resolvido por semestre em
#: ``GRADE_APPROVED_COLUMNS``.
GRADE_CONSISTENCY_COLUMN: str = "UnidadesCurriculares1SemestreAprovado"

#: Pareamento nota x aprovado do **mesmo** semestre, usado na consistencia (D4).
#: Comparar semestres diferentes geraria falsas violacoes.
GRADE_APPROVED_COLUMNS: dict[str, str] = {
    "UnidadesCurriculares1SemestreGrau": "UnidadesCurriculares1SemestreAprovado",
    "UnidadesCurriculares2SemestreGrau": "UnidadesCurriculares2SemestreAprovado",
}

# --------------------------------------------------------------------------- #
# Agrupamento de colunas / features
# --------------------------------------------------------------------------- #

CATEGORICAL_FEATURES: tuple[str, ...] = (
    "EstadoCivil",
    "Curso",
    "QualificacaoAnterior",
    "Nacionalidade",
    "Genero",
)

BINARY_FEATURES: tuple[str, ...] = (
    "NecessidadesEspeciais",
    "Devedor",
    "MensalidadesEmDia",
    "Bolsista",
    "International",
)

#: Subconjunto de BINARY_FEATURES usado na analise de ablação (D3).
FINANCIAL_FEATURES: tuple[str, ...] = (
    "Devedor",
    "MensalidadesEmDia",
)

ADMISSION_FEATURES: tuple[str, ...] = (
    "NotaAdmissao",
    "QualificacaoAnteriorGrau",
)

SEMESTER1_FEATURES: tuple[str, ...] = (
    "UnidadesCurriculares1SemestreCreditado",
    "UnidadesCurriculares1SemestreInscrito",
    "UnidadesCurriculares1SemestreAvaliacoes",
    "UnidadesCurriculares1SemestreAprovado",
    "UnidadesCurriculares1SemestreGrau",
    "UnidadesCurriculares1SemestreSemAvaliacoes",
)

SEMESTER2_FEATURES: tuple[str, ...] = (
    "UnidadesCurriculares2SemestreCreditado",
    "UnidadesCurriculares2SemestreInscrito",
    "UnidadesCurriculares2SemestreAvaliacoes",
    "UnidadesCurriculares2SemestreAprovado",
    "UnidadesCurriculares2SemestreGrau",
    "UnidadesCurriculares2SemestreSemAvaliacoes",
)

#: Variaveis macroeconomicas do periodo (D7).
MACRO_FEATURES: tuple[str, ...] = (
    "TaxaDesemprego",
    "TaxaInflacao",
    "PIB",
)

#: Todas as features numericas que entram no ColumnTransformer.
NUMERIC_FEATURES: tuple[str, ...] = (
    ADMISSION_FEATURES
    + SEMESTER1_FEATURES
    + SEMESTER2_FEATURES
    + MACRO_FEATURES
    + BINARY_FEATURES
)

# --------------------------------------------------------------------------- #
# Cenarios de features (D2 / D3 / D7)
# --------------------------------------------------------------------------- #

#: Features derivadas do 1o semestre, criadas por ``src.features``.
#: Sao transformacoes **linha a linha** (sem estatisticas entre linhas), portanto
#: nao introduzem vazamento entre treino e teste.
ENGINEERED_SEMESTER1_FEATURES: tuple[str, ...] = (
    "taxa_aprovacao_1sem",
    "disciplinas_nao_aprovadas_1sem",
    "razao_avaliacoes_inscrito_1sem",
)

#: Features derivadas que dependem do 2o semestre (apenas no cenario Completo).
ENGINEERED_SEMESTER2_FEATURES: tuple[str, ...] = (
    "taxa_aprovacao_2sem",
    "disciplinas_nao_aprovadas_2sem",
    "delta_taxa_aprovacao",
    "delta_grau",
    "total_sem_avaliacoes",
)

#: Todas as features derivadas.
ENGINEERED_FEATURES: tuple[str, ...] = (
    ENGINEERED_SEMESTER1_FEATURES + ENGINEERED_SEMESTER2_FEATURES
)

#: Cenario 1 - Early Warning: elegivel para deploy por permitir intervencao cedo.
EARLY_WARNING_FEATURES: tuple[str, ...] = (
    CATEGORICAL_FEATURES
    + BINARY_FEATURES
    + ADMISSION_FEATURES
    + SEMESTER1_FEATURES
    + MACRO_FEATURES
    + ENGINEERED_SEMESTER1_FEATURES
)

#: Cenario 2 - Completo: acrescenta o 2o semestre (benchmark).
FULL_FEATURES: tuple[str, ...] = (
    EARLY_WARNING_FEATURES
    + SEMESTER2_FEATURES
    + ENGINEERED_SEMESTER2_FEATURES
)

#: Cenario Early Warning sem as variaveis financeiras (ablacao - D3).
EARLY_WARNING_NO_FINANCIAL_FEATURES: tuple[str, ...] = tuple(
    column
    for column in EARLY_WARNING_FEATURES
    if column not in FINANCIAL_FEATURES
)

#: Cenario Early Warning sem as variaveis macroeconomicas (ablacao - D7).
EARLY_WARNING_NO_MACRO_FEATURES: tuple[str, ...] = tuple(
    column for column in EARLY_WARNING_FEATURES if column not in MACRO_FEATURES
)

#: Cenario Completo sem as variaveis macroeconomicas (ablacao - D7).
FULL_NO_MACRO_FEATURES: tuple[str, ...] = tuple(
    column for column in FULL_FEATURES if column not in MACRO_FEATURES
)

#: Registro dos cenarios avaliados na Fase 2.
#: Chave = nome curto usado em relatorios; valor = descricao + colunas.
SCENARIOS: dict[str, dict] = {
    "early_warning": {
        "label": "Early Warning",
        "description": "Cadastro, socioeconomico, financeiro, ingresso e 1o semestre.",
        "columns": EARLY_WARNING_FEATURES,
    },
    "full": {
        "label": "Completo",
        "description": "Early Warning + variaveis do 2o semestre (benchmark).",
        "columns": FULL_FEATURES,
    },
    "early_warning_no_financial": {
        "label": "Early Warning sem financeiras",
        "description": "Ablacao D3: remove Devedor e MensalidadesEmDia.",
        "columns": EARLY_WARNING_NO_FINANCIAL_FEATURES,
    },
    "early_warning_no_macro": {
        "label": "Early Warning sem macro",
        "description": "Ablacao D7: remove TaxaDesemprego, TaxaInflacao e PIB.",
        "columns": EARLY_WARNING_NO_MACRO_FEATURES,
    },
    "full_no_macro": {
        "label": "Completo sem macro",
        "description": "Ablacao D7 no cenario Completo.",
        "columns": FULL_NO_MACRO_FEATURES,
    },
}

# --------------------------------------------------------------------------- #
# Divisao, validacao cruzada e metricas (D8)
# --------------------------------------------------------------------------- #

TEST_SIZE: float = 0.30
N_SPLITS: int = 5

#: Paralelismo da validacao cruzada e da busca de hiperparametros.
#: Padrao ``1`` (serial): mais lento, porem deterministico e sem os avisos de
#: ``multiprocessing`` que aparecem em ambientes com restricao de processos
#: (ex.: sandbox do VS Code via snap). Use ``-1`` para todos os nucleos.
N_JOBS: int = 1

#: Classe positiva em todas as metricas: evasao (``Desistente``).
POSITIVE_LABEL: int = 1

#: Metrica principal de otimizacao de hiperparametros (D8).
PRIMARY_SCORING: str = "f2"

#: Metricas reportadas em treino, validacao cruzada e teste.
REPORTED_METRICS: tuple[str, ...] = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "f2",
    "roc_auc",
)

#: Limiares de decisao avaliados no conjunto de validacao (D8).
THRESHOLDS: tuple[float, ...] = (
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
)
