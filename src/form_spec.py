"""Especificacao dos campos do formulario da aplicacao.

Fica em ``src`` (e nao no ``app``) por dois motivos:

1. e uma decisao de **dominio** (quais campos o modelo precisa e quais valores
   sao validos), nao de interface;
2. permite **testar a construcao do payload** sem subir o Streamlit.

O formulario e montado a partir de ``model_metadata.json``: so aparecem os campos
que o modelo realmente usa. Assim, se um dia o cenario mudar para *Completo*, o app
passa a pedir tambem o 2o semestre automaticamente.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field

import pandas as pd

#: Valores de contingencia usados quando o dataset processado nao esta disponivel
#: (por exemplo, em um deploy somente com o modelo). Os valores espelham as modas
#: e medianas observadas na base da Fase 1.
FALLBACK_OPTIONS: dict[str, tuple[str, ...]] = {
    "EstadoCivil": (
        "Casado",
        "Divorciado",
        "Separado Judicialmente",
        "Solteiro",
        "União de Fato",
        "Viúvo",
    ),
    "Genero": ("Feminino", "Masculino"),
    "QualificacaoAnterior": (
        "Curso de Especialização Tecnológica",
        "Curso Técnico Superior Profissional",
        "Ensino Básico (3º Ciclo)",
        "Ensino Secundário",
        "Ensino Superior - Licenciatura",
        "Outro - 11º Ano de Escolaridade",
    ),
    "Nacionalidade": ("Português",),
    "Curso": (
        "Agronomia",
        "Design de Animação e Multimédia",
        "Design de Comunicação",
        "Enfermagem",
        "Enfermagem Veterinária",
        "Engenharia Informática",
        "Ensino Básico",
        "Equincultura",
        "Gestão",
        "Higiene Oral",
        "Jornalismo e Comunicação",
        "Serviço Social",
        "Turismo",
    ),
}


@dataclass(frozen=True)
class FieldSpec:
    """Como um campo do modelo deve ser apresentado e validado no formulario."""

    column: str
    label: str
    kind: str  # "categorical" | "binary" | "numeric"
    help: str
    min_value: float | None = None
    max_value: float | None = None
    step: float = 1.0
    default: object = 0
    unit: str = ""
    group: str = dataclass_field(default="Geral")


#: Ordem dos campos de desempenho do 1o semestre (usada nos formularios).
_SEMESTER1 = "1º semestre"


FIELD_SPECS: dict[str, FieldSpec] = {
    # --- cadastro ---------------------------------------------------------- #
    "EstadoCivil": FieldSpec(
        "EstadoCivil", "Estado civil", "categorical", "Situacao civil do estudante.",
        default="Solteiro", group="Cadastro",
    ),
    "Genero": FieldSpec(
        "Genero", "Genero", "categorical", "Genero informado no cadastro.",
        default="Feminino", group="Cadastro",
    ),
    "Nacionalidade": FieldSpec(
        "Nacionalidade", "Nacionalidade", "categorical", "Nacionalidade do estudante.",
        default="Português", group="Cadastro",
    ),
    "Curso": FieldSpec(
        "Curso", "Curso", "categorical", "Curso em que o estudante esta matriculado.",
        default="Enfermagem", group="Cadastro",
    ),
    "QualificacaoAnterior": FieldSpec(
        "QualificacaoAnterior", "Qualificacao anterior", "categorical",
        "Habilitacao com que o estudante ingressou.", default="Ensino Secundário",
        group="Cadastro",
    ),
    "NecessidadesEspeciais": FieldSpec(
        "NecessidadesEspeciais", "Necessidades especiais", "binary",
        "1 = possui necessidades educativas especiais.", default=0, group="Cadastro",
    ),
    "International": FieldSpec(
        "International", "Estudante internacional", "binary",
        "1 = estudante internacional.", default=0, group="Cadastro",
    ),
    # --- socioeconomico / financeiro --------------------------------------- #
    "Devedor": FieldSpec(
        "Devedor", "Devedor", "binary",
        "1 = possui debito com a instituicao.", default=0, group="Financeiro",
    ),
    "MensalidadesEmDia": FieldSpec(
        "MensalidadesEmDia", "Mensalidades em dia", "binary",
        "1 = mensalidades em dia.", default=1, group="Financeiro",
    ),
    "Bolsista": FieldSpec(
        "Bolsista", "Bolsista", "binary", "1 = recebe bolsa de estudos.",
        default=0, group="Financeiro",
    ),
    # --- ingresso ---------------------------------------------------------- #
    "NotaAdmissao": FieldSpec(
        "NotaAdmissao", "Nota de admissao", "numeric",
        "Nota de ingresso (escala 95-190).", min_value=95.0, max_value=190.0,
        step=0.5, default=127.0, group="Ingresso",
    ),
    "QualificacaoAnteriorGrau": FieldSpec(
        "QualificacaoAnteriorGrau", "Grau da qualificacao anterior", "numeric",
        "Pontuacao da habilitacao anterior (escala 95-190).",
        min_value=95.0, max_value=190.0, step=0.5, default=133.0, group="Ingresso",
    ),
    # --- 1o semestre ------------------------------------------------------- #
    "UnidadesCurriculares1SemestreCreditado": FieldSpec(
        "UnidadesCurriculares1SemestreCreditado", "Disciplinas creditadas", "numeric",
        "Unidades curriculares com credito no 1o semestre.",
        min_value=0.0, max_value=20.0, default=0.0, group=_SEMESTER1,
    ),
    "UnidadesCurriculares1SemestreInscrito": FieldSpec(
        "UnidadesCurriculares1SemestreInscrito", "Disciplinas inscritas", "numeric",
        "Unidades curriculares em que se inscreveu.",
        min_value=0.0, max_value=26.0, default=6.0, group=_SEMESTER1,
    ),
    "UnidadesCurriculares1SemestreAvaliacoes": FieldSpec(
        "UnidadesCurriculares1SemestreAvaliacoes", "Avaliacoes realizadas", "numeric",
        "Quantidade de avaliacoes no 1o semestre.",
        min_value=0.0, max_value=45.0, default=8.0, group=_SEMESTER1,
    ),
    "UnidadesCurriculares1SemestreAprovado": FieldSpec(
        "UnidadesCurriculares1SemestreAprovado", "Disciplinas aprovadas", "numeric",
        "Unidades curriculares aprovadas no 1o semestre.",
        min_value=0.0, max_value=26.0, default=5.0, group=_SEMESTER1,
    ),
    "UnidadesCurriculares1SemestreGrau": FieldSpec(
        "UnidadesCurriculares1SemestreGrau", "Nota media (0-20)", "numeric",
        "Media das notas do 1o semestre, na escala portuguesa (0 a 20).",
        min_value=0.0, max_value=20.0, step=0.1, default=12.5, group=_SEMESTER1,
    ),
    "UnidadesCurriculares1SemestreSemAvaliacoes": FieldSpec(
        "UnidadesCurriculares1SemestreSemAvaliacoes", "Disciplinas sem avaliacao",
        "numeric", "Unidades curriculares sem nenhuma avaliacao.",
        min_value=0.0, max_value=12.0, default=0.0, group=_SEMESTER1,
    ),
    # --- 2o semestre (apenas no cenario Completo) -------------------------- #
    "UnidadesCurriculares2SemestreCreditado": FieldSpec(
        "UnidadesCurriculares2SemestreCreditado", "Disciplinas creditadas", "numeric",
        "Unidades curriculares com credito no 2o semestre.",
        min_value=0.0, max_value=19.0, default=0.0, group="2º semestre",
    ),
    "UnidadesCurriculares2SemestreInscrito": FieldSpec(
        "UnidadesCurriculares2SemestreInscrito", "Disciplinas inscritas", "numeric",
        "Unidades curriculares em que se inscreveu.",
        min_value=0.0, max_value=23.0, default=6.0, group="2º semestre",
    ),
    "UnidadesCurriculares2SemestreAvaliacoes": FieldSpec(
        "UnidadesCurriculares2SemestreAvaliacoes", "Avaliacoes realizadas", "numeric",
        "Quantidade de avaliacoes no 2o semestre.",
        min_value=0.0, max_value=33.0, default=8.0, group="2º semestre",
    ),
    "UnidadesCurriculares2SemestreAprovado": FieldSpec(
        "UnidadesCurriculares2SemestreAprovado", "Disciplinas aprovadas", "numeric",
        "Unidades curriculares aprovadas no 2o semestre.",
        min_value=0.0, max_value=20.0, default=5.0, group="2º semestre",
    ),
    "UnidadesCurriculares2SemestreGrau": FieldSpec(
        "UnidadesCurriculares2SemestreGrau", "Nota media (0-20)", "numeric",
        "Media das notas do 2o semestre, na escala portuguesa (0 a 20).",
        min_value=0.0, max_value=20.0, step=0.1, default=12.5, group="2º semestre",
    ),
    "UnidadesCurriculares2SemestreSemAvaliacoes": FieldSpec(
        "UnidadesCurriculares2SemestreSemAvaliacoes", "Disciplinas sem avaliacao",
        "numeric", "Unidades curriculares sem nenhuma avaliacao.",
        min_value=0.0, max_value=12.0, default=0.0, group="2º semestre",
    ),
    # --- contexto macroeconomico ------------------------------------------- #
    "TaxaDesemprego": FieldSpec(
        "TaxaDesemprego", "Taxa de desemprego (%)", "numeric",
        "Taxa de desemprego do periodo (contexto, nao do aluno).",
        min_value=7.6, max_value=16.2, step=0.1, default=12.7, group="Macroeconomia",
    ),
    "TaxaInflacao": FieldSpec(
        "TaxaInflacao", "Taxa de inflacao (%)", "numeric",
        "Taxa de inflacao do periodo.", min_value=-0.8, max_value=3.7,
        step=0.1, default=0.5, group="Macroeconomia",
    ),
    "PIB": FieldSpec(
        "PIB", "PIB (variacao %)", "numeric", "Variacao do PIB do periodo.",
        min_value=-4.06, max_value=3.51, step=0.01, default=1.74,
        group="Macroeconomia",
    ),
}

#: Ordem em que os grupos aparecem no formulario.
GROUP_ORDER: tuple[str, ...] = (
    "Cadastro",
    "Financeiro",
    "Ingresso",
    _SEMESTER1,
    "2º semestre",
    "Macroeconomia",
    "Geral",
)


def columns_for_model(input_columns: list[str]) -> list[str]:
    """Mantem a ordem do modelo, ignorando campos sem especificacao."""
    return [column for column in input_columns if column in FIELD_SPECS]


def unknown_columns(input_columns: list[str]) -> list[str]:
    """Campos exigidos pelo modelo que ainda nao tem especificacao de formulario."""
    return [column for column in input_columns if column not in FIELD_SPECS]


def group_fields(columns: list[str]) -> dict[str, list[FieldSpec]]:
    """Agrupa as especificacoes por secao, na ordem definida em ``GROUP_ORDER``."""
    grouped: dict[str, list[FieldSpec]] = {}
    for column in columns:
        spec = FIELD_SPECS[column]
        grouped.setdefault(spec.group, []).append(spec)

    ordered: dict[str, list[FieldSpec]] = {}
    for group in GROUP_ORDER:
        if group in grouped:
            ordered[group] = grouped[group]
    for group, specs in grouped.items():
        if group not in ordered:
            ordered[group] = specs
    return ordered


def default_payload(columns: list[str]) -> dict:
    """Valores iniciais do formulario (perfil tipico da base)."""
    return {column: FIELD_SPECS[column].default for column in columns}


def build_payload(values: dict, columns: list[str]) -> dict:
    """Monta o dicionario enviado ao modelo, com apenas os campos esperados."""
    return {column: values[column] for column in columns if column in values}


def categorical_options(
    data: pd.DataFrame | None,
    column: str,
) -> list[str]:
    """Valores validos de um campo categorico.

    Prefere os valores observados na base processada; se ela nao estiver
    disponivel, usa a lista de contingencia (``FALLBACK_OPTIONS``).
    """
    if data is not None and column in data.columns:
        values = (
            data[column].dropna().astype(str).drop_duplicates().sort_values()
        )
        if not values.empty:
            return values.tolist()

    fallback = FALLBACK_OPTIONS.get(column)
    if fallback:
        return sorted(fallback)
    return []
