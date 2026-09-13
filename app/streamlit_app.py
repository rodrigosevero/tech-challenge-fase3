"""Aplicacao Streamlit do Tech Challenge - Fase 3 (previsao de evasao).

Execucao local
--------------
    streamlit run app/streamlit_app.py

A aplicacao **nao treina nada**: ela carrega o pipeline serializado em
``models/model.joblib`` e chama ``src.predict.predict_risk``. Toda a logica de
previsao vive no pacote ``src``, evitando duplicacao entre treino e producao.

Principio de uso (D10)
----------------------
A saida e apresentada como **risco**, nunca como decisao automatica sobre o
estudante. O app sempre acompanha a probabilidade de uma recomendacao de acao.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Garante que o pacote ``src`` seja importavel quando o app roda a partir de
# ``app/`` (o Streamlit adiciona apenas a pasta do script ao sys.path).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.form_spec import (  # noqa: E402
    FIELD_SPECS,
    build_payload,
    categorical_options,
    columns_for_model,
    default_payload,
    group_fields,
    unknown_columns,
)
from src.predict import (  # noqa: E402
    ModelNotFoundError,
    load_metadata,
    load_model,
    predict_risk,
)

RISK_COLORS = {"Baixo": "🟢", "Moderado": "🟡", "Alto": "🔴"}


# --------------------------------------------------------------------------- #
# Carregamento (com cache)
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner=False)
def get_model():
    """Carrega o pipeline uma unica vez por sessao."""
    return load_model()


@st.cache_data(show_spinner=False)
def get_metadata() -> dict:
    """Le os metadados do modelo (limiar, metricas, features)."""
    return load_metadata()


@st.cache_data(show_spinner=False)
def get_reference_data() -> pd.DataFrame | None:
    """Base processada, usada apenas para popular as opcoes dos campos."""
    try:
        return pd.read_csv(config.PROCESSED_DATASET_PATH)
    except Exception:  # pragma: no cover - depende do ambiente de deploy
        return None


# --------------------------------------------------------------------------- #
# Exemplos para demonstracao
# --------------------------------------------------------------------------- #

HIGH_RISK_EXAMPLE = {
    "EstadoCivil": "Solteiro",
    "Genero": "Masculino",
    "Nacionalidade": "Português",
    "Curso": "Enfermagem",
    "QualificacaoAnterior": "Ensino Secundário",
    "NecessidadesEspeciais": 0,
    "International": 0,
    "Devedor": 1,
    "MensalidadesEmDia": 0,
    "Bolsista": 0,
    "NotaAdmissao": 110.0,
    "QualificacaoAnteriorGrau": 120.0,
    "UnidadesCurriculares1SemestreCreditado": 0,
    "UnidadesCurriculares1SemestreInscrito": 6,
    "UnidadesCurriculares1SemestreAvaliacoes": 2,
    "UnidadesCurriculares1SemestreAprovado": 0,
    "UnidadesCurriculares1SemestreGrau": 0.0,
    "UnidadesCurriculares1SemestreSemAvaliacoes": 4,
    "TaxaDesemprego": 12.7,
    "TaxaInflacao": 0.5,
    "PIB": 1.74,
}

LOW_RISK_EXAMPLE = {
    **HIGH_RISK_EXAMPLE,
    "Devedor": 0,
    "MensalidadesEmDia": 1,
    "Bolsista": 1,
    "NotaAdmissao": 160.0,
    "UnidadesCurriculares1SemestreAprovado": 6,
    "UnidadesCurriculares1SemestreGrau": 15.5,
    "UnidadesCurriculares1SemestreAvaliacoes": 9,
    "UnidadesCurriculares1SemestreSemAvaliacoes": 0,
}


# --------------------------------------------------------------------------- #
# Formulario
# --------------------------------------------------------------------------- #


def seed_session_state(columns: list[str], data: pd.DataFrame | None) -> None:
    """Preenche o estado inicial dos campos (perfil tipico da base)."""
    defaults = default_payload(columns)
    for column in columns:
        if column in st.session_state:
            continue
        spec = FIELD_SPECS[column]
        if spec.kind == "categorical":
            options = categorical_options(data, column)
            value = spec.default if spec.default in options else None
            if value is None and options:
                value = options[0]
            st.session_state[column] = value
        else:
            st.session_state[column] = defaults[column]


def apply_example(values: dict) -> None:
    """Sobrescreve os campos com um exemplo e recarrega a pagina."""
    for column, value in values.items():
        if column in FIELD_SPECS:
            st.session_state[column] = value
    st.session_state.pop("resultado", None)
    st.rerun()


def render_field(column: str, data: pd.DataFrame | None) -> None:
    """Renderiza um campo conforme o seu tipo."""
    spec = FIELD_SPECS[column]

    if spec.kind == "categorical":
        options = categorical_options(data, column)
        if not options:
            st.text_input(spec.label, key=column, help=spec.help)
            return
        if st.session_state.get(column) not in options:
            st.session_state[column] = options[0]
        st.selectbox(spec.label, options=options, key=column, help=spec.help)
        return

    if spec.kind == "binary":
        current = st.session_state.get(column, spec.default)
        if current not in (0, 1):
            st.session_state[column] = spec.default
        st.selectbox(
            spec.label,
            options=[0, 1],
            key=column,
            help=spec.help,
            format_func=lambda value: "Sim" if value == 1 else "Não",
        )
        return

    st.number_input(
        spec.label,
        min_value=float(spec.min_value) if spec.min_value is not None else None,
        max_value=float(spec.max_value) if spec.max_value is not None else None,
        step=float(spec.step),
        key=column,
        help=spec.help,
    )


def render_form(columns: list[str], data: pd.DataFrame | None) -> dict | None:
    """Desenha o formulario e devolve o payload quando enviado."""
    grouped = group_fields(columns)

    with st.form("form_estudante", border=True):
        for group, specs in grouped.items():
            st.markdown(f"**{group}**")
            left, right = st.columns(2)
            for index, spec in enumerate(specs):
                target = left if index % 2 == 0 else right
                with target:
                    render_field(spec.column, data)
            st.divider()

        submitted = st.form_submit_button(
            "Calcular risco de evasão", type="primary", use_container_width=True
        )

    if not submitted:
        return None

    return build_payload(
        {column: st.session_state.get(column) for column in columns}, columns
    )


# --------------------------------------------------------------------------- #
# Resultado
# --------------------------------------------------------------------------- #


def render_result(resultado: dict, metadata: dict) -> None:
    """Mostra a probabilidade, a faixa de risco e a recomendacao."""
    probability = float(resultado["probability"])
    band = resultado["risk_band"]
    icon = RISK_COLORS.get(band, "⚪")

    st.subheader("Resultado")
    left, right = st.columns([1, 2])

    with left:
        st.metric("Probabilidade de evasão", f"{probability:.1%}")
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption(
            f"Classificação com limiar {resultado['threshold']:.2f} → "
            f"{'alerta de risco' if resultado['is_dropout_risk'] else 'sem alerta'}"
        )

    with right:
        if band == "Alto":
            st.error(f"{icon} **Risco {band}** — {resultado['recommendation']}")
        elif band == "Moderado":
            st.warning(f"{icon} **Risco {band}** — {resultado['recommendation']}")
        else:
            st.success(f"{icon} **Risco {band}** — {resultado['recommendation']}")

        st.info(
            "Esta é uma **estimativa de risco** gerada por um modelo estatístico. "
            "Ela **não é uma decisão** sobre o estudante e não deve ser usada de "
            "forma isolada: serve para **priorizar** ações de acompanhamento e "
            "apoio, sempre com avaliação humana."
        )


def render_model_info(metadata: dict, model) -> None:
    """Transparencia: o que o modelo e, como foi avaliado e o que ele ve."""
    with st.expander("Sobre o modelo (transparência)"):
        test = metadata.get("test_metrics_threshold_tuned", {})
        diagnosis = metadata.get("diagnosis", {})
        threshold = metadata.get("decision_threshold")
        threshold_text = f"{float(threshold):.2f}" if threshold is not None else "—"

        st.markdown(
            f"""
- **Algoritmo:** `{metadata.get('model_name')}`
- **Cenário:** {metadata.get('scenario_label')} ({metadata.get('n_features')} features)
- **Alvo:** {metadata.get('target_definition')}
- **Limiar de decisão:** {threshold_text}
  (escolhido na validação, nunca no teste)
"""
        )

        if test:
            st.markdown("**Desempenho no conjunto de teste (avaliado uma única vez)**")
            colunas = st.columns(4)
            for coluna, (rotulo, chave) in zip(
                colunas,
                (
                    ("Recall", "recall"),
                    ("Precision", "precision"),
                    ("F1", "f1"),
                    ("ROC-AUC", "roc_auc"),
                ),
            ):
                coluna.metric(rotulo, f"{test.get(chave, float('nan')):.3f}")

            st.markdown(
                f"""
**Matriz de confusão no teste:** {test.get('tp')} evasões detectadas,
{test.get('fn')} não detectadas, {test.get('fp')} alarmes falsos e
{test.get('tn')} acertos de não evasão.

**Diagnóstico de ajuste:** `{diagnosis.get('diagnosis')}` — {diagnosis.get('explanation')}
"""
            )

        st.caption(
            "Lembrete: o recall mede quantos evadidos o modelo captura. Priorizamos "
            "o recall da evasão, aceitando mais alarmes falsos, porque o custo de "
            "não identificar um estudante em risco é maior."
        )

        st.markdown("**Campos utilizados pelo modelo**")
        st.code(", ".join(metadata.get("input_columns", [])), language="text")


# --------------------------------------------------------------------------- #
# Principal
# --------------------------------------------------------------------------- #


def main() -> None:
    st.set_page_config(
        page_title="Previsão de evasão estudantil",
        page_icon="🎓",
        layout="wide",
    )

    st.title("🎓 Previsão de evasão estudantil")
    st.caption(
        "Tech Challenge — Fase 3 | Pós-Graduação em Machine Learning Engineering (FIAP)"
    )

    try:
        metadata = get_metadata()
        model = get_model()
    except ModelNotFoundError as error:
        st.error(
            "O modelo ainda não foi gerado.\n\n"
            "Execute no terminal:\n\n"
            "```bash\npython -m src.select_model\n```\n\n"
            f"Detalhe técnico: {error}"
        )
        st.stop()

    if not metadata:
        st.error(
            "Metadados do modelo não encontrados. "
            "Execute `python -m src.select_model` para gerá-los."
        )
        st.stop()

    input_columns = metadata.get("input_columns", [])
    model_columns = columns_for_model(input_columns)
    missing = unknown_columns(input_columns)
    if missing:
        st.warning(
            "Estes campos são exigidos pelo modelo mas não têm formulário: "
            + ", ".join(missing)
        )

    data = get_reference_data()

    with st.sidebar:
        st.header("Como usar")
        st.markdown(
            "1. Preencha os dados do estudante.\n"
            "2. Clique em **Calcular risco de evasão**.\n"
            "3. Use os botões de exemplo para ver os extremos."
        )
        st.divider()
        st.markdown("**Exemplos rápidos**")
        if st.button("🔴 Perfil de risco alto", use_container_width=True):
            apply_example(HIGH_RISK_EXAMPLE)
        if st.button("🟢 Perfil de risco baixo", use_container_width=True):
            apply_example(LOW_RISK_EXAMPLE)
        if st.button("↩️ Restaurar padrão", use_container_width=True):
            for key in list(st.session_state):
                if key in FIELD_SPECS or key == "resultado":
                    del st.session_state[key]
            st.rerun()
        st.divider()
        st.caption(
            "Uso acadêmico. O resultado é uma estimativa de risco e não substitui "
            "a avaliação de um profissional."
        )

    seed_session_state(model_columns, data)

    payload = render_form(model_columns, data)
    if payload is not None:
        try:
            st.session_state["resultado"] = predict_risk(
                payload, model=model, metadata=metadata
            )
        except Exception as error:  # pragma: no cover - protecao de runtime
            st.error(f"Não foi possível calcular a previsão: {error}")

    resultado = st.session_state.get("resultado")
    if resultado:
        render_result(resultado, metadata)

    st.divider()
    render_model_info(metadata, model)


if __name__ == "__main__":
    main()
