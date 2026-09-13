"""Construcao do dataset processado e do relatorio de qualidade da base.

Uso
---
    python -m src.build_dataset

Fluxo (Fase 1)
--------------
1. Carrega a base bruta de ``data/raw`` e verifica a integridade (SHA-256).
2. Valida o schema (colunas e classes do alvo).
3. Registra valores ausentes.
4. Corrige a escala das colunas ``...SemestreGrau`` (D4).
5. Valida a consistencia ``Aprovado == 0 => Grau == 0`` (antes e depois).
6. Remove duplicatas exatas no estagio processado (D9).
7. Cria as duas binarizacoes do alvo (D1).
8. Salva ``data/processed/students_processed.csv``.
9. Gera ``reports/data_quality_report.{json,md}``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from . import config
from .cleaning import (
    correct_grade_columns,
    create_target,
    describe_target,
    remove_exact_duplicates,
    summarize_missing_values,
    validate_correction_by_distribution,
    validate_grade_consistency,
)
from .data_loader import compute_sha256, load_raw_data
from .schema import EXPECTED_COLUMNS, validate_schema


def build_processed_dataset() -> tuple[pd.DataFrame, dict]:
    """Executa o pipeline de preparacao da Fase 1.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        Dataset processado e relatorio de qualidade.
    """
    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_state": config.RANDOM_STATE,
        "steps": [],
    }

    # --- 1. Carga + integridade ------------------------------------------- #
    raw_df = load_raw_data(config.RAW_DATA_PATH)
    actual_sha256 = compute_sha256(config.RAW_DATA_PATH)
    report["source"] = {
        "raw_path": str(config.RAW_DATA_PATH),
        "original_path_preserved": str(config.LEGACY_RAW_PATH),
        "sha256": actual_sha256,
        "sha256_matches_expected": actual_sha256 == config.RAW_FILE_SHA256,
    }

    # --- 2. Schema --------------------------------------------------------- #
    schema_report = validate_schema(raw_df)
    report["schema"] = schema_report.as_dict()

    # --- 3. Valores ausentes ---------------------------------------------- #
    missing = summarize_missing_values(raw_df)
    report["missing_values"] = {
        "total_missing_cells": int(sum(missing.values())),
        "columns_with_missing": missing,
    }

    # --- 4/5. Correcao de escala + consistencia --------------------------- #
    consistency_before = {
        column: validate_grade_consistency(raw_df, grade_column=column)
        for column in config.GRADE_COLUMNS
    }
    corrected_df, grade_stats = correct_grade_columns(raw_df)
    consistency_after = {
        column: validate_grade_consistency(corrected_df, grade_column=column)
        for column in config.GRADE_COLUMNS
    }
    distribution_validation = [
        validate_correction_by_distribution(raw_df, column)
        for column in config.GRADE_COLUMNS
    ]
    report["grade_correction"] = {
        "method": (
            "Divisao sucessiva por 10 enquanto |valor| > 20 "
            "(consistente com perda do separador decimal)"
        ),
        "columns": grade_stats,
        "consistency_check": {
            "before": consistency_before,
            "after": consistency_after,
        },
        "consistency_check_note": (
            "Checagem de NAO-REGRESSAO: nao havia violacoes nem antes nem "
            "depois, pois os valores corrompidos ocorrem apenas em linhas "
            "com Aprovado > 0. Nao deve ser usada isoladamente como prova."
        ),
        "distribution_validation": distribution_validation,
    }

    # --- 6. Duplicatas ---------------------------------------------------- #
    deduplicated_df, duplicate_report = remove_exact_duplicates(
        corrected_df, subset=list(EXPECTED_COLUMNS)
    )
    report["duplicates"] = duplicate_report

    # --- 7. Alvo ---------------------------------------------------------- #
    processed_df = create_target(deduplicated_df)
    report["target"] = describe_target(processed_df)

    # --- 8. Persistencia -------------------------------------------------- #
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(config.PROCESSED_DATASET_PATH, index=False)
    report["processed_dataset"] = {
        "path": str(config.PROCESSED_DATASET_PATH),
        "n_rows": int(processed_df.shape[0]),
        "n_cols": int(processed_df.shape[1]),
        "columns": list(processed_df.columns),
        "dtypes": {
            column: str(dtype) for column, dtype in processed_df.dtypes.items()
        },
    }

    # --- Decisoes metodologicas aplicadas --------------------------------- #
    report["decisions_applied"] = {
        "D1": {
            "target_principal": f"{config.CLASS_DROPOUT} = 1; "
            f"{config.CLASS_GRADUATE} + {config.CLASS_ENROLLED} = 0",
            "coluna": config.TARGET_COLUMN,
            "alvo_sensibilidade": f"{config.CLASS_DROPOUT} = 1; "
            f"{config.CLASS_GRADUATE} = 0; {config.CLASS_ENROLLED} = excluido",
            "coluna_sensibilidade": config.TARGET_SENSITIVITY_COLUMN,
        },
        "D4": "Correcao aplicada somente na preparacao; arquivo original intacto.",
        "D5": "Copia de trabalho em data/raw; processado em data/processed.",
        "D9": "Uma linha = um estudante; duplicata exata removida apos a "
        "separacao bruto/processado.",
    }

    return processed_df, report


def _format_report_markdown(report: dict) -> str:
    """Renderiza o relatorio de qualidade em Markdown."""
    source = report["source"]
    schema = report["schema"]
    duplicate = report["duplicates"]
    target = report["target"]
    processed = report["processed_dataset"]

    lines: list[str] = []
    lines.append("# Relatorio de qualidade da base - Fase 1")
    lines.append("")
    lines.append(f"Gerado em: `{report['generated_at']}`")
    lines.append(f"`RANDOM_STATE` = `{report['random_state']}`")
    lines.append("")

    # Fonte
    lines.append("## 1. Fonte e integridade")
    lines.append("")
    lines.append(f"- Copia de trabalho: `{source['raw_path']}`")
    lines.append(f"- Original preservado: `{source['original_path_preserved']}`")
    lines.append(f"- SHA-256: `{source['sha256']}`")
    lines.append(
        f"- Integridade conferida: "
        f"{'OK' if source['sha256_matches_expected'] else 'DIVERGENTE'}"
    )
    lines.append("")

    # Schema
    lines.append("## 2. Validacao de schema")
    lines.append("")
    lines.append(f"- Linhas: {schema['n_rows']} | Colunas: {schema['n_cols']}")
    lines.append(f"- Colunas ausentes: {schema['missing_columns'] or 'nenhuma'}")
    lines.append(
        f"- Colunas inesperadas: {schema['unexpected_columns'] or 'nenhuma'}"
    )
    lines.append(f"- Resultado: {'OK' if schema['ok'] else 'FALHOU'}")
    lines.append("")

    # Alvo
    lines.append("## 3. Variavel-alvo")
    lines.append("")
    lines.append("### 3.1 Classes originais")
    lines.append("")
    lines.append("| Classe | N |")
    lines.append("|---|---|")
    for label, count in target.get("classes_originais", {}).items():
        lines.append(f"| {label} | {count} |")
    lines.append("")

    principal = target.get("target_principal", {})
    lines.append("### 3.2 Alvo principal (D1)")
    lines.append("")
    lines.append("`Desistente = 1` vs. `Graduado` + `Matriculado` = 0")
    lines.append("")
    lines.append("| Classe binaria | N |")
    lines.append("|---|---|")
    lines.append(f"| 1 (evasao) | {principal.get('desistente_1')} |")
    lines.append(f"| 0 (nao evasao) | {principal.get('outros_0')} |")
    lines.append(f"| Positivos | {principal.get('proporcao_positivos')} |")
    lines.append("")

    sensitivity = target.get("target_sensibilidade", {})
    lines.append("### 3.3 Alvo de sensibilidade (D1)")
    lines.append("")
    lines.append("`Desistente = 1` vs. `Graduado = 0` (Matriculado excluido)")
    lines.append("")
    lines.append("| Classe binaria | N |")
    lines.append("|---|---|")
    lines.append(f"| 1 (evasao) | {sensitivity.get('desistente_1')} |")
    lines.append(f"| 0 (graduado) | {sensitivity.get('graduado_0')} |")
    lines.append(
        f"| Excluidos (Matriculado) | {sensitivity.get('matriculado_excluido')} |"
    )
    lines.append(f"| Positivos | {sensitivity.get('proporcao_positivos')} |")
    lines.append("")

    # Duplicatas
    lines.append("## 4. Duplicatas exatas (D9)")
    lines.append("")
    lines.append(f"- Linhas antes: {duplicate['n_rows_before']}")
    lines.append(
        f"- Linhas em grupos duplicados: {duplicate['n_rows_in_duplicate_groups']}"
    )
    lines.append(f"- Grupos duplicados: {duplicate['n_duplicate_groups']}")
    lines.append(f"- Linhas removidas: {duplicate['n_removed']}")
    lines.append(f"- Linhas depois: {duplicate['n_rows_after']}")
    lines.append("")

    # Valores ausentes
    missing = report["missing_values"]
    lines.append("## 5. Valores ausentes")
    lines.append("")
    lines.append(f"- Total de celulas ausentes: {missing['total_missing_cells']}")
    if missing["columns_with_missing"]:
        for column, count in missing["columns_with_missing"].items():
            lines.append(f"  - `{column}`: {count}")
    else:
        lines.append("- Nenhuma coluna com valores ausentes.")
    lines.append("")

    # Correcao de escala
    grade = report["grade_correction"]
    lines.append("## 6. Correcao de escala das notas (D4)")
    lines.append("")
    lines.append(
        "Anomalia descrita como **consistente com a perda do separador "
        "decimal** (nao como causa comprovada)."
    )
    lines.append("")
    lines.append(f"Metodo: {grade['method']}")
    lines.append("")
    lines.append(
        "| Coluna | Valores | Corrigidos | % | Min antes | Max antes "
        "| Min depois | Max depois |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for column, stats in grade["columns"].items():
        lines.append(
            f"| `{column}` | {stats['n_values']} | {stats['n_corrected']} "
            f"| {stats['pct_corrected']} | {stats['min_before']:.4g} "
            f"| {stats['max_before']:.6g} | {stats['min_after']} "
            f"| {stats['max_after']} |"
        )
    lines.append("")

    lines.append("### 6.1 Exemplos representativos")
    lines.append("")
    for column, stats in grade["columns"].items():
        lines.append(f"**`{column}`**")
        lines.append("")
        lines.append("| Antes | Depois |")
        lines.append("|---|---|")
        for example in stats["examples"]:
            lines.append(
                f"| {example['before']:.6g} | {example['after']:.6g} |"
            )
        lines.append("")

    lines.append("### 6.2 Validacoes executadas")
    lines.append("")
    lines.append("**(a) Nao-regressao de consistencia**")
    lines.append("")
    lines.append(f"> {grade.get('consistency_check_note', '')}")
    lines.append("")
    for stage, checks in grade["consistency_check"].items():
        label = "antes" if stage == "before" else "depois"
        for column, check in checks.items():
            status = "OK" if check.get("ok") else "FALHOU"
            lines.append(
                f"- ({label}) `Aprovado == 0` x `{column}`: "
                f"{check.get('n_violations')} violacoes - {status}"
            )
    lines.append("")
    lines.append("**(b) Comparacao de distribuicoes (evidencia principal)**")
    lines.append("")
    lines.append("Valores nunca corrompidos vs. valores corrigidos:")
    lines.append("")
    lines.append("| Coluna | Grupo | N | Media | Mediana | Min | Max |")
    lines.append("|---|---|---|---|---|---|---|")
    for item in grade.get("distribution_validation", []):
        for group, label in (
            ("untouched", "nunca corrompidos"),
            ("corrected", "corrigidos"),
        ):
            summary = item.get(group, {})
            lines.append(
                f"| `{item['column']}` | {label} | {summary.get('n')} "
                f"| {summary.get('mean')} | {summary.get('median')} "
                f"| {summary.get('min')} | {summary.get('max')} |"
            )
    lines.append("")
    lines.append("**(c) Verificacoes adicionais**")
    lines.append("")
    for item in grade.get("distribution_validation", []):
        lines.append(
            f"- `{item['column']}`: todos os valores corrigidos dentro da escala"
            f" = {item.get('all_corrected_within_scale')}; valores corrompidos"
            f" que sao inteiros exatos = {item.get('n_corrupted_integers')}/"
            f"{item.get('n_corrupted')} ({item.get('pct_corrupted_integers')}%)"
        )
    lines.append("")

    # Estrutura processada
    lines.append("## 7. Estrutura da base processada")
    lines.append("")
    lines.append(f"- Arquivo: `{processed['path']}`")
    lines.append(f"- Dimensoes: {processed['n_rows']} linhas x {processed['n_cols']} colunas")
    lines.append("")
    lines.append("| # | Coluna | Tipo |")
    lines.append("|---|---|---|")
    for index, column in enumerate(processed["columns"]):
        lines.append(f"| {index} | `{column}` | {processed['dtypes'][column]} |")
    lines.append("")

    # Decisoes
    lines.append("## 8. Decisoes metodologicas aplicadas")
    lines.append("")
    for key, value in report["decisions_applied"].items():
        if isinstance(value, dict):
            lines.append(f"- **{key}**")
            for sub_key, sub_value in value.items():
                lines.append(f"  - {sub_key}: {sub_value}")
        else:
            lines.append(f"- **{key}**: {value}")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Executa a Fase 1 e escreve os artefatos."""
    processed_df, report = build_processed_dataset()

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.QUALITY_REPORT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    config.QUALITY_REPORT_MD.write_text(
        _format_report_markdown(report), encoding="utf-8"
    )

    corrected_total = sum(
        stats["n_corrected"] for stats in report["grade_correction"]["columns"].values()
    )

    print("=" * 64)
    print("FASE 1 - Base processada com sucesso")
    print("=" * 64)
    print(f"Linhas brutas ............: {report['schema']['n_rows']}")
    print(f"Linhas processadas .......: {processed_df.shape[0]}")
    print(f"Duplicatas removidas .....: {report['duplicates']['n_removed']}")
    print(f"Valores de nota corrigidos: {corrected_total}")
    print(f"Valores ausentes .........: {report['missing_values']['total_missing_cells']}")
    print()
    print("Alvo principal (D1):")
    for key, value in report["target"]["target_principal"].items():
        print(f"  {key}: {value}")
    print()
    print(f"Dataset processado: {config.PROCESSED_DATASET_PATH}")
    print(f"Relatorio JSON ....: {config.QUALITY_REPORT_JSON}")
    print(f"Relatorio MD ......: {config.QUALITY_REPORT_MD}")


if __name__ == "__main__":
    main()
