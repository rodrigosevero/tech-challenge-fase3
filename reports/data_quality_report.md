# Relatorio de qualidade da base - Fase 1

Gerado em: `2026-09-13T02:59:18+00:00`
`RANDOM_STATE` = `42`

## 1. Fonte e integridade

- Copia de trabalho: `/var/www/html/tech-challenge-fase3/data/raw/StudentsPrepared.xlsx`
- Original preservado: `/var/www/html/tech-challenge-fase3/data/StudentsPrepared.xlsx`
- SHA-256: `10687fd56d075289d7cc112592d977d56c3c88df90d6a2eb6deeb7f6be1206ba`
- Integridade conferida: OK

## 2. Validacao de schema

- Linhas: 4424 | Colunas: 28
- Colunas ausentes: nenhuma
- Colunas inesperadas: nenhuma
- Resultado: OK

## 3. Variavel-alvo

### 3.1 Classes originais

| Classe | N |
|---|---|
| Graduado | 2209 |
| Desistente | 1420 |
| Matriculado | 794 |

### 3.2 Alvo principal (D1)

`Desistente = 1` vs. `Graduado` + `Matriculado` = 0

| Classe binaria | N |
|---|---|
| 1 (evasao) | 1420 |
| 0 (nao evasao) | 3003 |
| Positivos | 0.321 |

### 3.3 Alvo de sensibilidade (D1)

`Desistente = 1` vs. `Graduado = 0` (Matriculado excluido)

| Classe binaria | N |
|---|---|
| 1 (evasao) | 1420 |
| 0 (graduado) | 2209 |
| Excluidos (Matriculado) | 794 |
| Positivos | 0.3913 |

## 4. Duplicatas exatas (D9)

- Linhas antes: 4424
- Linhas em grupos duplicados: 2
- Grupos duplicados: 1
- Linhas removidas: 1
- Linhas depois: 4423

## 5. Valores ausentes

- Total de celulas ausentes: 0
- Nenhuma coluna com valores ausentes.

## 6. Correcao de escala das notas (D4)

Anomalia descrita como **consistente com a perda do separador decimal** (nao como causa comprovada).

Metodo: Divisao sucessiva por 10 enquanto |valor| > 20 (consistente com perda do separador decimal)

| Coluna | Valores | Corrigidos | % | Min antes | Max antes | Min depois | Max depois |
|---|---|---|---|---|---|---|---|
| `UnidadesCurriculares1SemestreGrau` | 4424 | 1790 | 40.461 | 0 | 1.73333e+16 | 0.0 | 18.875 |
| `UnidadesCurriculares2SemestreGrau` | 4424 | 1675 | 37.862 | 0 | 1.85714e+16 | 0.0 | 18.571428571428502 |

### 6.1 Exemplos representativos

**`UnidadesCurriculares1SemestreGrau`**

| Antes | Depois |
|---|---|
| 1.34286e+16 | 13.4286 |
| 1.23333e+16 | 12.3333 |
| 1.18571e+16 | 11.8571 |
| 13875 | 13.875 |
| 1.23333e+16 | 12.3333 |

**`UnidadesCurriculares2SemestreGrau`**

| Antes | Depois |
|---|---|
| 1.36667e+16 | 13.6667 |
| 14345 | 14.345 |
| 1.41429e+16 | 14.1429 |
| 1.32143e+16 | 13.2143 |
| 14545 | 14.545 |

### 6.2 Validacoes executadas

**(a) Nao-regressao de consistencia**

> Checagem de NAO-REGRESSAO: nao havia violacoes nem antes nem depois, pois os valores corrompidos ocorrem apenas em linhas com Aprovado > 0. Nao deve ser usada isoladamente como prova.

- (antes) `Aprovado == 0` x `UnidadesCurriculares1SemestreGrau`: 0 violacoes - OK
- (antes) `Aprovado == 0` x `UnidadesCurriculares2SemestreGrau`: 0 violacoes - OK
- (depois) `Aprovado == 0` x `UnidadesCurriculares1SemestreGrau`: 0 violacoes - OK
- (depois) `Aprovado == 0` x `UnidadesCurriculares2SemestreGrau`: 0 violacoes - OK

**(b) Comparacao de distribuicoes (evidencia principal)**

Valores nunca corrompidos vs. valores corrigidos:

| Coluna | Grupo | N | Media | Mediana | Min | Max |
|---|---|---|---|---|---|---|
| `UnidadesCurriculares1SemestreGrau` | nunca corrompidos | 1916 | 12.4434 | 12.25 | 9.8 | 18.0 |
| `UnidadesCurriculares1SemestreGrau` | corrigidos | 1790 | 12.9796 | 12.8571 | 10.1667 | 18.875 |
| `UnidadesCurriculares2SemestreGrau` | nunca corrompidos | 1879 | 12.4557 | 12.33 | 10.0 | 17.6 |
| `UnidadesCurriculares2SemestreGrau` | corrigidos | 1675 | 13.0472 | 12.9667 | 10.1667 | 18.5714 |

**(c) Verificacoes adicionais**

- `UnidadesCurriculares1SemestreGrau`: todos os valores corrigidos dentro da escala = True; valores corrompidos que sao inteiros exatos = 1790/1790 (100.0%)
- `UnidadesCurriculares2SemestreGrau`: todos os valores corrigidos dentro da escala = True; valores corrompidos que sao inteiros exatos = 1675/1675 (100.0%)

## 7. Estrutura da base processada

- Arquivo: `/var/www/html/tech-challenge-fase3/data/processed/students_processed.csv`
- Dimensoes: 4423 linhas x 31 colunas

| # | Coluna | Tipo |
|---|---|---|
| 0 | `EstadoCivil` | object |
| 1 | `Curso` | object |
| 2 | `QualificacaoAnterior` | object |
| 3 | `QualificacaoAnteriorGrau` | float64 |
| 4 | `Nacionalidade` | object |
| 5 | `NotaAdmissao` | float64 |
| 6 | `NecessidadesEspeciais` | int64 |
| 7 | `Devedor` | int64 |
| 8 | `MensalidadesEmDia` | int64 |
| 9 | `Genero` | object |
| 10 | `Bolsista` | int64 |
| 11 | `International` | int64 |
| 12 | `UnidadesCurriculares1SemestreCreditado` | int64 |
| 13 | `UnidadesCurriculares1SemestreInscrito` | int64 |
| 14 | `UnidadesCurriculares1SemestreAvaliacoes` | int64 |
| 15 | `UnidadesCurriculares1SemestreAprovado` | int64 |
| 16 | `UnidadesCurriculares1SemestreGrau` | float64 |
| 17 | `UnidadesCurriculares1SemestreSemAvaliacoes` | int64 |
| 18 | `UnidadesCurriculares2SemestreCreditado` | int64 |
| 19 | `UnidadesCurriculares2SemestreInscrito` | int64 |
| 20 | `UnidadesCurriculares2SemestreAvaliacoes` | int64 |
| 21 | `UnidadesCurriculares2SemestreAprovado` | int64 |
| 22 | `UnidadesCurriculares2SemestreGrau` | float64 |
| 23 | `UnidadesCurriculares2SemestreSemAvaliacoes` | int64 |
| 24 | `TaxaDesemprego` | float64 |
| 25 | `TaxaInflacao` | float64 |
| 26 | `PIB` | float64 |
| 27 | `Target` | object |
| 28 | `target_class` | string |
| 29 | `target` | int64 |
| 30 | `target_excl_enrolled` | Int64 |

## 8. Decisoes metodologicas aplicadas

- **D1**
  - target_principal: Desistente = 1; Graduado + Matriculado = 0
  - coluna: target
  - alvo_sensibilidade: Desistente = 1; Graduado = 0; Matriculado = excluido
  - coluna_sensibilidade: target_excl_enrolled
- **D4**: Correcao aplicada somente na preparacao; arquivo original intacto.
- **D5**: Copia de trabalho em data/raw; processado em data/processed.
- **D9**: Uma linha = um estudante; duplicata exata removida apos a separacao bruto/processado.
