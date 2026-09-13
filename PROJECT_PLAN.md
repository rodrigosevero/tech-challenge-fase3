# PROJECT_PLAN.md — Tech Challenge Fase 3 (FIAP — Machine Learning Engineering)

> Documento de **planejamento**, atualizado após a autorização de implementação.
>
> **Status de execução**
>
> | Fase | Escopo | Situação |
> |---|---|---|
> | 1 | Ambiente, organização, carga, limpeza, correção, testes, relatório | ✅ concluída e executada |
> | 2 | Baseline, modelos, validação cruzada, cenários e ablações | ✅ concluída (ver §22) |
> | 3 | Seleção do modelo, teste único, over/underfitting, artefatos | ✅ concluída (ver §23) |
> | 4 | Streamlit, documentação, deploy, roteiro do vídeo | ✅ concluída (ver §24) |
>
> As decisões **D1–D10** foram confirmadas pelo autor e estão registradas na §20. As seções
> abaixo foram atualizadas para refletir essas decisões.

---

## 0. Diagnóstico de entrada (discrepâncias encontradas)

Antes de qualquer decisão técnica, foi feita a verificação física dos arquivos no workspace.
Há divergências relevantes entre o que foi descrito na solicitação e o que existe em disco:

| Esperado (na solicitação) | Encontrado em disco | Situação |
|---|---|---|
| `docs/enunciado-fase-3.pdf` | `docs/MLET - Prova Substitutiva - Fase 3.pdf` | Nome diferente (**2 páginas**, "Prova Substitutiva Fase 3") |
| `data/base-sub-fase-3.csv` | `data/StudentsPrepared.xlsx` | Formato diferente (`.xlsx`, não `.csv`) |
| `README.md` / `PROJECT_PLAN.md` | não existiam | Criados agora |
| `PROM.md` | existe, **0 bytes** (vazio) | Sem conteúdo a considerar |
| Python 3.11 ou 3.12 | Apenas **Python 3.14.4** (`pip` 25.1.1 disponível; sem `pyenv`/`uv`/`conda`) | ⚠️ **D6**: usar 3.12; comandos no README |

> **Nada foi renomeado, movido ou convertido.** O arquivo original permanece intacto em
> `data/StudentsPrepared.xlsx`; foi criada uma **cópia idêntica** em
> `data/raw/StudentsPrepared.xlsx` (mesmo SHA-256), conforme **D5**.

---

## 1. Entendimento do problema

O enunciado pede uma **pipeline de um modelo preditivo binário** capaz de prever a
**evasão de estudantes** (dropout) de uma faculdade, a partir de uma base com informações
socioeconômicas, de admissão e de desempenho acadêmico.

O problema é, portanto, de **classificação binária supervisionada**, com foco declarado
(na solicitação do autor) no **recall da classe que representa evasão** — ou seja, o custo de
**não identificar** um aluno em risco é considerado maior que o custo de um falso positivo.

### Contexto de negócio / uso
Instituição de ensino que quer **priorizar ações de retenção** (tutoria, apoio financeiro,
acompanhamento pedagógico) para alunos com alta probabilidade de evasão.

### Definição do momento da predição — ✅ decidido em D2
A base contém variáveis de **1º e 2º semestres**. Decisão confirmada: **não** tratar as
variáveis de 2º semestre automaticamente como vazamento, mas **implementar e comparar dois
cenários**:

- **Cenário 1 — Early Warning:** cadastro, socioeconômico, financeiro, ingresso e 1º semestre.
  É o **candidato preferencial ao deploy**, por permitir intervenção mais cedo.
- **Cenário 2 — Completo:** Early Warning **+ 2º semestre**. Usado como **benchmark**.

A comparação entre os dois cenários será reportada explicitamente (treino, CV e teste),
discutindo o ganho obtido e o custo temporal de obtê-lo. Grupos de features definidos em
`src/config.py` (`EARLY_WARNING_FEATURES` e `FULL_FEATURES`).

---

## 2. Requisitos do enunciado → entrega técnica concreta

Transcrição fiel dos requisitos do PDF e o mapeamento para entregáveis:

| # | Requisito do enunciado (PDF) | Entrega técnica concreta |
|---|---|---|
| R1 | "Realize técnicas de feature engineering nos dados para tratar dados numéricos e categóricos." | `src/preprocessing.py` + `src/features.py`: `Pipeline` + `ColumnTransformer` (imputação + escala para numéricas; `OneHotEncoder` para categóricas) e features derivadas (taxa de aprovação, média de grau, status financeiro). |
| R2 | "Realize a separação da base em treino e teste." | `train_test_split` **estratificado** (70/30) em `src/data_loader.py`, com `random_state` centralizado. |
| R3 | "Treine um modelo binário com a estratégia de validação cruzada." | `StratifiedKFold` (k=5) em `src/train.py`, comparando **Regressão Logística, Random Forest e Gradient Boosting**. |
| R4 | "Analise os resultados do modelo justificando se houve caso de overfitting e underfitting." | `src/evaluate.py` + notebook: comparação **treino × validação cruzada × teste**, curvas de aprendizado/validação e análise de gap. |
| R5 | "Realize o deploy do modelo utilizando a aplicação do streamlit." | `app/streamlit_app.py` consumindo `models/model.joblib`. |
| R6 | "Disponibilize o modelo em um repositório do GitHub com documentação dos passos desenvolvidos e conclusões da análise." | `README.md`, `PROJECT_PLAN.md`, notebook executável e `docs/` versionados em repositório Git. |
| R7 | "Grave um vídeo de no mínimo 5 minutos explicando as técnicas realizadas e apresente a aplicação deployada." | Roteiro detalhado em §17 deste documento (com marcação de tempo por bloco). |
| R8 | "A entrega deve ser um arquivo .txt com o link do repositório do git, link da aplicação deployada no streamlit e link do vídeo." | `docs/entregas/links.txt` com os 3 links. |

### Requisitos técnicos adicionais (definidos pelo autor)
Python 3.11/3.12 · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Joblib ·
Streamlit · notebook Jupyter executável no VS Code · ambiente virtual local · código
organizado e reproduzível · `random_state` centralizado · prevenção de data leakage ·
`Pipeline`/`ColumnTransformer` sempre que aplicável · comparação entre Regressão Logística,
Random Forest e Gradient Boosting · métricas Accuracy, Precision, Recall, F1, ROC-AUC e
matriz de confusão · atenção especial ao recall da classe de evasão · comparação entre
desempenho de treino, validação cruzada e teste.

---

## 3. Descrição inicial da base

Fonte: `data/StudentsPrepared.xlsx` (inspeção feita com a biblioteca padrão do Python,
sem instalar dependências).

### 3.1 Estrutura geral
- **Formato:** Excel `.xlsx` (OpenXML), **1 aba** (`Sheet1`), planilha única.
- **Dimensão:** `A1:AB4425` → **4.424 linhas de dados** × **28 colunas**.
- **Codificação:** strings em *shared strings* internas do XLSX (sem problemas de encoding
  aparentes; acentuação correta: "Português", "União de Fato", "Viúvo").
- **Identificador:** **não existe** coluna de ID/`matrícula`. **D9:** uma linha = **um
  estudante**, definição confirmada pela documentação original da base UCI.
- **Valores ausentes:** **0** em todas as 28 colunas.
- **Duplicatas:** **1 linha exata duplicada** (1 grupo de 2 registros).

### 3.2 Dicionário de colunas (28)

**Socioeconômicas / cadastrais (categóricas e flags):**

| Coluna | Tipo | Valores distintos | Observação |
|---|---|---|---|
| `EstadoCivil` | categórica | 6 | Solteiro 3919, Casado 379, Divorciado 91, União de Fato 25, Separado Judicialmente 6, Viúvo 4 |
| `Curso` | categórica | 17 | Enfermagem 766 (maior), Tecnologias de Produção de Biocombustíveis 12 (menor) |
| `QualificacaoAnterior` | categórica | 16 | Ensino Secundário 3717 (dominante); cauda longa |
| `Nacionalidade` | categórica | 21 | Português 4314; cauda longa (muitas com 1–3 registros) |
| `Genero` | categórica | 2 | Feminino 2868, Masculino 1556 |
| `NecessidadesEspeciais` | flag 0/1 | 2 | 1 → 51 |
| `Devedor` | flag 0/1 | 2 | 1 → 503 |
| `MensalidadesEmDia` | flag 0/1 | 2 | 1 → 3896 |
| `Bolsista` | flag 0/1 | 2 | 1 → 1099 |
| `International` | flag 0/1 | 2 | 1 → 110 |

**Admissão / ingresso (numéricas):**

| Coluna | Tipo | Observação |
|---|---|---|
| `NotaAdmissao` | numérica contínua | min 95, max 190, média 126,98; 620 valores distintos |
| `QualificacaoAnteriorGrau` | numérica | 101 distintos, min 95, max 190, média 132,61 — mesma escala de `NotaAdmissao` |

**Desempenho acadêmico 1º semestre (numéricas):**

| Coluna | min–max | Observação |
|---|---|---|
| `UnidadesCurriculares1SemestreCreditado` | 0–20 | média 0,71 |
| `UnidadesCurriculares1SemestreInscrito` | 0–26 | média 6,27 |
| `UnidadesCurriculares1SemestreAvaliacoes` | 0–45 | média 8,30 |
| `UnidadesCurriculares1SemestreAprovado` | 0–26 | média 4,71 |
| `UnidadesCurriculares1SemestreGrau` | **0–1,73e16** | ⚠️ **CORROMPIDA** — ver §3.3 |
| `UnidadesCurriculares1SemestreSemAvaliacoes` | 0–12 | média baixa (4130 zeros) |

**Desempenho acadêmico 2º semestre (numéricas):**
Mesma estrutura do 1º semestre (6 colunas), com `...2SemestreGrau` **igualmente corrompida**
(até 1,86e16).

**Macroeconômicas (numéricas — 10 valores distintos cada):**

| Coluna | min–max | Valores |
|---|---|---|
| `TaxaDesemprego` | 7,6–16,2 | 10 valores (discretos/anualizados) |
| `TaxaInflacao` | −0,8–3,7 | 9 valores |
| `PIB` | −4,06–3,51 | 10 valores |

> Padrão notável: `TaxaDesemprego`, `TaxaInflacao` e `PIB` compartilham exatamente as mesmas
> contagens de frequência (893/571/533/445/419/414/397/390/368/362). Isso indica que formam
> **tripletas macroeconômicas por período/ano** — ou seja, há **colinearidade/agrupamento
> temporal** entre elas. Considerar tratar como agrupamento ou reduzir a dimensionalidade.

### 3.3 ⚠️ Inconsistência crítica: colunas `...SemestreGrau`

As duas colunas de "Grau" contêm valores **impossíveis** como `1,34285714285714e+16`
e `13875`. Investigação detalhada:

- Valores > 20 são **1.790 no 1º semestre** e **1.675 no 2º semestre**.
- **Causa raiz identificada:** as colunas sofreram um artefato de **perda do separador
  decimal** — o valor original (nota entre 0 e 20, escala portuguesa) foi multiplicado por
  uma potência de 10 e armazenado como inteiro grande. Ex.: `13.4285714285714` → `1.34285714285714e+16`;
  `10.375` → `10375`.
- **Regra de correção validada:** aplicar `enquanto valor > 20: valor = valor / 10`.
  Resultado: todos os 4.424 valores caem em **[0, 20]**, faixa compatível com o sistema
  de avaliação português (ver §6).
- **Teste de consistência cruzada:** nenhuma linha tem `...Aprovado == 0` com
  `...Grau > 0` após a correção (0 violações) → **a regra de correção é sólida**.

### 3.4 Distribuição da variável-alvo (`Target`)

**3 classes** (não binária), sem valores ausentes:

| Classe | Significado | N | % |
|---|---|---|---|
| `Graduado` | concluiu o curso | 2.209 | 49,9% |
| `Desistente` | evadiu | 1.421 | 32,1% |
| `Matriculado` | ainda matriculado | 794 | 18,0% |

- **Balanceamento:** desbalanceada, mas não extremo. Já como problema binário
  "Desistente vs. demais": **1.421 vs. 3.003 (32,1% positivos)** → claramente tratável,
  com **estratificação** obrigatória na divisão e na validação cruzada.

### 3.5 Possíveis inconsistências adicionais
1. `QualificacaoAnteriorGrau` (numérica, escala 95–190) **não é grau** no sentido de nota
   do curso — parece codificação de nível/escala de admissão. Precisa ser interpretada
   com cautela (o nome é enganoso).
2. `NecessidadesEspeciais` com apenas 51 casos na classe 1 → variável quase constante.
3. `Nacionalidade` (21 categorias, muitas com 1–3 registros) e `QualificacaoAnterior`
   (16 categorias) têm **cauda longa** → exigem agrupamento/`handle_unknown='ignore'`.
4. 1 linha duplicada exata.
5. Ausência de ID → risco de contagem indevida de estudantes. **Resolvido em D9** (uma linha
   = um estudante, conforme a documentação original da base UCI).

---

## 4. Hipótese para a variável-alvo

**Variável-alvo: coluna `Target`.**

Binarização **confirmada em D1**. O enunciado exige classificação **binária**, mas a base é
**multiclasse**. As opções avaliadas foram:

| Opção | Definição | Positivos | Negativos | Positivo (%) | Comentário |
|---|---|---|---|---|---|
| **A (recomendada)** | `Desistente = 1`; `Graduado` + `Matriculado` = 0 | 1.421 | 3.003 | 32,1% | Usa a base inteira; "evasão" = saiu vs. permaneceu/concluiu. Mais aderente a "prever evasão". |
| B | `Desistente = 1`; `Graduado = 0`; **descartar `Matriculado`** | 1.421 | 2.209 | 39,1% | Evita misturar "ainda cursando" com "concluiu"; perde 794 registros. |
| C | `Desistente` + `Matriculado` = 1 (não-graduado) | 2.215 | 2.209 | 50,1% | Excelente balanceamento, mas "Matriculado" **não é evasão** → alvo semanticamente incorreto. |

**Decisão confirmada (D1): Opção A** como alvo principal (`Desistente = 1`; `Graduado` +
`Matriculado` = 0), por ser a leitura mais direta de "prever a evasão" e por preservar 100%
dos dados. A **Opção B** (`Desistente` vs. `Graduado`, excluindo `Matriculado`) é o alvo da
**análise de sensibilidade**, implementado na coluna `target_excl_enrolled`; a diferença
entre as duas definições será documentada no notebook e no README.

> Os números da tabela acima referem-se à **base bruta** (4.424 linhas). Após a remoção da
> duplicata exata (D9), a base processada tem **4.423 linhas**, com **1.420** evasões no alvo
> principal (32,10% de positivos) e 1.420 vs. 2.209 no alvo de sensibilidade (39,13%).

### 4.1 Risco de vazamento de dados (data leakage) — análise quantitativa

Associação univariada de cada feature com `Desistente` (eta para numéricas, χ² para categóricas):

| Feature | Associação | Leitura |
|---|---|---|
| `...2SemestreGrau` (corrigida) | eta = **0,572** | 🔴 altíssima — indicador de desfecho |
| `...2SemestreAprovado` | eta = **0,570** | 🔴 altíssima |
| `...1SemestreGrau` (corrigida) | eta = 0,481 | 🟠 forte (legítima no Cenário A) |
| `...1SemestreAprovado` | eta = 0,479 | 🟠 forte (legítima no Cenário A) |
| `MensalidadesEmDia` | eta = 0,429 | 🟠 forte — ver risco abaixo |
| `Bolsista` | eta = 0,245 | 🟡 moderada |
| `Devedor` | eta = 0,229 | 🟠 forte — ver risco abaixo |
| `Genero` | χ² = 184,1 | 🟡 moderada |
| `Curso` | χ² = 298,3 | 🟡 moderada |
| `QualificacaoAnterior` | χ² = 201,7 | 🟡 moderada |
| `NotaAdmissao` | eta = 0,096 | 🟢 fraca |
| `TaxaDesemprego` / `TaxaInflacao` / `PIB` | eta = 0,013 / 0,028 / 0,046 | 🟢 muito fracas |
| `NecessidadesEspeciais` | eta = 0,003 | 🟢 irrelevante |

**Riscos de vazamento identificados:**

1. **Features de 2º semestre** (`...2Semestre*`): correlacionam-se fortemente com o desfecho.
   ⚠️ **Ponto contraintuitivo importante:** 1.344 dos 1.421 `Desistente` possuem dados de
   2º semestre (`inscrito_2sem > 0`), assim como 2.134 dos 2.209 `Graduado`. Ou seja, a base
   **não** separa "quem saiu antes do 2º semestre". Usar essas features implica que o modelo
   só funciona **após** o 2º semestre — o que pode ser tarde para ações de retenção.
   → **Tratado em D2:** o 2º semestre entra apenas no **Cenário Completo** (benchmark), não
   no Early Warning candidato ao deploy.
2. **`MensalidadesEmDia`**: 457 dos 528 inadimplentes são `Desistente`.
   É plausivelmente registrado/atualizado **depois** da evasão (o aluno que sai para de pagar)
   → relação pode ser consequência, não causa. **D3:** não remover automaticamente; executar
   análise de ablação (com e sem variáveis financeiras) e documentar a dependência temporal
   como **hipótese**, sem afirmar vazamento sem evidência sobre o momento de coleta.
3. **`Devedor`**: mesma natureza financeira de `MensalidadesEmDia` (103 de 503 devedores evadem
   vs. 1.109 de 3.921 não-devedores) → risco moderado.
4. **`...1SemestreGrau`/`...Aprovado`**: legítimas no Cenário A, mas **são o desfecho
   acadêmico parcial** — é o previsor mais realista disponível e deve ser mantido.
5. **`TaxaDesemprego`/`TaxaInflacao`/`PIB`**: são **macroeconômicas do período**, não do aluno.
   Se a base for usada para prever anos futuros, valores de macro do ano-alvo não estariam
   disponíveis → risco de vazamento temporal. Associação é baixa, então o ganho de mantê-las
   é pequeno.

**Mitigação:** comparar explicitamente cenários de features (A: sem 2º semestre;
B: com 2º semestre), documentar a diferença de desempenho e **adotar o cenário sem vazamento**
como modelo oficial.

---

## 5. Estratégia de análise exploratória (EDA)

Notebook `notebooks/01_eda_e_modelagem.ipynb`, executável no VS Code, com `random_state`
fixo no topo. Seções previstas:

1. **Carga e visão geral:** `shape`, `dtypes`, `head`, `info`, `describe` (numéricas e categóricas).
2. **Qualidade de dados:** contagem de nulos, duplicatas, cardinalidade, valores únicos por coluna.
3. **Validação do artefato de escala** nas colunas `...Grau` (histograma antes/depois da correção).
4. **Distribuição da variável-alvo:** `countplot` — 3 classes originais e binário (Opção A/B).
5. **Análise univariada:**
   - Numéricas: histogramas + boxplots (com detecção de outliers por IQR).
   - Categóricas: `countplot`/barras horizontais, com destaque para cauda longa.
6. **Análise bivariada (features × alvo):**
   - Numéricas: boxplots/violin por classe; médias por classe; teste de Mann-Whitney/χ²
     quando fizer sentido.
   - Categóricas: tabelas de contingência **normalizadas por linha** e taxa de evasão por
     categoria (evitar leitura enganosa de contagens absolutas).
7. **Mapa de correlação:** heatmap `seaborn` (Pearson + Spearman) para numéricas;
   detectar multicolinearidade (esperada no trio macro e entre contagens acadêmicas).
8. **Análise de vazamento:** evidência visual da relação `...2SemestreGrau × Target` antes/depois.
9. **Conclusões da EDA:** lista de features candidatas, features a descartar, e features
   a monitorar por vazamento.

**Saídas:** figuras salvas em `reports/figures/`.

---

## 6. Estratégia de limpeza

Todas as transformações serão feitas em código versionado (`src/`), nunca "na mão" no Excel.

| # | Problema | Tratamento proposto |
|---|---|---|
| L1 | `...1SemestreGrau` e `...2SemestreGrau` com escala corrompida | Normalizar com `while v > 20: v /= 10`. Validado: 100% dos valores caem em [0, 20] e 0 violações de consistência com `Aprovado`. |
| L2 | 1 linha duplicada exata | `drop_duplicates()` (registrar a decisão no notebook). |
| L3 | Ausentes | **Não há ausentes.** Mesmo assim, o `ColumnTransformer` incluirá imputação (`SimpleImputer`) para robustez da pipeline em produção (novos dados podem vir incompletos). |
| L4 | Categorias de cauda longa (`Nacionalidade`, `QualificacaoAnterior`, `EstadoCivil`) | Agrupar categorias raras em `"Outros"` (limiar a definir, ex.: < 10 ocorrências) e/ou `OneHotEncoder(handle_unknown="ignore")`. |
| L5 | Outliers numéricos | Investigar (não remover automaticamente). Decidir entre *clipping*, transformação ou manutenção, conforme impacto medido na EDA. |
| L6 | Colunas macro (`TaxaDesemprego`, `TaxaInflacao`, `PIB`) | Decidir entre manter como numéricas, tratar como agrupamento temporal, ou remover (associação muito baixa + risco temporal). |
| L7 | Escalas diferentes (`NotaAdmissao` 95–190 vs. contagens 0–26) | Padronizar numéricas com `StandardScaler` **dentro** do pipeline (fit apenas no treino). |
| L8 | Tipos mistos | Garantir `dtype` numérico nas numéricas e `string`/`category` nas categóricas na carga. |
| L9 | Flags 0/1 lidas corretamente | Forçar `int`/`bool`, confirmando que não há rótulos textuais inconsistentes. |

**Princípio:** nenhuma estatística de limpeza (média, mediana, categorias, escala) pode ser
calculada com o conjunto de teste. Toda transformação que "aprende" parâmetros (imputação,
escala, encoding) será encapsulada em `Pipeline`/`ColumnTransformer` e ajustada
**somente no treino**, via `fit` / `fit_transform` no `Pipeline`.

---

## 7. Feature engineering

### 7.1 Tratamento das categóricas
- `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` para `Curso`, `EstadoCivil`,
  `Nacionalidade`, `QualificacaoAnterior`, `Genero`.
- Agrupamento de categorias raras antes do encoding (ver L4).
- **Alternativa a avaliar:** `OrdinalEncoder` para `QualificacaoAnterior` se houver
  ordenação natural (nível de escolaridade) — comparar desempenho.

### 7.2 Tratamento das numéricas
- `SimpleImputer(strategy="median")` + `StandardScaler` (dentro do `ColumnTransformer`).
- Manter as colunas `...Grau` **corrigidas** (não as brutas).

### 7.3 Features derivadas (hipóteses a testar — não implementadas nesta etapa)
| Feature proposta | Fórmula | Racional |
|---|---|---|
| `taxa_aprovacao_1sem` | `Aprovado / Inscrito` (1º sem.) | Eficiência acadêmica direta; evita o efeito do volume de disciplinas. |
| `taxa_aprovacao_2sem` | idem (2º sem.) | Mesma lógica (apenas no Cenário B). |
| `taxa_reprovacao_1sem` | `(Avaliacoes - Aprovado) / Inscrito` | Esforço sem sucesso. |
| `delta_aprovado` | `Aprovado_2sem − Aprovado_1sem` | Trajetória/melhora ao longo do tempo. |
| `delta_grau` | `Grau_2sem − Grau_1sem` | Trajetória de nota. |
| `risco_financeiro` | `Devedor == 1` **e/ou** `MensalidadesEmDia == 0` | Sinal financeiro combinado (⚠️ avaliar vazamento). |
| `apoio_recebido` | `Bolsista` | Benefício socioeconômico. |
| `aluno_internacional` | `International` | Perfil de risco/adaptação. |

**Regra:** features derivadas serão criadas **dentro** do pipeline (ou com `FunctionTransformer`)
para evitar vazamento de estatísticas entre treino e teste.

⚠️ Features derivadas de 2º semestre seguem a mesma restrição de cenário de §1.

---

## 8. Divisão entre treino e teste

- **Método:** `train_test_split` com **`stratify=y`** (obrigatório — classes desbalanceadas).
- **Proporção:** **70% treino / 30% teste**.
- **Reprodutibilidade:** `random_state = RANDOM_STATE` (constante em `src/config.py`, ex.: 42).
- **O teste é sagrado:** usado **uma única vez**, ao final, para o modelo já escolhido.
  Nenhuma decisão de feature, hiperparâmetro ou limiar será tomada olhando o teste.
- Toda seleção de modelo acontece **dentro do treino**, via validação cruzada (§10).

---

## 9. Modelos candidatos

Três modelos, sempre encapsulados em `Pipeline(ColumnTransformer + classifier)`:

| Modelo | Papel | Justificativa |
|---|---|---|
| **Regressão Logística** | *Baseline* interpretável | Simples, coeficientes interpretáveis, boa referência para detectar se o problema é linearmente separável. Usar `class_weight="balanced"`. |
| **Random Forest** | Não-linear, robusto | Captura interações, resistente a outliers, fornece `feature_importances_`. Usar `class_weight="balanced"`. |
| **Gradient Boosting** | Estado da arte tabular | Alto poder preditivo; comparar `HistGradientBoostingClassifier` (rápido) com tuning de learning rate/profundidade. |

**Modelos comparados:** `DummyClassifier` (baseline), `LogisticRegression`,
`RandomForestClassifier` e `HistGradientBoostingClassifier`/`GradientBoostingClassifier`.

**Observação:** serão comparados **antes e depois** do ajuste de hiperparâmetros, para
demonstrar ganho de desempenho. `class_weight` será aplicado **apenas nos modelos que o
suportam** (`LogisticRegression`, `RandomForestClassifier`) e o resultado **com e sem**
balanceamento será comparado e reportado.

---

## 10. Estratégia de validação cruzada

- **Esquema:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)`.
  Justificativa: preserva a proporção das classes em cada fold, essencial com 32% de positivos.
- **Integração:** `cross_validate` / `GridSearchCV` **recebendo o Pipeline completo** como
  estimador — garante que imputação, escala e encoding sejam ajustados **apenas** no fold de
  treino de cada iteração (**sem vazamento de fold**).
- **Métricas de CV:** `accuracy`, `precision`, `recall`, `f1`, `f2`, `roc_auc`.
  **D8:** o `scoring`/`refit` usa **F2-score**, que pondera o recall 4x mais que a precision —
  adequado ao objetivo (evasão é a classe crítica) **sem** otimizar recall isoladamente.
- **Busca de hiperparâmetros:** `GridSearchCV` (grade pequena e explícita, para ser didática
  no vídeo) ou `RandomizedSearchCV` se a grade crescer. Registrar a grade no notebook.
- **Intervalo:** reportar média **e desvio-padrão** das métricas entre folds (estabilidade).

---

## 11. Métricas utilizadas

| Métrica | Por quê |
|---|---|
| **Accuracy** | Visão geral, mas **insuficiente** com 32% de positivos (um modelo "sempre não-evade" já acerta ~68%). |
| **Precision (evasão)** | Mede o custo de falsos alarmes (ações de retenção desnecessárias). |
| **Recall (evasão)** | ⭐ **Métrica prioritária**: mede quantos evadidos o modelo efetivamente captura. |
| **F1-score (evasão)** | Equilíbrio precision/recall em uma só métrica. |
| **F2-score (evasão)** | ⭐ **Métrica de otimização (D8)** — pondera o recall 4x mais que a precision. |
| **ROC-AUC** | Qualidade do *ranking* de probabilidade, independente do limiar. |
| **Matriz de confusão** | Leitura explícita dos 4 tipos de erro; obrigatória no relatório e no vídeo. |

- Definir **classe positiva = `Desistente` (1)** de forma explícita em todos os cálculos.
- **Ajuste de limiar (*threshold tuning*) — D8:** após escolher o melhor modelo, avaliar
  diferentes limiares (`THRESHOLDS` em `src/config.py`) **exclusivamente no conjunto de
  validação**, nunca no teste. O limiar escolhido é congelado e aplicado ao teste **uma única
  vez**. Reportar o trade-off precision/recall.
- **Prioridade operacional:** o **recall da evasão** orienta a decisão de negócio, mas a
  otimização usa **F2** — evitando maximizar recall isoladamente (o que degradaria a precision).
- Reportar também `classification_report` completo e a matriz de confusão normalizada.

---

## 12. Abordagem para identificar overfitting e underfitting

Comparação sistemática dos três níveis, para cada modelo:

1. **Desempenho no treino** (`model.score` / predições no próprio treino).
2. **Desempenho na validação cruzada** (média ± desvio dos 5 folds).
3. **Desempenho no teste** (avaliação final, uma única vez).

**Interpretação (regra de decisão explícita):**

| Padrão observado | Diagnóstico | Ação |
|---|---|---|
| Treino ≈ CV ≈ Teste, todos **altos** | Ajuste adequado | Aprovado |
| Treino **muito alto** e CV/Teste **muito menores** (gap grande) | **Overfitting** | Regularizar (reduzir profundidade, aumentar `min_samples_leaf`, reduzir `n_estimators`/learning rate, aumentar regularização L2/C, reduzir nº de features) |
| Treino **baixo**, CV/Teste **baixos** e próximos | **Underfitting** | Aumentar complexidade (mais features, modelo não-linear, menos regularização) |
| CV **muito melhor** que Teste | Sorte/instabilidade na divisão | Revisar estratificação, tamanho de folds, variância entre folds |

**Ferramentas visuais obrigatórias:**
- **Curva de aprendizado** (`learning_curve`): desempenho × tamanho do conjunto de treino →
  evidencia overfitting (gap que não fecha) e underfitting (ambas as curvas baixas).
- **Curva de validação** (`validation_curve`): desempenho × hiperparâmetro-chave
  (ex.: `C` da Logística, `max_depth` do RF/GB) → mostra o ponto de virada.
- **Gap treino−CV** quantificado em tabela, por modelo.
- **Estabilidade:** desvio-padrão entre folds.

Narrativa esperada para o vídeo: mostrar **um caso de overfitting** (provavelmente RF/GB com
profundidade alta) e a correção aplicada, e comentar se a Regressão Logística ficou próxima de
**underfitting** — justificando a escolha final com **evidência**, não com opinião.

---

## 13. Critérios para escolha do modelo final

Ordem de prioridade (definida **antes** de ver os resultados, para evitar *cherry-picking*):

1. **Recall da classe `Desistente`** — critério principal (objetivo do projeto).
2. **F1-score e ROC-AUC** — equilíbrio e qualidade do ranking.
3. **Estabilidade** — desvio-padrão baixo entre folds (não escolher por sorte de fold).
4. **Gap treino−CV** — preferir o modelo com menor sinal de overfitting.
5. **Desempenho no teste** — confirmação final (não usado para "escolher", e sim para validar).
6. **Interpretabilidade e custo computacional** — desempate; Regressão Logística/importâncias
   do RF ajudam a explicar o modelo no vídeo.

**Regra de desempate:** se o ganho de recall for marginal (< 2 p.p.) e houver aumento relevante
de variância ou de complexidade, escolher o **modelo mais simples e explicável**.

**Entrega:** tabela comparativa final (modelo × {treino, CV, teste} × {métricas}) + escolha
justificada por escrito no notebook e no README.

---

## 14. Estratégia de serialização

- **Formato:** `joblib.dump` / `joblib.load`.
- **O que é salvo:** o **Pipeline completo** (pré-processamento + modelo), não apenas o
  classificador. Isso garante que a aplicação Streamlit aplique **exatamente** as mesmas
  transformações aprendidas no treino — prevenindo o "trem de inconsistência" entre treino
  e inferência.
- **Artefato:** `models/model.joblib`.
- **Metadados:** salvar junto (`models/model_metadata.json`) com: data de treino, versão do
  scikit-learn, `RANDOM_STATE`, definição do alvo (Opção A/B), lista de features, cenário
  (com/sem 2º semestre), limiar de decisão adotado e métricas do teste.
- **Boas práticas:** versionar/congelar `requirements.txt`; registrar a versão do
  scikit-learn (pipelines não são garantidamente compatíveis entre versões maiores);
  não versionar o `.joblib` se for muito grande (usar `.gitignore` + instrução de regeneração).
- **Função de inferência:** `src/predict.py` com API única
  (ex.: `predict_proba(dict) -> float`) usada tanto pela Streamlit quanto pelos testes,
  evitando duplicação de lógica.

---

## 15. Estrutura da aplicação Streamlit

`app/streamlit_app.py`, interface simples e demonstrável em vídeo:

1. **Cabeçalho:** título, contexto do problema, aviso de foco acadêmico.
2. **Entrada de dados** — `st.sidebar` ou formulário em colunas, agrupado por tema:
   - Cadastro/socioeconômico: `EstadoCivil`, `Genero`, `Nacionalidade`, `Curso`,
     `QualificacaoAnterior`, `Bolsista`, `International`, `NecessidadesEspeciais`, `Devedor`,
     `MensalidadesEmDia`.
   - Admissão: `NotaAdmissao`, `QualificacaoAnteriorGrau`.
   - Desempenho (1º semestre; 2º semestre apenas se o cenário escolhido permitir).
   - Contexto macro (se mantido no modelo final).
3. **Predição:** botão → `predict_proba` → exibição da **probabilidade de evasão** com
   `st.metric`. Classificação via limiar escolhido em §11.
4. **Feedback visual:** barra/medidor de risco, faixa textual
   ("risco alto / médio / baixo") e mensagem de recomendação de ação de retenção.
5. **Transparência:** seção expansível com as features enviadas, a definição do alvo, o
   limiar usado e as métricas do modelo (carregadas do `metadata.json`).
6. **Robustez:** `st.cache_resource` para carregar o modelo uma vez; validação de entrada;
   mensagem amigável se `model.joblib` não existir; sem qualquer lógica de treino no app.

**Arquivo de apoio:** `.streamlit/config.toml` (tema/porta) — se necessário.

---

## 16. Estratégia de deploy

**Recomendado: Streamlit Community Cloud** (gratuito, integrado ao GitHub, ideal para o
trabalho acadêmico).

Procedimento previsto:
1. Publicar o repositório no GitHub (com `requirements.txt` explícito e o modelo acessível).
2. No Streamlit Community Cloud, criar o app apontando para `app/streamlit_app.py`, branch `main`.
3. Garantir que o artefato `models/model.joblib` esteja versionado **ou** que o app o
   baixe/regenera de forma determinística no primeiro uso (definir na implementação).
4. Registrar a URL pública para o `docs/entregas/links.txt`.

**Alternativas** (fallback, documentar no README):
- **Streamlit local** (`streamlit run app/streamlit_app.py`) — obrigatório como prova de
  reprodutibilidade no vídeo.
- **Docker** (opcional) — `Dockerfile` para portabilidade.
- **Hugging Face Spaces** (Streamlit SDK) — alternativa caso a cota do Community Cloud seja
  um problema.

⚠️ Riscos de deploy: limite de memória do plano gratuito; incompatibilidade de versão do
scikit-learn entre treino e deploy (mitigar com versão fixada no `requirements.txt`).

---

## 17. Documentação e roteiro do vídeo

### 17.1 Documentação (arquivos do repositório)
| Arquivo | Conteúdo |
|---|---|
| `README.md` | Objetivo, contexto acadêmico, estrutura, instruções de configuração/execução, link do app e do vídeo. |
| `PROJECT_PLAN.md` | Este documento (diagnóstico/planejamento). |
| `notebooks/01_eda_e_modelagem.ipynb` | EDA + modelagem executada, com gráficos e conclusões comentadas. |
| `docs/` | PDF do enunciado + `entregas/links.txt`. |
| `reports/figures/` | Figuras exportadas (para uso no README e no vídeo). |

### 17.2 Roteiro do vídeo (mínimo 5 min — planejado para ~7 min)
| Tempo | Bloco | Conteúdo |
|---|---|---|
| 0:00–0:40 | Problema e contexto | Objetivo: prever evasão; por que importa (retenção); métrica prioritária = recall. |
| 0:40–1:40 | Base de dados e EDA | Origem da base, 4.424 alunos, 28 colunas, as 3 classes; gráfico da distribuição do alvo. |
| 1:40–2:30 | Qualidade de dados | **Mostrar o artefato das colunas de Grau** e a correção aplicada (ponto forte de storytelling). |
| 2:30–3:20 | Feature engineering e pipeline | Explicar `ColumnTransformer`, one-hot, escala, features derivadas, `random_state`, prevenção de vazamento (e a decisão sobre o 2º semestre). |
| 3:20–4:20 | Modelagem e validação | Os 3 modelos, `StratifiedKFold`, `GridSearchCV`, a métrica de refit. |
| 4:20–5:20 | Resultados | Tabela treino/CV/teste; **overfitting e underfitting**; matriz de confusão; foco no recall. |
| 5:20–6:00 | Escolha do modelo | Critérios de §13 e justificativa final. |
| 6:00–7:00 | Demo Streamlit | Preencher o formulário, mostrar a probabilidade de evasão e o alerta de risco. |
| 7:00–7:20 | Conclusões e limitações | Aprendizados, riscos (vazamento financeiro/2º semestre) e trabalhos futuros. |

> Reforço: o vídeo deve mostrar a aplicação **realmente deployada** (URL pública), não apenas local.

---

## 18. Estrutura de diretórios definitiva (proposta)

```
tech-challenge-fase3/
├── .gitignore
├── README.md
├── PROJECT_PLAN.md
├── requirements.txt
├── data/
│   ├── raw/                    # base original, imutável (StudentsPrepared.xlsx)
│   └── processed/              # datasets limpos gerados pelo pipeline
├── docs/
│   ├── MLET - Prova Substitutiva - Fase 3.pdf
│   └── entregas/
│       └── links.txt           # repositório + app + vídeo
├── notebooks/
│   └── 01_eda_e_modelagem.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py               # RANDOM_STATE, caminhos, constantes, definição do alvo
│   ├── data_loader.py          # carga + split estratificado
│   ├── cleaning.py             # correção de escala das colunas Grau, duplicatas
│   ├── features.py             # features derivadas
│   ├── preprocessing.py        # ColumnTransformer / pipeline
│   ├── train.py                # CV + GridSearch + comparação de modelos
│   ├── evaluate.py             # métricas, curvas, matrizes de confusão
│   └── predict.py              # inferência (usada pelo app e pelos testes)
├── app/
│   └── streamlit_app.py
├── models/
│   ├── model.joblib            # gerado na implementação
│   └── model_metadata.json
├── reports/
│   └── figures/
├── tests/
│   ├── test_cleaning.py
│   └── test_pipeline.py
└── .venv/                      # ambiente virtual local (não versionado)
```

**Estado após a Fase 1 (executada com sucesso):**

- Ambiente **Python 3.12.14** instalado via `uv` em `/home/rodrigo/.local/share/uv/python`
  (sem `sudo`, sem alterar o sistema); `.venv/` criado e dependências instaladas a partir de
  `requirements.txt`.
- Repositório Git local inicializado (branch `main`, **sem commit** — conforme a restrição do autor).
- Base copiada para `data/raw/StudentsPrepared.xlsx` com SHA-256 idêntico; o arquivo original
  em `data/StudentsPrepared.xlsx` **não** foi alterado, movido nem removido.
- Módulos implementados e executados: `config.py`, `schema.py`, `grade_scale.py`,
  `cleaning.py`, `data_loader.py`, `build_dataset.py`.
- Testes: `test_grade_scale.py`, `test_cleaning.py`, `test_schema.py`,
  `test_data_integrity.py` (+ `pytest.ini`) — **67 testes passando**.
- Artefatos gerados: `data/processed/students_processed.csv` (4.423 x 31) e
  `reports/data_quality_report.{md,json}`.
- **Correção aplicada após a primeira execução:** a checagem de consistência passou a parear
  cada nota com o `Aprovado` do **mesmo semestre** (`GRADE_APPROVED_COLUMNS`). Antes, a nota do
  2º semestre era comparada com o aprovado do 1º, o que gerava 42 falsas violações.
- **Fase 2:** implementados `features.py`, `preprocessing.py`, `evaluate.py` e `train.py`,
  com testes novos (`test_features.py` e `test_preprocessing.py`).
- **Fase 3:** implementados `select_model.py` e `predict.py`, com testes novos
  (`test_select_model.py` e `test_predict.py`) — **111 testes passando**. Artefatos:
  `models/model.joblib`, `models/model_metadata.json`, `reports/phase3_final_model.md`,
  6 figuras e 3 tabelas, e o notebook executado (`notebooks/01_eda_e_modelagem.ipynb`,
  14 células, 0 erros).
- **Fase 4:** implementados `app/streamlit_app.py` e `src/form_spec.py`, com testes novos
  (`test_form_spec.py` e `test_app.py`) — **149 testes passando**. Documentação e roteiro do
  vídeo em `docs/`.
- **Restam apenas ações do autor:** publicar no GitHub, publicar no Streamlit Community Cloud
  e gravar o vídeo (roteiro pronto em `docs/roteiro-video.md`).

---

## 19. Checklist final da entrega

**Código e reprodutibilidade**
- [x] Ambiente virtual (`.venv`) com **Python 3.12.14** e `requirements.txt` instalado.
- [x] `RANDOM_STATE` centralizado em `src/config.py`.
- [x] `Pipeline` + `ColumnTransformer` cobrindo imputação, escala e encoding.
- [x] Zero vazamento: nenhuma estatística ajustada no conjunto de teste.
- [x] Correção das colunas `...SemestreGrau` implementada e coberta por testes automatizados.
- [x] Base bruta preservada, com verificação de integridade por SHA-256.
- [x] Relatório de qualidade de dados produzido pelo pipeline. *(executado)*
- [x] Notebook `01_eda_e_modelagem.ipynb` executável de ponta a ponta no VS Code (14 células, 0 erros).

**Modelagem**
- [x] Divisão estratificada 70/30 documentada e **congelada** em disco (`train_test_split.json`).
- [x] `StratifiedKFold` (k=5) com o pipeline completo como estimador.
- [x] Comparação entre `DummyClassifier`, Regressão Logística, Random Forest e Gradient Boosting.
- [x] Cenários **Early Warning × Completo** e ablações (financeiras e macroeconômicas).
- [x] Métricas: Accuracy, Precision, Recall, F1, **F2**, ROC-AUC (classe positiva = evasão).
- [x] Matriz de confusão do modelo final.
- [x] Análise de overfitting/underfitting com curvas de aprendizado e de validação.
- [x] Avaliação do conjunto de teste **uma única vez**, com limiar congelado.
- [x] Tabela final treino × CV × teste e escolha do modelo justificada por escrito (§23).

**Deploy**
- [x] `models/model.joblib` + `model_metadata.json` gerados.
- [x] `app/streamlit_app.py` funcionando localmente (validado no navegador).
- [ ] App publicado no Streamlit Community Cloud com URL pública, preferencialmente com o modelo **Early Warning** (D10).
- [x] Teste com entradas de "risco alto" e "risco baixo" para validar a demonstração.

**Documentação e entrega**
- [x] `README.md` completo (objetivo, estrutura, como executar, resultados, limitações).
- [x] `PROJECT_PLAN.md` atualizado com os números finais.
- [x] Notebook executável com as análises principais.
- [ ] Repositório publicado no GitHub. *(pendente: `git push` do autor)*
- [x] `docs/entregas/links.txt` criado com repositório + app + vídeo. *(links a preencher)*
- [x] Roteiro do vídeo redigido em `docs/roteiro-video.md` (bloco a bloco, ~8 min).
- [ ] Vídeo de **no mínimo 5 minutos** gravado, mostrando a app deployada.

---

## 20. Decisões confirmadas pelo autor (D1–D10)

| # | Tema | Decisão confirmada | Onde foi implementado |
|---|---|---|---|
| **D1** | Alvo binário | Principal: `Desistente = 1`; `Graduado` + `Matriculado` = 0. Sensibilidade: `Desistente = 1` vs. `Graduado = 0`, excluindo `Matriculado`. Documentar a diferença entre as duas definições. | `create_target()` → colunas `target` e `target_excl_enrolled` |
| **D2** | 2º semestre | Não é vazamento automático. Implementar **Early Warning** e **Completo** e comparar. Early Warning é o preferido para deploy; Completo é o benchmark. | `EARLY_WARNING_FEATURES` / `FULL_FEATURES` em `config.py` |
| **D3** | Financeiras | Não remover automaticamente. Ablação do Early Warning **com e sem** variáveis financeiras; documentar a possível dependência temporal sem afirmar vazamento. | `EARLY_WARNING_NO_FINANCIAL_FEATURES` |
| **D4** | Colunas `...SemestreGrau` | Correção autorizada (`dividir por 10 enquanto > 20`). Original intacto; correção só no pipeline; gerar cópia processada; registrar contagens, min/max, validações e exemplos; descrever como "consistente com perda do separador decimal"; testes automatizados. | `src/grade_scale.py`, `correct_grade_columns()`, `validate_correction_by_distribution()` |
| **D5** | Base | `StudentsPrepared.xlsx` é a Base Sub Fase 3. Copiar para `data/raw` sem alterar nome/conteúdo; nunca sobrescrever o original; processados em `data/processed`. | `config.RAW_DATA_PATH`, teste de SHA-256 |
| **D6** | Ambiente | **Python 3.12**; não desenvolver em 3.14. Se indisponível, não instalar globalmente nem alterar o SO — informar os comandos ao autor. | `requirements.txt` + seção de setup do README |
| **D7** | Macro | Manter inicialmente; avaliar importância e contribuição por CV; comparar com e sem as três variáveis; remover só se não agregarem. Sem PCA. | `EARLY_WARNING_NO_MACRO_FEATURES`, `FULL_NO_MACRO_FEATURES` |
| **D8** | Otimização | **F2-score** como métrica principal; reportar Accuracy, Precision, Recall, F1, F2, ROC-AUC e matriz de confusão; recall é prioridade operacional, mas não otimizado isoladamente; thresholds avaliados na **validação**. | `PRIMARY_SCORING`, `THRESHOLDS`, `REPORTED_METRICS` |
| **D9** | Granularidade | 1 linha = 1 estudante (confirmado pela documentação original da base UCI). Duplicata exata removida após a separação bruto/processado e registrada no relatório. | README + `remove_exact_duplicates()` |
| **D10** | Deploy | Streamlit Community Cloud; publicar preferencialmente o modelo **Early Warning**; o app deixa claro que a saída é **risco**, não decisão automática. | `app/streamlit_app.py` (Fase 4) |

### 20.1 Itens ainda em aberto (não bloqueantes)

| # | Item | Situação |
|---|---|---|
| A1 | Execução do pipeline da Fase 1 | ✅ **resolvido** — Python 3.12.14 instalado; pipeline e testes executados com sucesso |
| A2 | Momento de coleta de `MensalidadesEmDia`/`Devedor` | Não confirmável a partir da base; tratado como hipótese a documentar (D3) |
| A3 | Impacto da limpeza | 3.465 valores de nota corrigidos; 1 duplicata exata removida (4.424 → 4.423 linhas). A duplicata pertencia à classe `Desistente`, então o alvo principal passou de 1.421 para **1.420** evasões. |

---

## 21. Referências internas
- Enunciado: `docs/MLET - Prova Substitutiva - Fase 3.pdf`
- Base: `data/StudentsPrepared.xlsx`

---

## 22. Resultados da Fase 2 (comparação de modelos e cenários)

Executado com `python -m src.train`. Divisão congelada: **3.096 linhas de treino** e
**1.327 de teste** — o teste **não foi avaliado** (D8).

### 22.1 Alvo principal (D1) — F2 na validação cruzada, após ajuste

| Cenário | Modelo | Features | Treino | CV | Desvio | Gap |
|---|---|---|---|---|---|---|
| Completo | `logistic_balanced` | 35 | 0,8115 | **0,7945** | 0,0181 | 0,0170 |
| Completo sem macro | `logistic_balanced` | 32 | 0,8088 | 0,7913 | 0,0191 | 0,0174 |
| Completo sem macro | `random_forest_balanced` | 32 | 0,8212 | 0,7867 | 0,0250 | 0,0345 |
| Completo | `random_forest_balanced` | 35 | 0,8236 | 0,7800 | 0,0256 | 0,0436 |
| Early Warning | `random_forest_balanced` | 24 | 0,8302 | 0,7721 | 0,0175 | 0,0581 |
| Early Warning | `logistic_balanced` | 24 | 0,7822 | **0,7712** | 0,0200 | **0,0110** |
| Early Warning sem macro | `logistic_balanced` | 21 | — | 0,7697 | — | 0,0104 |
| Early Warning sem financeiras | `logistic_balanced` | 22 | — | 0,7487 | — | 0,0062 |
| (qualquer) | `dummy_prior` | — | — | 0,0000 | — | 0,0000 |

> Valores de treino/gap omitidos onde o modelo não ficou entre os seis melhores.

### 22.2 Conclusões da Fase 2

1. **`class_weight="balanced"` é a maior alavanca isolada.** Na Regressão Logística o F2
   sobe de ~0,699 para ~0,769 — ganho de **7 p.p.**, superior a qualquer ajuste de
   hiperparâmetro.
2. **A Regressão Logística balanceada domina todos os cenários**, com a melhor combinação
   de desempenho e estabilidade (gap treino−CV de apenas ~0,01–0,02).
3. **Random Forest sofre overfitting severo com parâmetros padrão:** F2 de treino = **1,0000**
   contra **0,7419** na validação (gap 0,2581). O ajuste de `max_depth`/`min_samples_leaf`
   reduz o gap para 0,0436 e ainda **melhora** a validação (0,7419 → 0,7800).
4. **Gradient Boosting** também mostra gap alto (0,1485 no cenário Completo), mas entrega a
   maior precision (0,8421).
5. **O cenário Completo supera o Early Warning em apenas ~2,3 p.p. de F2** (0,7945 vs.
   0,7712). O custo é esperar o 2º semestre, atrasando a intervenção → mantém-se a
   recomendação de **Early Warning para o deploy** (D2), com o Completo como benchmark.
6. **Remover as variáveis financeiras piora bastante:** 0,7712 → 0,7487 (−2,25 p.p.).
   `Devedor` e `MensalidadesEmDia` carregam sinal preditivo real; **permanecem**, com a
   dependência temporal documentada como **hipótese** (D3).
7. **Remover as variáveis macroeconômicas quase não altera o resultado:** 0,7712 → 0,7697
   (−0,15 p.p.) no Early Warning e 0,7945 → 0,7913 (−0,32 p.p.) no Completo — contribuição
   **marginal** (D7), o que também elimina o risco de vazamento temporal.
8. **Alvo de sensibilidade** (`Desistente` vs. `Graduado`): desempenho bem maior
   (F2 até **0,8737**), como esperado ao remover a classe ambígua `Matriculado`.

### 22.3 Artefatos gerados

| Arquivo | Conteúdo |
|---|---|
| `reports/phase2_model_comparison.md` | Relatório completo da Fase 2 |
| `reports/tables/phase2_cv_comparison.csv` | Métricas × cenários × modelos × alvos |
| `reports/tables/phase2_tuned_comparison.csv` | Recorte do alvo principal |
| `reports/figures/phase2_*.png` | 13 figuras (F2, recall, ROC-AUC; treino vs. CV) |
| `data/processed/train_test_split.json` | Índices da divisão congelada |

### 22.4 Notas técnicas

- A divisão treino/teste foi **congelada em disco** para que a Fase 3 use exatamente o mesmo
  conjunto de teste.
- `N_JOBS = 1` por padrão: o paralelismo do `joblib` gera erros de `multiprocessing` neste
  ambiente (sandbox do VS Code). O ganho medido com `n_jobs=-1` era pequeno (~30% em Random
  Forest), o que não compensa a instabilidade.
- O aviso `OptimizeWarning: Unknown solver options: iprint` (scipy novo × scikit-learn 1.5.2)
  é benigno e foi filtrado em `src/train.py` e no `pytest.ini`.

---

## 23. Resultados da Fase 3 (modelo final e teste único)

Executado com `python -m src.select_model`.

### 23.1 Modelo final escolhido

| Item | Valor |
|---|---|
| Modelo | `LogisticRegression(class_weight="balanced", C=0.1)` |
| Cenário | **Early Warning** (24 features) — candidato ao deploy (D2) |
| Alvo | `Desistente = 1` vs. `Graduado + Matriculado = 0` (D1) |
| Limiar de decisão | **0,40** — escolhido na validação (*out-of-fold*), nunca no teste (D8) |

**Por que este modelo (seção 13, definida antes de ver o teste):** o melhor F2 da Fase 2 no
cenário Early Warning era do `random_forest_balanced` (0,7721), mas o `logistic_balanced`
ficou a menos de 1 p.p. (0,7712) **com gap treino−CV 5× menor** (0,0110 contra 0,0581).
Pela regra de desempate, escolheu-se o modelo **mais estável e interpretável**.

### 23.2 Desempenho nos três níveis (F2)

| Nível | F2 | Recall | Precision | ROC-AUC |
|---|---|---|---|---|
| Treino | 0,8070 | — | — | — |
| Validação cruzada (média) | 0,7712 | 0,7817 | 0,7336 | 0,8933 |
| **Teste (uma vez)** | **0,8136** | **0,8568** | 0,6772 | **0,9071** |

- Gap treino − CV = **0,0358** (abaixo da tolerância de 0,05)
- Gap CV − teste = **−0,0424** (o teste foi **melhor** que a validação → sem sinal de overfitting)
- **Diagnóstico: `ajuste_adequado`**

### 23.3 Matriz de confusão no teste (limiar 0,40, n = 1.327)

| | Previsto não evasão | Previsto evasão |
|---|---|---|
| **Real não evasão** | 727 (TN) | 174 (FP) |
| **Real evasão** | **61 (FN)** | **365 (TP)** |

**O efeito do limiar (a decisão de negócio):**

| Limiar | Recall | Precision | F2 | Evasões não detectadas |
|---|---|---|---|---|
| 0,50 (padrão) | 0,8122 | 0,7473 | 0,7983 | 80 |
| **0,40 (escolhido)** | **0,8568** | 0,6772 | **0,8136** | **61** |

Baixar o limiar de 0,50 para 0,40 faz o modelo **capturar 19 evasões a mais** (61 em vez de
80 não detectadas), ao custo de 57 alarmes falsos adicionais. Como o objetivo é **retenção**,
perder um evadido custa mais do que um contato desnecessário — a troca se justifica (D8).

### 23.4 Diagnóstico de overfitting / underfitting

- **Curva de aprendizado** (`phase3_curva_aprendizado.png`): as curvas de treino e validação
  convergem e o gap **diminui** conforme o volume de treino cresce → não há overfitting.
- **Curva de validação** para `C` (`phase3_curva_validacao_C.png`): mostra o ponto de virada
  em que regularização menor começa a não trazer ganho — justificando o `C=0,1` escolhido.
- **Contraste didático:** o `RandomForest` padrão da Fase 2 (F2 de treino = 1,0000 contra
  0,7419 na validação) é o caso de overfitting documentado; a Regressão Logística
  balanceada não apresenta o problema.

### 23.5 Artefatos gerados

| Arquivo | Conteúdo |
|---|---|
| `models/model.joblib` | Pipeline completo (engineering + pré-processamento + modelo) |
| `models/model_metadata.json` | Metadados: limiar, features, métricas, ambiente, diagnóstico |
| `reports/phase3_final_model.md` | Relatório da Fase 3 |
| `reports/figures/phase3_*.png` | 6 figuras (confusão, ROC, PR, limiar, aprendizado, validação) |
| `reports/tables/phase3_*.csv` | 3 tabelas (limiar, curva de aprendizado, curva de validação) |
| `notebooks/01_eda_e_modelagem.ipynb` | Notebook executado (14 células, 0 erros) |

### 23.6 Inferência

`src/predict.py` expõe `predict_risk()`, usada pelo app e pelos testes. Verificação manual:

| Perfil | Probabilidade | Faixa |
|---|---|---|
| Sinais de risco (devedor, sem bolsa, 0 aprovações) | **98,7%** | Alto |
| Bom desempenho (em dia, bolsista, 6 aprovações) | **10,6%** | Baixo |

> O resultado é apresentado sempre como **risco** e acompanhado de uma recomendação de ação —
nunca como decisão automática sobre o estudante (D10).

---

## 24. Entrega final (Fase 4)

### 24.1 Aplicação Streamlit

| Arquivo | Papel |
|---|---|
| `app/streamlit_app.py` | Interface: formulário, resultado e transparência do modelo |
| `src/form_spec.py` | Especificação dos campos (testável sem subir a interface) |
| `.streamlit/config.toml` | Tema e execução headless |

**Decisões de interface (D10):**

- O resultado é sempre apresentado como **risco**, acompanhado de uma recomendação de ação e
  de um aviso explícito de que **não é uma decisão** sobre o estudante.
- O formulário é montado a partir do `model_metadata.json`: aparecem **apenas** os campos que
  o modelo usa. Se o cenário mudasse para *Completo*, o 2º semestre surgiria automaticamente.
- A aplicação **não treina** nem reimplementa transformações: consome `src.predict.predict_risk`,
  que carrega o `Pipeline` completo do `.joblib`.
- Dois botões de exemplo (risco alto / risco baixo) facilitam a demonstração no vídeo.
- Se o modelo não existir, o app exibe instruções em vez de falhar.

**Validação em execução (navegador):**

| Perfil | Probabilidade | Faixa |
|---|---|---|
| Devedor, mensalidades em atraso, sem bolsa, 0 aprovações | **98,7%** | 🔴 Alto |
| Mensalidades em dia, bolsista, 6 aprovações, média 15,5 | **10,1%** | 🟢 Baixo |

### 24.2 Documentação e entrega

| Arquivo | Conteúdo |
|---|---|
| `docs/roteiro-video.md` | Roteiro do vídeo em 8 blocos cronometrados, com falas sugeridas e os números reais |
| `docs/entregas/links.txt` | Arquivo `.txt` exigido no enunciado (repositório + app + vídeo) |
| `reports/figures/app_streamlit_risco_baixo.png` | Captura da aplicação em execução |

### 24.3 Deploy (Streamlit Community Cloud)

1. `git push` do repositório para o GitHub.
2. <https://share.streamlit.io> → **New app** → selecionar o repositório.
3. *Main file path*: `app/streamlit_app.py`.
4. **Deploy** e copiar a URL pública para `docs/entregas/links.txt`.

> O modelo está versionado (`models/model.joblib`), então o deploy não treina nada e
> inicia em segundos. `requirements.txt` já inclui o Streamlit.

### 24.4 Ações pendentes do autor

| # | Ação |
|---|---|
| 1 | Revisar e subir o repositório para o GitHub (`git push`) |
| 2 | Publicar no Streamlit Community Cloud e registrar a URL |
| 3 | Gravar o vídeo seguindo `docs/roteiro-video.md` |
| 4 | Preencher os três links em `docs/entregas/links.txt` |

---

