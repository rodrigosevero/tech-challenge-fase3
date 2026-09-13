# Roteiro do vídeo — Tech Challenge Fase 3

**Duração alvo:** ~8 minutos (o enunciado exige **mínimo de 5**)
**Formato sugerido:** gravação de tela com sua voz + webcam pequena no canto (opcional)

> Este roteiro usa os **números reais** do seu projeto. As falas estão entre aspas apenas
> como sugestão — fale com suas palavras, fica mais natural. O importante é **mostrar**, não
> apenas dizer.

---

## ✅ Antes de começar a gravar

Deixe aberto, nesta ordem (abas do VS Code + navegador):

| # | O que abrir | Para quê |
|---|---|---|
| 1 | `README.md` | tela inicial |
| 2 | `notebooks/01_eda_e_modelagem.ipynb` | gráficos da EDA e dos resultados |
| 3 | `reports/phase2_model_comparison.md` | tabela da comparação de modelos |
| 4 | `reports/figures/phase2_f2_treino_vs_cv.png` | caso de overfitting |
| 5 | `reports/figures/phase3_matriz_confusao_teste.png` | matriz de confusão final |
| 6 | `reports/phase3_final_model.md` | relatório final |
| 7 | App Streamlit rodando (`streamlit run app/streamlit_app.py`) | demonstração |

**Prepare a demo antes de gravar:** suba o app e já tenha clicado em "Perfil de risco alto"
uma vez, para não perder tempo esperando o carregamento.

---

## 🎬 Bloco a bloco

### 0:00 – 0:35 · Abertura e problema

**Na tela:** `README.md`, seção "Objetivo do projeto".

> "Olá, sou o Rodrigo. Este é o Tech Challenge da Fase 3 da Pós em Machine Learning
> Engineering da FIAP.
>
> O desafio é construir uma pipeline de Machine Learning para prever **evasão de
> estudantes** — ou seja, identificar quais alunos têm risco de abandonar o curso.
>
> Por que isso importa? Porque a instituição pode agir **antes**: oferecer tutoria, apoio
> financeiro ou acompanhamento. E por isso a métrica que mais importa aqui **não é acurácia**:
> é o **recall da evasão** — eu quero capturar o máximo de alunos em risco."

💡 **Dica:** mencione já no início que o resultado é uma **estimativa de risco**, não uma
decisão automática.

---

### 0:35 – 1:30 · A base de dados

**Na tela:** notebook, células da seção 1 e 2 (gráfico das classes).

> "A base é a 'Base Sub Fase 3', com **4.424 alunos e 28 colunas** — dados socioeconômicos,
> de ingresso e de desempenho acadêmico.
>
> O ponto de atenção: a coluna que eu quero prever tem **três classes** — *Graduado*,
> *Desistente* e *Matriculado* — mas o enunciado pede classificação **binária**.
>
> Então eu defini o alvo assim: **Desistente = 1**, e **Graduado + Matriculado = 0**.
> Isso deixa **1.420 evasões** contra 3.003 não evasões — **32% de positivos**. É uma base
> desbalanceada, o que exige estratificação na divisão e na validação cruzada."

💡 **Se sobrar tempo:** comente que também rodei uma análise de sensibilidade excluindo
"Matriculado" (a classe ambígua) e que os resultados ficam **melhores** (F2 de 0,87), como
esperado.

---

### 1:30 – 2:30 · 🎯 O achado mais interessante (não pule este!)

**Na tela:** notebook, seção 3 — a tabela de correções e o histograma.

> "Aqui está o problema mais curioso que encontrei. Duas colunas de notas — as do primeiro e
> do segundo semestre — tinham valores **impossíveis**. Olha este: **13 quatrilhões**.
>
> Investigando, o padrão era **consistente com a perda do separador decimal**: a nota
> original de 10,375 tinha virado 10375, e uma nota de 13,43 tinha virado 1,34 vezes 10 à
> décima sexta.
>
> Criei uma regra simples — dividir por 10 enquanto o valor passar de 20 — e ela recuperou
> **3.465 valores** para a faixa correta de 0 a 20.
>
> E não aceitei isso de olho: validei de três formas. Primeiro, **100% dos valores corrigidos
> caíram na escala de 0 a 20**. Segundo, **todos os valores corrompidos eram números inteiros
> exatos** — exatamente o que a hipótese prevê. Terceiro, a distribuição das notas corrigidas
> ficou **muito parecida** com a das notas que nunca estavam corrompidas.
>
> E o mais importante: o **arquivo original nunca foi alterado** — a correção acontece só no
> pipeline, e um teste automatizado verifica o SHA-256 do arquivo para garantir isso."

---

### 2:30 – 3:30 · Feature engineering e prevenção de vazamento

**Na tela:** `src/preprocessing.py` e `src/features.py` (role o código) + `PROJECT_PLAN.md`.

> "Para tratar os dados, monto um **Pipeline** com **ColumnTransformer**: as variáveis
> **numéricas** passam por imputação pela mediana e padronização; as **categóricas** por
> imputação e *one-hot encoding*, com agrupamento de categorias raras — porque 'Nacionalidade'
> tinha 21 valores, muitos com 1 ou 2 casos.
>
> O ponto crítico é **evitar vazamento de dados**: toda transformação que 'aprende' algo —
> a mediana, a média, as categorias — é ajustada **somente no treino**, dentro de cada fold da
> validação cruzada. O conjunto de teste nunca participa.
>
> Também criei variáveis derivadas, como **taxa de aprovação** e **disciplinas não aprovadas**,
> que são contas linha a linha — não usam estatística de ninguém, então não vazam informação."

---

### 3:30 – 4:30 · Modelagem e os dois cenários

**Na tela:** `reports/phase2_model_comparison.md`, tabela do topo.

> "Comparei **6 modelos**: um *baseline* burro que sempre chuta a classe mais comum, Regressão
> Logística, Random Forest e Gradient Boosting — e versões com `class_weight` balanceado, que
> só existe para os modelos que suportam.
>
> E comparei em **5 cenários**. Os dois principais são:
>
> **Early Warning:** só dados de cadastro, socioeconômicos, financeiros, ingresso e primeiro
> semestre. É o cenário que permite **intervir cedo** — e é o meu candidato ao deploy.
>
> **Completo:** acrescenta o segundo semestre. Serve como *benchmark*.
>
> O resultado principal: o cenário Completo supera o Early Warning em **apenas 2,3 pontos
> percentuais** de F2. Ou seja, esperar um semestre inteiro a mais traz muito pouco ganho —
> por isso escolhi o Early Warning."

💡 **Frase de impacto:** "A **Regressão Logística com `class_weight` balanceado** foi a maior
alavanca de todo o projeto: sozinha, subiu o F2 em **7 pontos percentuais**. Mais do que
qualquer ajuste de hiperparâmetro."

---

### 4:30 – 5:30 · 🔥 O caso de overfitting (ótimo momento didático)

**Na tela:** `reports/figures/phase2_f2_treino_vs_cv.png` — gráfico de barras treino vs. validação.

> "Aqui está um exemplo claro de **overfitting**.
>
> O Random Forest, com os parâmetros padrão, tem F2 de **1,0000 no treino** — perfeito! — mas
> só **0,7419 na validação cruzada**. Ele **decorou** os dados de treino.
>
> Depois de ajustar a profundidade máxima e o mínimo de amostras por folha, o gap caiu de
> **0,26 para 0,04**, e a validação ainda **melhorou** para 0,78. Ou seja: regularizar o
> modelo não só corrigiu o overfitting — ele ficou **melhor** em dados novos.
>
> Isso é exatamente o que o enunciado pede: analisar e **justificar** se houve overfitting.
> Houve, eu mostrei qual foi, e mostrei a correção."

---

### 5:30 – 6:15 · O modelo final e as métricas no teste

**Na tela:** `reports/phase3_final_model.md` e `phase3_matriz_confusao_teste.png`.

> "O modelo final foi a **Regressão Logística balanceada**, no cenário **Early Warning**.
>
> Escolhi ela e não o Random Forest por um critério que defini **antes** de olhar o teste: se
> dois modelos empatam tecnicamente, fico com o **mais estável e mais interpretável**. O
> Random Forest tinha F2 de 0,7721 e a Logística 0,7712 — uma diferença de 0,09 ponto — mas o
> gap treino-validação da Logística era **cinco vezes menor**.
>
> O conjunto de teste — **1.327 alunos** — foi avaliado **uma única vez**. E olha que
> interessante: o desempenho no teste ficou **melhor** que na validação cruzada.
>
> **Recall de 0,857**, F2 de 0,814, ROC-AUC de 0,907. O recall significa que o modelo
> identifica cerca de **86 de cada 100 alunos** que realmente evadem.
>
> E o diagnóstico de ajuste? **Ajuste adequado.** O gap entre treino e validação foi de 0,036
> — abaixo da tolerância de 0,05 que eu tinha definido. Não há overfitting no modelo final."

---

### 6:15 – 6:45 · A decisão de negócio do limiar

**Na tela:** `reports/figures/phase3_metricas_vs_limiar.png`.

> "Um detalhe que eu considero o mais importante do ponto de vista de negócio: o **limiar de
> decisão**.
>
> Por padrão, classificamos como evasão quando a probabilidade passa de 50%. Mas eu escolhi o
> limiar **na validação**, nunca no teste, e o melhor valor foi **0,40**.
>
> O efeito disso é bem concreto: com o limiar padrão, o modelo deixava passar **80 evasões**.
> Com 0,40, deixou passar **61**. Ou seja, **19 alunos a mais** entram no radar de retenção.
>
> O preço disso são 57 alarmes falsos adicionais. Mas pensando em retenção, **perder um aluno
> custa mais caro** do que fazer um contato desnecessário. Essa troca foi uma decisão
> consciente, e é o tipo de decisão que o dado informa, mas o negócio decide."

---

### 6:45 – 7:45 · 🖥️ Demonstração da aplicação

**Na tela:** app Streamlit rodando (`streamlit run app/streamlit_app.py`).

> "E aqui está a aplicação, publicada no Streamlit.
>
> Ela recebe os dados de um estudante — cadastro, situação financeira, ingresso e desempenho
> do primeiro semestre — e devolve a **probabilidade de evasão**.
>
> Vou usar o botão de exemplo de **risco alto**: devedor, mensalidades em atraso, sem bolsa,
> zero disciplinas aprovadas no primeiro semestre.
>
> Olha o resultado: **98,7%** de probabilidade, faixa de **risco alto**, com a recomendação de
> priorizar o contato.
>
> Agora o oposto: aluno em dia com as mensalidades, bolsista, 6 disciplinas aprovadas e média
> 15,5.
>
> **10,1%** — risco baixo, sem alerta.
>
> E repare nesta mensagem que a aplicação sempre exibe: ela deixa explícito que isto é uma
> **estimativa de risco gerada por um modelo estatístico**, que **não é uma decisão** sobre o
> estudante e não deve ser usada isoladamente. A ideia é **priorizar** o trabalho humano de
> acompanhamento — nunca substituí-lo."

---

### 7:45 – 8:15 · Conclusões

**Na tela:** `README.md` ou o final do notebook.

> "Para fechar, três conclusões:
>
> **Primeiro:** o maior ganho de desempenho não veio de um modelo sofisticado, e sim de
> **tratar o desbalanceamento** com `class_weight` balanceado, e de **cuidar dos dados** —
> a correção das 3.465 notas.
>
> **Segundo:** escolhi um modelo **simples e explicável** porque ele generaliza tão bem quanto
> os complexos, e é muito mais estável. Nem sempre o modelo mais sofisticado é a melhor
> escolha.
>
> **Terceiro:** o resultado é útil para **priorizar** ações de retenção, mas tem limitações —
> as variáveis financeiras podem ser consequência da evasão e não causa, e isso está
> documentado como hipótese no projeto.
>
> Obrigado!"

---

## 📋 Checklist rápido de gravação

- [ ] Microfone testado (grave 10 segundos e ouça)
- [ ] Abas abertas e app já carregado
- [ ] Não mostrar senhas, tokens ou dados pessoais reais
- [ ] Falar o nome e o contexto do desafio no início
- [ ] Mostrar **o app funcionando** (não só slides)
- [ ] Passar de 5 minutos com folga (alvo: 7 a 8)
- [ ] Assistir uma vez antes de enviar

## 🗣️ Frases de segurança (caso trave)

- "Estratifiquei a divisão porque a base é desbalanceada, com 32% de positivos."
- "Ajustei tudo dentro do pipeline para não vazar informação do teste."
- "Avaliei o teste uma única vez, no final, com o limiar já congelado."
- "O diagnóstico de ajuste comparou treino, validação cruzada e teste."
