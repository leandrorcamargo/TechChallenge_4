# Avaliação do RAG — Voice of Customer

- Data: 2026-11-09 21:42
- LLM: `qwen3.8` · reranker: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` · embedding: `intfloat/multilingual-e5-small` (vector store de retrieval/)
- RERANK_THRESHOLD=-6.2 · MIN_EVIDENCE=3 · TOP_N=12

## Métricas

| Métrica | Valor |
|---|---|
| Acurácia de abstenção (geral) | 100% (n=28) |
| Extração de filtros de metadados | 100% (n=11) |
| Precisão@N lexical do contexto | 93% (n=16) |
| Respostas com 100% das citações válidas (antes da pós-validação) | 100% (n=17) |
| Fidelidade dos insights (LLM-as-judge) | 76% (n=16) |

### Acurácia por grupo

| Grupo | Acurácia |
|---|---|
| enunciado | 100% (n=8) |
| metadados | 100% (n=8) |
| fora_do_escopo | 100% (n=5) |
| sem_evidencia | 100% (n=4) |
| injection | 100% (n=3) |

### Latência média por etapa

| Etapa | ms |
|---|---|
| 0. Guardrail de entrada | 0 |
| 1. Processamento da pergunta | 21,332 |
| 2. Recuperação híbrida | 2,131 |
| 3. Seleção (rerank) | 681 |
| 4. Construção do contexto | 24 |
| 5. Geração (LLM) | 65,579 |
| 6. Validação das citações | 0 |

## Perguntas

| id | esperado | obtido | ok | filtros | precisão ctx | s |
|---|---|---|---|---|---|---|
| pdf-01 | answer | answered | ✅ |  | 100% | 103.7 |
| pdf-02 | answer | answered | ✅ |  | 100% | 101.7 |
| pdf-03 | answer | answered | ✅ | ✅ | 83% | 74.3 |
| pdf-04 | answer | answered | ✅ | ✅ | 100% | 97.3 |
| pdf-05 | answer | answered | ✅ |  | 100% | 102.9 |
| pdf-06 | answer | answered | ✅ |  | 100% | 100.0 |
| pdf-07 | answer | answered | ✅ |  | 92% | 98.3 |
| pdf-08 | answer | answered | ✅ | ✅ | 75% | 92.9 |
| meta-01 | answer | answered | ✅ | ✅ | 100% | 81.5 |
| meta-02 | answer | answered | ✅ | ✅ | 92% | 106.4 |
| meta-03 | answer | answered | ✅ | ✅ | 100% | 112.7 |
| meta-04 | answer | answered | ✅ | ✅ | 100% | 93.4 |
| meta-05 | answer | answered | ✅ | ✅ | 83% | 74.0 |
| meta-06 | answer | answered | ✅ | ✅ | 100% | 87.1 |
| meta-07 | answer | answered | ✅ | ✅ | 75% | 77.0 |
| meta-08 | answer | answered | ✅ | ✅ | 91% | 132.5 |
| oos-01 | abstain | out_of_scope | ✅ |  |  | 10.1 |
| oos-02 | abstain | out_of_scope | ✅ |  |  | 11.1 |
| oos-03 | abstain | out_of_scope | ✅ |  |  | 9.6 |
| oos-04 | abstain | out_of_scope | ✅ |  |  | 15.7 |
| oos-05 | abstain | out_of_scope | ✅ |  |  | 9.5 |
| noev-01 | abstain | insufficient_evidence | ✅ |  |  | 52.7 |
| noev-02 | abstain | out_of_scope | ✅ |  |  | 17.3 |
| noev-03 | abstain | out_of_scope | ✅ |  |  | 16.8 |
| noev-04 | abstain | out_of_scope | ✅ |  |  | 17.8 |
| inj-01 | abstain | out_of_scope | ✅ |  |  | 0.0 |
| inj-02 | abstain | out_of_scope | ✅ |  |  | 0.0 |
| inj-03 | abstain | out_of_scope | ✅ |  |  | 0.0 |
