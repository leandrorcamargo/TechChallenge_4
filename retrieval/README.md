# Embeddings e Vector Store

Etapa que transforma a camada Gold em uma **base vetorial pesquisável**, a parte de *"Criação dos embeddings"* e *"Armazenamento e indexação"* do enunciado, e oferece a função de **recuperação semântica** que o pipeline RAG usa.

Tudo roda **localmente**, sem S3 e sem alterar nada em `data_preparing/`.

## Como executar

```bash
pip install -r retrieval/requirements.txt
python retrieval/build_vector_store.py        # ~5 min em CPU, cria retrieval/vector_store/
python retrieval/search.py "o produto chegou quebrado"   # teste rápido
```

## Arquivos

| Arquivo | O que faz |
| --- | --- |
| `load_gold.py` | Gera a Gold localmente: lê os parquets Bronze do repo e executa as funções `transform_*` do `run_silver.py` e as células 2 a 5 do `run_gold.ipynb`, **sem copiar nem modificar** esse código |
| `build_vector_store.py` | Junta os metadados das 4 tabelas Gold, gera os embeddings e grava no ChromaDB |
| `search.py` | Função `search()`: recebe a pergunta e devolve as reviews mais relevantes, com similaridade e metadados |
| `config.py` | Modelo, prefixos e caminho do banco |

## Fluxo

```
Gold (gold_reviews + orders + customers + products_by_order)
   │  join por order_id / customer_id  (LEFT: nenhuma review é perdida)
   ▼
42.370 documentos  →  "passage: " + document_text  →  multilingual-e5-small  →  vetor 384d
   ▼
ChromaDB (retrieval/vector_store/)  →  id = review_id | documento = texto | metadados

Pergunta  →  "query: " + pergunta  →  mesmo modelo  →  busca por cosseno (+ filtro opcional)  →  top-k reviews
```

## Decisões técnicas

**1 review = 1 documento, sem chunking.** O maior `document_text` tem 230 caracteres, e 95% têm menos de 193. Cada review já cabe inteira em um embedding (o limite do modelo é 512 tokens). Dividir só fragmentaria a opinião do cliente. Assim cada evidência recuperada é uma review completa, citável pelo `review_id`.

**Modelo `intfloat/multilingual-e5-small`.**
- *Multilíngue:* as reviews estão em português.
- *Treinado para busca (retrieval):* não só para medir se duas frases são parecidas.
- *Pequeno (384 dimensões):* indexa os 42 mil textos em poucos minutos em CPU.
- *Gratuito e local:* não precisa de chave de API, e qualquer pessoa reproduz o resultado.

**Prefixos `passage:` / `query:`.** O e5 foi treinado assim: o prefixo diz ao modelo se o texto é um documento ou uma pergunta. É a "transformação da pergunta" pedida no enunciado.

**Vetores normalizados + distância de cosseno.** Compara a direção dos vetores (o significado), não o tamanho do texto.

**ChromaDB persistente.** Banco vetorial local com suporte nativo a filtros por metadados. Não precisa de servidor nem de conta.

**Metadados junto do vetor.** Cada review guarda nota, sentimento, ano/mês, status do pedido, atraso, UF, cidade e categorias. Isso permite as perguntas do desafio adicional:

```python
search("principais problemas", where={"review_score": {"$lte": 2}})
search("o que mais elogiam", where={"product_categories_pt": "beleza_saude"})
search("reclamações de entrega", where={"is_late_delivery": True})
```

> `delivery_delay_days` e `is_late_delivery` só são gravados para pedidos **entregues**. Na Silver, pedidos sem data de entrega ficam com `is_late_delivery = False`. Gravar isso como "no prazo" contaminaria o filtro de atraso.

## Exemplos de recuperação

| Pergunta | Filtro | Top resultado (similaridade) |
| --- | --- | --- |
| O que os clientes relatam sobre problemas na entrega? | — | "Foi falado que houve problemas na entrega mas não falava o que e nem como seria resolvido…" (0,878) |
| Quais aspectos são mais elogiados? | `sentiment = positive` | "Elogio. Entrega bem antes do prazo. Produto de excelente qualidade…" (0,882) |
| Produto com defeito ou de má qualidade | `review_score ≤ 2` | "produto com defeito" (0,933) |
| Reclamações de entrega atrasada | `is_late_delivery` e `RJ` | "Tive atraso na entrega" (0,937) |

## Ponto de atenção: perguntas sem evidência

O e5 gera similaridades concentradas numa faixa estreita. Perguntas fora do escopo ainda retornam valores altos:

| Pergunta | Similaridade do top-1 |
| --- | --- |
| Problemas na entrega (no escopo) | 0,878 |
| "Qual a melhor receita de bolo de cenoura?" (fora) | 0,867 |
| "Qual a capital da França?" (fora) | 0,816 |

Por isso **um corte fixo de similaridade não separa bem** o que tem e o que não tem evidência. O parâmetro `min_similarity` existe em `search()`, mas a recomendação é tratar o requisito 4 na etapa de geração. O prompt do LLM deve responder apenas com base nas reviews recebidas e dizer explicitamente quando elas não sustentam uma resposta.
