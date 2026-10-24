# Tech Challenge Fase 4 — Voice of Customer (RAG)

Pipeline de preparação de dados **Voice of Customer (VoC)** sobre o dataset público **Olist Brazilian E-commerce**, estruturado em camadas (Bronze → Silver → Gold) para alimentar um sistema de **RAG (Retrieval-Augmented Generation)** de análise de reviews de clientes.

## Arquitetura

```
Bronze (db_bronze/, parquets Olist)
   │
   ├── run_silver ──────► Silver (S3 db_silver/, 9 tabelas tratadas)
   │
   └── run_gold ────────► Gold (S3 db_gold/, 4 tabelas interligadas para RAG)
```

## Estrutura do projeto

| Caminho | Descrição |
| --- | --- |
| `data_preparing/db_bronze/` | Dados brutos Olist (9 parquets: orders, reviews, customers, products, items, payments, sellers, geolocation, tradução de categorias) |
| `data_preparing/run_silver` | Notebook Bronze → Silver: conversão de tipos, deduplicação (reviews), flags derivadas (`has_title`, `has_message`, `is_late_delivery`) e `delivery_delay_days` |
| `data_preparing/bronze_manifest` | Notebook que gera o `manifest.json` com estatísticas da camada Bronze |
| `data_preparing/run_gold` | Notebook Silver → Gold: monta as 4 tabelas interligadas do corpus RAG |
| `data_preparing/test_gold_contract.py` | Testes de contrato da camada Gold (schemas, PKs/FKs, integridade) |
| `data_preparing/gold_validation_and_rag_simulation.pdf` | Relatório de validação da Gold e simulação RAG |

## Camada Gold (4 tabelas interligadas)

| Tabela | Chave | Conteúdo |
| --- | --- | --- |
| `gold_reviews` | `review_id` (PK), `order_id` (FK) | Corpus principal do RAG: `document_text` (título + mensagem), sentimento, `text_length`, ano/mês da review |
| `gold_orders` | `order_id` (PK), `customer_id` (FK) | `order_status`, atraso de entrega, data de compra |
| `gold_customers` | `customer_id` (PK) | Estado e cidade do cliente |
| `gold_products_by_order` | `order_id` (PK) | Categorias de produtos por pedido (PT/EN) |

Relacionamentos: `gold_reviews.order_id → gold_orders.order_id → gold_customers.customer_id` e `gold_reviews.order_id → gold_products_by_order.order_id`.

## Configuração

1. Copie `.env.example` para `.env` e preencha as credenciais AWS e o bucket S3 (os dados Silver/Gold são gravados em S3).
2. Os notebooks rodam em **Databricks** (contas serverless suportadas).

## Como executar

1. `bronze_manifest` — gera o manifesto da Bronze (opcional, para auditoria)
2. `run_silver` — processa Bronze → Silver
3. `run_gold` — processa Silver → Gold
4. `pytest data_preparing/test_gold_contract.py` — valida o contrato da Gold

> O arquivo `.env` contém credenciais e **não** é versionado (ver `.gitignore`).
