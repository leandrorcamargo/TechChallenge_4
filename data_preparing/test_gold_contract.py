"""
Contract tests da camada Gold (Tech Challenge Fase 4).

Validam o contrato das 4 tabelas Gold exportadas para o S3 pelo notebook
`data_preparing/run_gold`, lendo os .parquet diretamente do bucket:

- gold_reviews            (PK review_id, FK order_id)
- gold_orders             (PK order_id,  FK customer_id)
- gold_customers           (PK customer_id)
- gold_products_by_order   (PK order_id)

E tambem o corpus VoC exportado como JSONL (voc_reviews.jsonl).

Executar:  python -m pytest test_gold_contract.py -v
"""

import io
import json
import os

import boto3
import pandas as pd
import pytest
from dotenv import load_dotenv

# ------------------------------------------------------------------
# Configuracao (le credenciais do .env na raiz do projeto — um nivel acima)
# ------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENV_PATH = os.path.join(_PROJECT_ROOT, ".env")
if os.path.exists(_ENV_PATH):
    load_dotenv(_ENV_PATH)

S3_BUCKET = os.environ.get("S3_BUCKET", "amzn-s3-fiap-tech-4")
GOLD_PREFIX = os.environ.get("GOLD_PREFIX", "db_gold/")


# ------------------------------------------------------------------
# Contrato esperado — colunas exatas de cada tabela Gold
# ------------------------------------------------------------------
GOLD_REVIEWS_COLUMNS = {
    "review_id": "object",
    "order_id": "object",
    "document_text": "object",
    "text_length": "int",
    "review_score": "int",
    "has_title": "bool",
    "has_message": "bool",
    "review_creation_date": "datetime",
    "review_answer_timestamp": "datetime",
    "sentiment": "object",
    "review_year": "int",
    "review_month": "int",
}

GOLD_ORDERS_COLUMNS = {
    "order_id": "object",
    "customer_id": "object",
    "order_status": "object",
    "order_purchase_timestamp": "datetime",
    "delivery_delay_days": "float",
    "is_late_delivery": "bool",
}

GOLD_CUSTOMERS_COLUMNS = {
    "customer_id": "object",
    "customer_state": "object",
    "customer_city": "object",
}

GOLD_PRODUCTS_COLUMNS = {
    "order_id": "object",
    "product_categories_pt": "object",
    "product_categories_en": "object",
    "num_categories": "int",
}

VALID_SENTIMENTS = {"positive", "neutral", "negative"}

VALID_ORDER_STATUS = {
    "delivered", "invoiced", "shipped", "processing",
    "unavailable", "canceled", "created", "approved",
}

# sentiment esperado para cada review_score (regra de negocio do Gold)
SENTIMENT_BY_SCORE = {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}


# ------------------------------------------------------------------
# Fixtures — le as tabelas Gold do S3 uma unica vez por execucao
# ------------------------------------------------------------------
@pytest.fixture(scope="module")
def s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_REGION", "us-east-2"),
    )


def _read_parquet(s3, table_name):
    key = f"{GOLD_PREFIX}{table_name}.parquet"
    resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return pd.read_parquet(io.BytesIO(resp["Body"].read()))


@pytest.fixture(scope="module")
def gold_reviews(s3_client):
    return _read_parquet(s3_client, "gold_reviews")


@pytest.fixture(scope="module")
def gold_orders(s3_client):
    return _read_parquet(s3_client, "gold_orders")


@pytest.fixture(scope="module")
def gold_customers(s3_client):
    return _read_parquet(s3_client, "gold_customers")


@pytest.fixture(scope="module")
def gold_products_by_order(s3_client):
    return _read_parquet(s3_client, "gold_products_by_order")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def assert_schema(df, expected, table_name):
    """Confere colunas exatas (nomes e ordem) e tipos genericos."""
    assert list(df.columns) == list(expected), (
        f"{table_name}: colunas divergem do contrato.\n"
        f"  esperado: {list(expected)}\n"
        f"  atual:    {list(df.columns)}"
    )
    for col, kind in expected.items():
        dtype = str(df[col].dtype)
        ok = dtype.startswith(kind)
        assert ok, f"{table_name}.{col}: tipo esperado {kind}, obtido {dtype}"


# ==================================================================
# 1. gold_reviews — corpus principal do RAG
# ==================================================================
class TestGoldReviews:
    def test_schema_contracts_exact_columns(self, gold_reviews):
        """gold_reviews deve ter exatamente as 12 colunas do contrato, na ordem."""
        assert_schema(gold_reviews, GOLD_REVIEWS_COLUMNS, "gold_reviews")

    def test_review_id_unique_and_not_null(self, gold_reviews):
        """review_id e PK: sem nulos e sem duplicatas."""
        assert gold_reviews["review_id"].notna().all()
        assert gold_reviews["review_id"].is_unique, (
            f"{gold_reviews['review_id'].duplicated().sum()} review_ids duplicados"
        )

    def test_document_text_present_and_non_empty(self, gold_reviews):
        """Todo documento do corpus precisa de texto nao vazio (regra: 1 review com texto = 1 doc)."""
        text = gold_reviews["document_text"]
        assert text.notna().all(), f"{text.isna().sum()} document_text nulos"
        assert (text.str.len() > 0).all(), "existem document_text vazios"

    def test_text_length_matches_document_text(self, gold_reviews):
        """text_length deve ser o comprimento exato de document_text."""
        mismatches = (gold_reviews["text_length"] != gold_reviews["document_text"].str.len()).sum()
        assert mismatches == 0, f"{mismatches} linhas com text_length inconsistente"

    def test_review_score_in_range(self, gold_reviews):
        """review_score deve estar entre 1 e 5."""
        assert gold_reviews["review_score"].between(1, 5).all()

    def test_sentiment_values_valid(self, gold_reviews):
        """sentiment so pode ser positive / neutral / negative."""
        invalid = ~gold_reviews["sentiment"].isin(VALID_SENTIMENTS)
        assert invalid.sum() == 0, gold_reviews.loc[invalid, "sentiment"].unique()

    def test_sentiment_consistent_with_score(self, gold_reviews):
        """sentiment deve seguir a regra: 1-2 negative, 3 neutral, 4-5 positive."""
        expected = gold_reviews["review_score"].map(SENTIMENT_BY_SCORE)
        mismatches = (gold_reviews["sentiment"] != expected).sum()
        assert mismatches == 0, f"{mismatches} sentiments inconsistentes com o score"

    def test_year_month_derived_from_answer_timestamp(self, gold_reviews):
        """review_year/review_month devem derivar de review_answer_timestamp."""
        ts = gold_reviews["review_answer_timestamp"]
        assert (gold_reviews["review_year"] == ts.dt.year).all()
        assert (gold_reviews["review_month"] == ts.dt.month).all()
        assert gold_reviews["review_month"].between(1, 12).all()

    def test_has_title_or_has_message(self, gold_reviews):
        """Cada review precisa ter titulo OU mensagem (criterio de inclusao)."""
        assert (gold_reviews["has_title"] | gold_reviews["has_message"]).all()


# ==================================================================
# 2. gold_orders — contexto do pedido
# ==================================================================
class TestGoldOrders:
    def test_schema_contracts_exact_columns(self, gold_orders):
        """gold_orders deve ter exatamente as 6 colunas do contrato."""
        assert_schema(gold_orders, GOLD_ORDERS_COLUMNS, "gold_orders")

    def test_order_id_unique_and_not_null(self, gold_orders):
        """order_id e PK: sem nulos e sem duplicatas."""
        assert gold_orders["order_id"].notna().all()
        assert gold_orders["order_id"].is_unique

    def test_order_status_valid(self, gold_orders):
        """order_status deve conter apenas os 8 status conhecidos da Olist."""
        invalid = ~gold_orders["order_status"].isin(VALID_ORDER_STATUS)
        assert invalid.sum() == 0, gold_orders.loc[invalid, "order_status"].unique()

    def test_is_late_delivery_consistent_with_delay(self, gold_orders):
        """is_late_delivery deve ser True exatamente quando delivery_delay_days > 0."""
        expected = gold_orders["delivery_delay_days"] > 0
        mismatches = (gold_orders["is_late_delivery"] != expected).sum()
        assert mismatches == 0, f"{mismatches} linhas com is_late_delivery inconsistente"

    def test_customer_id_not_null(self, gold_orders):
        """Toda ordem deve referenciar um cliente (FK customer_id)."""
        assert gold_orders["customer_id"].notna().all()


# ==================================================================
# 3. gold_customers — contexto do cliente
# ==================================================================
class TestGoldCustomers:
    def test_schema_contracts_exact_columns(self, gold_customers):
        """gold_customers deve ter exatamente as 3 colunas do contrato."""
        assert_schema(gold_customers, GOLD_CUSTOMERS_COLUMNS, "gold_customers")

    def test_customer_id_unique_and_not_null(self, gold_customers):
        """customer_id e PK: sem nulos e sem duplicatas."""
        assert gold_customers["customer_id"].notna().all()
        assert gold_customers["customer_id"].is_unique

    def test_customer_state_is_uf(self, gold_customers):
        """customer_state deve ser sigla de UF com 2 caracteres."""
        states = gold_customers["customer_state"]
        assert states.notna().all()
        assert (states.str.len() == 2).all()

    def test_customer_city_not_null(self, gold_customers):
        """Toda linha deve ter cidade preenchida."""
        assert gold_customers["customer_city"].notna().all()


# ==================================================================
# 4. gold_products_by_order — categorias por pedido
# ==================================================================
class TestGoldProductsByOrder:
    def test_schema_contracts_exact_columns(self, gold_products_by_order):
        """gold_products_by_order deve ter exatamente as 4 colunas do contrato."""
        assert_schema(gold_products_by_order, GOLD_PRODUCTS_COLUMNS, "gold_products_by_order")

    def test_order_id_unique_and_not_null(self, gold_products_by_order):
        """order_id e PK: um registro por pedido."""
        assert gold_products_by_order["order_id"].notna().all()
        assert gold_products_by_order["order_id"].is_unique

    def test_num_categories_positive(self, gold_products_by_order):
        """Todo pedido agregado deve ter pelo menos 1 categoria."""
        assert (gold_products_by_order["num_categories"] >= 1).all()

    def test_num_categories_matches_pt_list(self, gold_products_by_order):
        """num_categories deve bater com os itens de product_categories_pt."""
        counted = gold_products_by_order["product_categories_pt"].str.split(", ").apply(len)
        mismatches = (gold_products_by_order["num_categories"] != counted).sum()
        assert mismatches == 0, f"{mismatches} linhas com num_categories inconsistente"

    def test_categories_not_null(self, gold_products_by_order):
        """Categorias PT e EN devem estar preenchidas em toda linha."""
        assert gold_products_by_order["product_categories_pt"].notna().all()
        assert gold_products_by_order["product_categories_en"].notna().all()


# ==================================================================
# 5. Integridade referencial entre as 4 tabelas
# ==================================================================
class TestReferentialIntegrity:
    def test_reviews_reference_existing_orders(self, gold_reviews, gold_orders):
        """Todo order_id de gold_reviews deve existir em gold_orders."""
        missing = ~gold_reviews["order_id"].isin(set(gold_orders["order_id"]))
        assert missing.sum() == 0, f"{missing.sum()} reviews apontam para pedidos inexistentes"

    def test_orders_reference_existing_customers(self, gold_orders, gold_customers):
        """Todo customer_id de gold_orders deve existir em gold_customers."""
        missing = ~gold_orders["customer_id"].isin(set(gold_customers["customer_id"]))
        assert missing.sum() == 0, f"{missing.sum()} pedidos apontam para clientes inexistentes"

    def test_products_reference_existing_orders(self, gold_products_by_order, gold_orders):
        """Todo order_id de gold_products_by_order deve existir em gold_orders."""
        missing = ~gold_products_by_order["order_id"].isin(set(gold_orders["order_id"]))
        assert missing.sum() == 0, f"{missing.sum()} agregacoes de categoria sem pedido"

    def test_reviews_categories_coverage(self, gold_reviews, gold_products_by_order):
        """Pelo menos 98% dos pedidos das reviews devem ter categorias agregadas.

        (Historicamente ~1.2% dos reviews ficam sem categoria no join — margem admitida.)
        """
        covered = gold_reviews["order_id"].isin(set(gold_products_by_order["order_id"]))
        ratio = covered.mean()
        assert ratio >= 0.98, f"cobertura de categorias caiu para {ratio:.1%}"


# ==================================================================
# 6. Corpus VoC exportado como JSONL
# ==================================================================
class TestVocReviewsJsonl:
    @pytest.fixture(scope="class")
    def jsonl_records(self, s3_client):
        resp = s3_client.get_object(Bucket=S3_BUCKET, Key=f"{GOLD_PREFIX}voc_reviews.jsonl")
        body = resp["Body"].read().decode("utf-8")
        return [json.loads(line) for line in body.splitlines() if line.strip()]

    def test_jsonl_row_count_matches_parquet(self, jsonl_records, gold_reviews):
        """O JSONL deve ter exatamente o mesmo numero de documentos do parquet."""
        assert len(jsonl_records) == len(gold_reviews), (
            f"jsonl: {len(jsonl_records)} docs | parquet: {len(gold_reviews)} docs"
        )

    def test_jsonl_record_contract(self, jsonl_records, gold_reviews):
        """Cada registro JSONL deve ter review_id e document_text preenchidos."""
        first = jsonl_records[0]
        assert "review_id" in first
        assert "document_text" in first
        assert first["document_text"]

    def test_jsonl_ids_match_parquet(self, jsonl_records, gold_reviews):
        """Os review_ids do JSONL devem ser exatamente os do parquet."""
        jsonl_ids = {r["review_id"] for r in jsonl_records}
        parquet_ids = set(gold_reviews["review_id"])
        assert jsonl_ids == parquet_ids, (
            f"{len(jsonl_ids - parquet_ids)} ids so no JSONL, "
            f"{len(parquet_ids - jsonl_ids)} ids so no parquet"
        )
