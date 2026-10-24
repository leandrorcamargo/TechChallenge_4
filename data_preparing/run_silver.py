# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# DBTITLE 1,Camada Silver — Pipeline de Limpeza e Tipagem
# MAGIC %md
# MAGIC # Camada Silver — Pipeline de Limpeza e Tipagem dos Dados Olist
# MAGIC
# MAGIC **Tech Challenge Fase 4 — Voice of Customer Intelligence com RAG**
# MAGIC
# MAGIC Objetivo: Ler os 9 arquivos `.parquet` da camada Bronze no S3, aplicar limpeza, tipagem e derivações, e gravar os resultados de volta no S3 no formato Parquet.
# MAGIC
# MAGIC **Fluxo:** Bronze (S3) → Silver (S3) via boto3 + pandas

# COMMAND ----------

# DBTITLE 1,Imports e Configuração
import io
import os
import time
import boto3
import pandas as pd
from dotenv import load_dotenv

# Carrega credenciais do arquivo .env
load_dotenv("/Workspace/Users/justinofilipe03@gmail.com/Tech4/.env")

# Credenciais AWS
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")

# Bucket e caminhos no S3
S3_BUCKET = os.environ.get("S3_BUCKET", "amzn-s3-fiap-tech-4")
BRONZE_PREFIX = os.environ.get("BRONZE_PREFIX", "db_bronze/db_bronze/")
SILVER_PREFIX = os.environ.get("SILVER_PREFIX", "db_silver/")

# Lista das 9 tabelas
TABLES = [
    "olist_customers_dataset",
    "olist_geolocation_dataset",
    "olist_order_items_dataset",
    "olist_order_payments_dataset",
    "olist_order_reviews_dataset",
    "olist_orders_dataset",
    "olist_products_dataset",
    "olist_sellers_dataset",
    "product_category_name_translation",
]

# COMMAND ----------

# DBTITLE 1,Funções de I/O — S3
def get_s3_client():
    """Cria e retorna um cliente S3 com as credenciais configuradas."""
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION,
    )


def read_s3_parquet(s3, table_name):
    """Le um arquivo .parquet do S3 e retorna um DataFrame pandas."""
    key = f"{BRONZE_PREFIX}{table_name}.parquet"
    response = s3.get_object(Bucket=S3_BUCKET, Key=key)
    df = pd.read_parquet(io.BytesIO(response["Body"].read()))
    print(f"  [LEITURA] {table_name}: {len(df):,} linhas x {len(df.columns)} colunas")
    return df


def write_s3_parquet(s3, df, table_name):
    """Grava um DataFrame como .parquet no S3."""
    key = f"{SILVER_PREFIX}{table_name}.parquet"
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False, engine="pyarrow")
    buffer.seek(0)
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=buffer.getvalue())
    print(f"  [ESCRITA] {table_name}: {len(df):,} linhas gravadas em s3://{S3_BUCKET}/{key}")

# COMMAND ----------

# DBTITLE 1,Transformações — Customers, Geolocation, Items, Payments
def transform_customers(df):
    """normalizacao leve: strip de espacos em city/state."""
    for col in ["customer_city", "customer_state"]:
        df[col] = df[col].astype(str).str.strip()
    return df


def transform_geolocation(df):
    """normalizacao leve: strip de espacos em city/state."""
    for col in ["geolocation_city", "geolocation_state"]:
        df[col] = df[col].astype(str).str.strip()
    return df


def transform_order_items(df):
    """conversao de data e validacao de tipos."""
    df["shipping_limit_date"] = pd.to_datetime(df["shipping_limit_date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["freight_value"] = pd.to_numeric(df["freight_value"], errors="coerce")
    df["order_item_id"] = df["order_item_id"].astype(int)
    return df


def transform_order_payments(df):
    """validacao de tipos."""
    df["payment_sequential"] = df["payment_sequential"].astype(int)
    df["payment_installments"] = df["payment_installments"].astype(int)
    df["payment_value"] = pd.to_numeric(df["payment_value"], errors="coerce")
    return df

# COMMAND ----------

# DBTITLE 1,Transformação — Reviews (tabela central do corpus VoC)
def transform_order_reviews(df):
    """
    Tabela central do corpus VoC.
    1. Converte datas string -> datetime
    2. Cria flags has_title e has_message (boolean)
    3. Strip de espacos em campos de texto
    4. Deduplicacao por review_id (mantem primeira ocorrencia)
    """
    rows_before = len(df)

    # 1. Conversao de datas
    for col in ["review_creation_date", "review_answer_timestamp"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # 2. Flags de presenca de texto
    df["has_title"] = df["review_comment_title"].notna()
    df["has_message"] = df["review_comment_message"].notna()

    # 3. Strip de espacos nos campos de texto (sem converter NaN)
    for col in ["review_comment_title", "review_comment_message"]:
        mask = df[col].notna()
        df.loc[mask, col] = df.loc[mask, col].astype(str).str.strip()
        df.loc[mask & (df[col] == ""), col] = None
        if col == "review_comment_title":
            df["has_title"] = df["review_comment_title"].notna()
        else:
            df["has_message"] = df["review_comment_message"].notna()

    # 4. Deduplicacao por review_id
    df = df.drop_duplicates(subset=["review_id"], keep="first").reset_index(drop=True)
    rows_after = len(df)
    dups_removed = rows_before - rows_after
    if dups_removed > 0:
        print(f"  [DEDUP] {dups_removed} review_id duplicados removidos ({rows_before:,} -> {rows_after:,})")

    df["review_score"] = df["review_score"].astype(int)
    return df

# COMMAND ----------

# DBTITLE 1,Transformações — Orders, Products, Sellers, Translation
def transform_orders(df):
    """
    1. Converte 5 colunas de data string -> datetime
    2. Cria delivery_delay_days (entrega_real - entrega_estimada, em dias)
    3. Cria is_late_delivery (boolean: atrasou?)
    """
    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Negativo = adiantado | Positivo = atrasou | NaN = nao entregue
    df["delivery_delay_days"] = (
        df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
    ).dt.days

    df["is_late_delivery"] = df["delivery_delay_days"] > 0
    return df


def transform_products(df):
    """validacao de tipos. 610 produtos sem categoria permanecem NaN."""
    numeric_cols = [
        "product_name_lenght", "product_description_lenght", "product_photos_qty",
        "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def transform_sellers(df):
    """normalizacao leve: strip de espacos em city/state."""
    for col in ["seller_city", "seller_state"]:
        df[col] = df[col].astype(str).str.strip()
    return df


def transform_category_translation(df):
    """validacao de tipos: ambas as colunas sao string."""
    df["product_category_name"] = df["product_category_name"].astype(str).str.strip()
    df["product_category_name_english"] = df["product_category_name_english"].astype(str).str.strip()
    return df

# COMMAND ----------

# DBTITLE 1,Mapeamento e Relatório de Qualidade
TRANSFORM_MAP = {
    "olist_customers_dataset": transform_customers,
    "olist_geolocation_dataset": transform_geolocation,
    "olist_order_items_dataset": transform_order_items,
    "olist_order_payments_dataset": transform_order_payments,
    "olist_order_reviews_dataset": transform_order_reviews,
    "olist_orders_dataset": transform_orders,
    "olist_products_dataset": transform_products,
    "olist_sellers_dataset": transform_sellers,
    "product_category_name_translation": transform_category_translation,
}


def generate_quality_report(results):
    """
    Relatorio de qualidade com comparacao Antes (Bronze) x Depois (Silver).
    Mostra linhas, colunas e nulos antes e depois da transformacao.
    """
    print("\n" + "=" * 100)
    print("RELATORIO DE QUALIDADE — CAMADA SILVER (Antes x Depois)")
    print("=" * 100)

    # Linha de cabecalho
    header = (
        f"{'Tabela':<42} "
        f"{'Linhas Ant':>10} {'Linhas Dep':>10} {'Δ':>6} | "
        f"{'Cols Ant':>8} {'Cols Dep':>8} {'Δ':>4} | "
        f"{'Nulos Ant':>9} {'Nulos Dep':>9}"
    )
    print(header)
    print("-" * 100)

    for r in results:
        delta_rows = r["rows_after"] - r["rows_before"]
        delta_cols = r["cols_after"] - r["cols_before"]
        delta_rows_str = f"{delta_rows:+,}" if delta_rows != 0 else "="
        delta_cols_str = f"{delta_cols:+d}" if delta_cols != 0 else "="

        print(
            f"{r['table']:<42} "
            f"{r['rows_before']:>10,} {r['rows_after']:>10,} {delta_rows_str:>6} | "
            f"{r['cols_before']:>8} {r['cols_after']:>8} {delta_cols_str:>4} | "
            f"{r['nulls_before']:>9,} {r['nulls_after']:>9,}"
        )

    print("-" * 100)
    total = len(results)
    ok = sum(1 for r in results if r["status"] == "OK")
    print(f"Total: {total} tabelas | OK: {ok} | Erro: {total - ok}")
    print("=" * 100 + "\n")

# COMMAND ----------

# DBTITLE 1,Pipeline Principal
def run_silver():
    """
    Executa o pipeline completo da camada Silver.
    Para cada uma das 9 tabelas: le da Bronze, transforma, grava na Silver.
    """
    print("\n" + "=" * 80)
    print("PIPELINE SILVER — Iniciando processamento")
    print(f"  Bucket: s3://{S3_BUCKET}")
    print(f"  Bronze: {BRONZE_PREFIX}")
    print(f"  Silver: {SILVER_PREFIX}")
    print(f"  Tabelas: {len(TABLES)}")
    print("=" * 80 + "\n")

    s3 = get_s3_client()
    results = []
    start_time = time.time()

    for table_name in TABLES:
        print(f"\n--- Processando: {table_name} ---")
        table_start = time.time()

        try:
            # Le da Bronze e captura estatisticas ANTES
            df = read_s3_parquet(s3, table_name)
            rows_before = len(df)
            cols_before = len(df.columns)
            nulls_before = int(df.isna().sum().sum())

            # Aplica transformacao
            df = TRANSFORM_MAP[table_name](df)

            # Captura estatisticas DEPOIS
            rows_after = len(df)
            cols_after = len(df.columns)
            nulls_after = int(df.isna().sum().sum())

            # Grava na Silver
            write_s3_parquet(s3, df, table_name)

            elapsed = time.time() - table_start
            print(f"  [OK] {table_name} concluido em {elapsed:.1f}s")

            results.append({
                "table": table_name,
                "rows_before": rows_before,
                "rows_after": rows_after,
                "cols_before": cols_before,
                "cols_after": cols_after,
                "nulls_before": nulls_before,
                "nulls_after": nulls_after,
                "status": "OK",
                "time_s": round(elapsed, 1),
            })

        except Exception as e:
            print(f"  [ERRO] {table_name}: {e}")
            results.append({
                "table": table_name,
                "rows_before": 0, "rows_after": 0,
                "cols_before": 0, "cols_after": 0,
                "nulls_before": 0, "nulls_after": 0,
                "status": "ERRO", "time_s": 0,
            })

    generate_quality_report(results)
    total_time = time.time() - start_time
    print(f"Pipeline Silver concluido em {total_time:.1f}s")
    return results

# COMMAND ----------

# DBTITLE 1,Execução
# Executa o pipeline Silver completo
results = run_silver()

# COMMAND ----------

