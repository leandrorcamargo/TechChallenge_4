# Databricks notebook source
# DBTITLE 1,Manifesto da Camada Bronze
# MAGIC %md
# MAGIC # Manifesto da Camada Bronze
# MAGIC
# MAGIC Gera um arquivo `.json` com os metadados de cada tabela da camada Bronze no S3:
# MAGIC nome, arquivo, tamanho, linhas, colunas, tipos, nulos e timestamp de geracao.

# COMMAND ----------

# DBTITLE 1,Imports e Configuração
import io
import os
import json
import boto3
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timezone

# Carrega credenciais do arquivo .env
load_dotenv("/Workspace/Users/justinofilipe03@gmail.com/Tech4/.env")

# Credenciais AWS
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")

# Bucket e caminhos no S3
S3_BUCKET = os.environ.get("S3_BUCKET", "amzn-s3-fiap-tech-4")
BRONZE_PREFIX = os.environ.get("BRONZE_PREFIX", "db_bronze/db_bronze/")

# Caminho do manifesto no S3
MANIFEST_KEY = f"{BRONZE_PREFIX}manifest.json"

# COMMAND ----------

# DBTITLE 1,Funções de I/O
def get_s3_client():
    """Cria cliente S3 com as credenciais configuradas."""
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION,
    )


def read_s3_parquet(s3, key):
    """Le um parquet do S3 e retorna o DataFrame."""
    resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return pd.read_parquet(io.BytesIO(resp["Body"].read()))

# COMMAND ----------

# DBTITLE 1,Construção dos Metadados por Tabela
def build_table_metadata(s3, file_obj):
    """
    Constroi o dicionario de metadados de uma tabela.
    Inclui: nome, arquivo, tamanho, linhas, colunas com tipos e nulos.
    """
    key = file_obj["Key"]
    file_name = key.split("/")[-1]
    table_name = file_name.replace(".parquet", "")
    file_size = file_obj["Size"]

    # Le o parquet para extrair schema e estatisticas
    df = read_s3_parquet(s3, key)

    # Lista de colunas com tipo e contagem de nulos
    columns = []
    for col in df.columns:
        columns.append({
            "name": col,
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isna().sum()),
            "unique_count": int(df[col].nunique()),
        })

    return {
        "table_name": table_name,
        "file_name": file_name,
        "s3_path": f"s3://{S3_BUCKET}/{key}",
        "file_size_bytes": file_size,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": columns,
    }

# COMMAND ----------

# DBTITLE 1,Geração do Manifesto
def generate_manifest(s3):
    """
    Gera o manifesto completo da camada Bronze.
    Lista todos os .parquet no prefix, coleta metadados e grava o JSON no S3.
    """
    print("Gerando manifesto da camada Bronze...")

    # Lista todos os arquivos .parquet no prefix do Bronze
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=BRONZE_PREFIX)
    parquet_files = [
        obj for obj in response.get("Contents", [])
        if obj["Key"].endswith(".parquet")
    ]

    print(f"  Encontrados {len(parquet_files)} arquivos parquet")

    # Coleta metadados de cada tabela
    tables = []
    for i, fobj in enumerate(parquet_files, 1):
        table_name = fobj["Key"].split("/")[-1].replace(".parquet", "")
        print(f"  [{i}/{len(parquet_files)}] Processando: {table_name}")
        meta = build_table_metadata(s3, fobj)
        tables.append(meta)

    # Monta o manifesto
    manifest = {
        "layer": "bronze",
        "source": f"s3://{S3_BUCKET}/{BRONZE_PREFIX}",
        "description": "Brazilian E-Commerce Public Dataset by Olist - camada Bronze (dados crus em parquet)",
        "table_count": len(tables),
        "total_rows": sum(t["row_count"] for t in tables),
        "total_size_bytes": sum(t["file_size_bytes"] for t in tables),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tables": tables,
    }

    # Grava o JSON no S3
    json_bytes = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
    s3.put_object(Bucket=S3_BUCKET, Key=MANIFEST_KEY, Body=json_bytes)
    print(f"\nManifesto gravado em s3://{S3_BUCKET}/{MANIFEST_KEY}")
    print(f"  {len(tables)} tabelas | {manifest['total_rows']:,} linhas totais | {manifest['total_size_bytes']:,} bytes")

    return manifest

# COMMAND ----------

# DBTITLE 1,Execução
# Executa a geracao do manifesto
s3 = get_s3_client()
manifest = generate_manifest(s3)

# COMMAND ----------

# DBTITLE 1,Visualização do Manifesto
# Exibe o manifesto de forma legivel
print(json.dumps(manifest, indent=2, ensure_ascii=False))

# COMMAND ----------

