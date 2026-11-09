"""
Carrega a camada Gold LOCALMENTE, sem S3 e sem alterar o codigo do grupo.

Como funciona:
  1. Le os parquets Bronze que ja estao no repositorio (data_preparing/db_bronze/).
  2. Aplica as funcoes transform_* do data_preparing/run_silver.py  -> Silver.
  3. Executa as celulas 2 a 5 do data_preparing/run_gold.ipynb      -> Gold.

As funcoes e celulas sao lidas dos arquivos originais e executadas como estao,
entao qualquer mudanca que o grupo fizer la e refletida aqui automaticamente.
Nada e gravado no S3.
"""

import ast
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data_preparing"
BRONZE_DIR = DATA_DIR / "db_bronze"

# Mesmas tabelas que o run_gold.ipynb usa (SILVER_TABLES)
SILVER_TABLES = [
    "olist_order_reviews_dataset",
    "olist_orders_dataset",
    "olist_customers_dataset",
    "olist_order_items_dataset",
    "olist_products_dataset",
    "product_category_name_translation",
]

# Celulas do run_gold.ipynb que montam as tabelas (identificadas pelo titulo,
# nao pela posicao, para nao quebrar se alguem inserir uma celula nova).
GOLD_CELL_MARKERS = [
    "2. FILTRO",
    "3. CONSTRUCAO DO document_text",
    "4. AGREGACAO DE CATEGORIAS",
    "5. MONTAGEM DAS TABELAS GOLD",
]


def _load_silver_transforms():
    """Importa apenas as funcoes transform_* do run_silver.py.

    O arquivo e um notebook Databricks que, ao final, chama run_silver()
    (que acessa o S3). Por isso nao usamos `import`: lemos o codigo e
    executamos somente as definicoes das funcoes.
    """
    tree = ast.parse((DATA_DIR / "run_silver.py").read_text(encoding="utf-8"))
    namespace = {"pd": pd}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("transform_"):
            exec(compile(ast.Module([node], []), "run_silver.py", "exec"), namespace)
    return namespace


def build_silver():
    """Bronze (parquet local) -> Silver (DataFrames em memoria)."""
    ns = _load_silver_transforms()
    transform_map = {
        "olist_order_reviews_dataset": ns["transform_order_reviews"],
        "olist_orders_dataset": ns["transform_orders"],
        "olist_customers_dataset": ns["transform_customers"],
        "olist_order_items_dataset": ns["transform_order_items"],
        "olist_products_dataset": ns["transform_products"],
        "product_category_name_translation": ns["transform_category_translation"],
    }
    silver = {}
    for table in SILVER_TABLES:
        df = pd.read_parquet(BRONZE_DIR / f"{table}.parquet")
        silver[table] = transform_map[table](df)
    return silver


def build_gold():
    """Silver -> Gold, executando as celulas originais do run_gold.ipynb.

    Retorna um dict com as 4 tabelas: gold_reviews, gold_orders,
    gold_customers e gold_products_by_order.
    """
    silver = build_silver()
    notebook = json.loads((DATA_DIR / "run_gold.ipynb").read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]

    namespace = {"pd": pd, "silver": silver}
    for marker in GOLD_CELL_MARKERS:
        matches = [c for c in cells if marker in c]
        if len(matches) != 1:
            raise RuntimeError(f"Celula '{marker}' nao encontrada no run_gold.ipynb")
        exec(compile(matches[0], f"run_gold.ipynb [{marker}]", "exec"), namespace)

    return {
        name: namespace[name]
        for name in ["gold_reviews", "gold_orders", "gold_customers", "gold_products_by_order"]
    }


if __name__ == "__main__":
    gold = build_gold()
    print()
    for name, df in gold.items():
        print(f"{name:<25} {len(df):>7,} linhas x {len(df.columns)} colunas")
