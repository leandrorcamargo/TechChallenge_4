"""Configuracoes compartilhadas entre o build e a busca."""

from pathlib import Path

# Modelo de embedding: multilingue (entende portugues), pequeno (roda em CPU)
# e treinado especificamente para busca (retrieval).
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# Os modelos e5 foram treinados com prefixos que dizem o papel de cada texto.
# Usar o prefixo certo melhora a qualidade da busca.
PASSAGE_PREFIX = "passage: "   # documentos guardados (reviews)
QUERY_PREFIX = "query: "       # perguntas do usuario

# Onde o banco vetorial fica salvo (pasta local, nao versionada)
VECTOR_STORE_DIR = Path(__file__).resolve().parent / "vector_store"
COLLECTION_NAME = "olist_reviews"
