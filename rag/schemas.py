"""Modelos e objetos que circulam entre as etapas.

Os Pydantic são os contratos de saída do LLM (o structured output valida
contra eles). Os dataclasses são os objetos internos do pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

Sentiment = Literal["positive", "neutral", "negative"]
Intent = Literal["problemas", "elogios", "tema_especifico", "evidencias", "comparacao", "outro"]
Status = Literal["answered", "out_of_scope", "insufficient_evidence"]


# ----------------------------------------------------------------------
# Saídas do LLM
# ----------------------------------------------------------------------
class Filters(BaseModel):
    """Filtros de metadados extraídos da pergunta (todos opcionais)."""

    sentiment: Sentiment | None = Field(
        default=None, description="positive (notas 4-5), neutral (3) ou negative (1-2), se a pergunta indicar"
    )
    score_min: int | None = Field(default=None, ge=1, le=5, description="nota mínima (1-5)")
    score_max: int | None = Field(default=None, ge=1, le=5, description="nota máxima (1-5)")
    year: int | None = Field(default=None, description="ano da avaliação (2016, 2017 ou 2018)")
    month: int | None = Field(default=None, ge=1, le=12, description="mês da avaliação (1-12)")
    category: str | None = Field(default=None, description="categoria de produto, ex.: cama_mesa_banho")
    state: str | None = Field(default=None, description="UF do cliente com 2 letras, ex.: SP")
    late_delivery: bool | None = Field(default=None, description="true = só pedidos entregues com atraso")
    order_status: str | None = Field(default=None, description="status do pedido, ex.: canceled, delivered")

    def active(self) -> dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class QueryAnalysis(BaseModel):
    """Resultado da etapa de processamento/transformação da pergunta."""

    in_scope: bool = Field(
        description="true se a pergunta é sobre experiências/avaliações de clientes de e-commerce "
        "(entrega, produto, atendimento, vendedor, compra); false caso contrário"
    )
    out_of_scope_reason: str | None = Field(default=None, description="motivo, se in_scope=false")
    intent: Intent = Field(description="tipo de investigação")
    search_query: str = Field(
        description="a pergunta reformulada como uma frase de busca clara e autocontida em português"
    )
    hypothetical_review: str = Field(
        description="uma avaliação curta e realista, escrita como um cliente escreveria, "
        "que seria uma evidência típica para responder à pergunta"
    )
    keywords: list[str] = Field(
        description="5 a 12 palavras-chave e sinônimos em português, como clientes escrevem "
        "(inclua variações e negações, ex.: 'não chegou', 'atraso', 'demorou')"
    )
    filters: Filters = Field(default_factory=Filters)


class Insight(BaseModel):
    texto: str = Field(description="um insight objetivo, sustentado pelas avaliações citadas")
    citacoes: list[str] = Field(description="IDs das avaliações que sustentam o insight, ex.: ['R1', 'R4']")


class VocAnswer(BaseModel):
    resposta: str = Field(description="síntese direta respondendo à pergunta, em português")
    insights: list[Insight] = Field(description="2 a 6 insights, cada um com suas citações")
    nivel_confianca: Literal["Alto", "Médio", "Baixo"] = Field(
        description="Alto se várias avaliações concordam; Médio se poucas; Baixo se evidência fraca"
    )
    evidencia_suficiente: bool = Field(
        description="false se as avaliações fornecidas NÃO permitem responder à pergunta"
    )


# ----------------------------------------------------------------------
# Objetos internos
# ----------------------------------------------------------------------
@dataclass
class Candidate:
    """Uma review candidata, acumulando scores das etapas de recuperação/seleção."""

    review_id: str
    text: str
    meta: dict[str, Any]
    ranks: dict[str, int] = field(default_factory=dict)  # fonte -> posição (1 = melhor)
    rrf_score: float = 0.0
    rerank_score: float | None = None

    @property
    def sources(self) -> str:
        return "+".join(sorted(self.ranks))


@dataclass
class Evidence:
    ref: str  # R1, R2...
    review_id: str
    text: str
    review_score: int
    category: str
    date: str
    state: str
    late_delivery: bool
    rerank_score: float | None
    sources: str
    cited: bool = False


@dataclass
class StepTrace:
    step: str
    duration_ms: float
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class RagResult:
    question: str
    status: Status
    answer: str
    insights: list[dict[str, Any]] = field(default_factory=list)
    confidence: str | None = None
    evidences: list[Evidence] = field(default_factory=list)
    trace: list[StepTrace] = field(default_factory=list)

    @property
    def total_ms(self) -> float:
        return sum(t.duration_ms for t in self.trace)
