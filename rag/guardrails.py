"""Guardrails de entrada e de saída.

Na entrada, uma heurística de regex detecta prompt injection antes de
chamar qualquer LLM. Na saída, a validação das citações: toda afirmação
precisa citar reviews que estavam de fato no contexto; citação que não
existe é removida e insight que fica sem citação válida é descartado —
o sistema recusa em vez de responder sem base.
"""

from __future__ import annotations

import re
import unicodedata

from rag.schemas import VocAnswer

_INJECTION_PATTERNS = [
    r"ignor\w*\s+(as|todas|suas|quaisquer|the|all|previous|any)?\s*(instruc|regra|orienta|instruction)",
    r"esque[cç]\w*\s+(as|todas|suas|tudo)",
    r"(desconsider|desobedec)\w*\s+(as|suas|todas)",
    r"(voce|você)\s+agora\s+(e|é)",
    r"\b(system\s*prompt|prompt\s+do\s+sistema|jailbreak|developer\s+mode|modo\s+desenvolvedor)\b",
    r"revel\w*\s+(seu|suas|o)\s+(prompt|instruc)",
    r"act\s+as|pretend\s+(to\s+be|you)",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)
_REF_RE = re.compile(r"R\d+", re.IGNORECASE)


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def detect_injection(question: str) -> bool:
    return bool(_INJECTION_RE.search(question) or _INJECTION_RE.search(_fold(question)))


def validate_citations(answer: VocAnswer, valid_refs: set[str]) -> tuple[list[dict], dict]:
    """Normaliza e valida as citações. Devolve (insights válidos, relatório pro trace)."""
    kept: list[dict] = []
    invalid: list[str] = []
    dropped: list[str] = []
    for ins in answer.insights:
        refs: list[str] = []
        for raw in ins.citacoes:
            for ref in _REF_RE.findall(raw):
                ref = ref.upper()
                if ref in valid_refs and ref not in refs:
                    refs.append(ref)
                elif ref not in valid_refs:
                    invalid.append(ref)
        if refs:
            kept.append({"texto": ins.texto, "citacoes": refs})
        else:
            dropped.append(ins.texto)
    report = {
        "insights_gerados": len(answer.insights),
        "insights_validos": len(kept),
        "insights_descartados_sem_citacao": dropped,
        "citacoes_invalidas_removidas": sorted(set(invalid)),
    }
    return kept, report
