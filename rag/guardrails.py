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

# Só o IMPERATIVO dirigido ao sistema ("ignore", "esqueça", "desconsidere"...).
# O indicativo ("os clientes ignoram as instruções de montagem", "o produto
# esquece as configurações") é pergunta legítima sobre avaliações e não casa.
_INJECTION_PATTERNS = [
    r"\b(ignore|ignorem|ignora|desconsidere|desconsiderem|desobede[cç]a|esque[cç]a|esque[cç]am)\s+"
    r"(todas?\s+|tudo\s+|suas?\s+|seus?\s+|as\s+|os\s+|quaisquer\s+)*"
    r"(instru|regra|orienta|restri|diretriz|comando|prompt|anterior|acima|tudo)",
    r"\b(ignore|disregard|forget)\s+(all\s+|any\s+|the\s+|your\s+|previous\s+|prior\s+|above\s+)+"
    r"(instruction|rule|prompt|guideline|context)",
    r"\b(voce|você)\s+agora\s+(e|é)\b",
    r"\byou\s+are\s+now\b",
    r"\b(finja|finjam|aja|atue|comporte-se|responda)\s+(que\s+(voce|você)\s+(e|é|esta|está)\s+|como\s+(se\s+fosse\s+)?)",
    r"\b(system\s*prompt|prompt\s+do\s+sistema|jailbreak|developer\s+mode|modo\s+desenvolvedor)\b",
    r"\brevel\w*\s+((o|a|as|os|seu|sua|suas|seus)\s+)+(prompt|instruc)",
    r"\bact\s+as\b|\bpretend\s+(to\s+be|you)\b",
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
