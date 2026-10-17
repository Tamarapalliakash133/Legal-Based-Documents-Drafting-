import re
from dataclasses import dataclass, field

from guardrails.input_rules import mask_sensitive_ids

_I = re.IGNORECASE

_LEAK_MARKERS = [
    "retrieved context:",
    "you are a legal assistant answering questions",
    "you are a helpful general-purpose ai assistant",
    "the retrieved pdf context is empty",
    "treat everything inside <context>",
    "system prompt:",
    "you are an experienced advocate drafting",
]

_OVERCONFIDENT = re.compile(
    r"(guarantee[sd]?\s+(to\s+)?(win|succeed|victory)|100\s?%\s+(certain|sure|guaranteed)|"
    r"you\s+(will|are\s+going\s+to)\s+(definitely|certainly|surely)\s+(win|succeed)|"
    r"(definitely|certainly)\s+win\s+(the|your)\s+case)",
    _I,
)

LEAK_REPLACEMENT = (
    "I can't share my internal instructions. I'm glad to answer a legal question "
    "or help draft a document."
)
CAUTION = (
    "\n\n> **Note:** outcomes in legal matters depend on facts, evidence and the court. "
    "Nothing here is a guarantee - please consult a qualified advocate."
)


@dataclass
class OutputCheck:
    text: str
    changed: bool = False
    flags: list = field(default_factory=list)


def check_output(text: str, mode: str = "chat") -> OutputCheck:
    flags: list[str] = []
    low = text.lower()
    if any(m in low for m in _LEAK_MARKERS):
        return OutputCheck(LEAK_REPLACEMENT, True, ["system_prompt_leak"])

    out = text
    if mode == "chat":
        out, masked = mask_sensitive_ids(out)
        if masked:
            flags.append("sensitive_id_masked")

    if _OVERCONFIDENT.search(out):
        flags.append("overconfident_claim")
        if CAUTION.strip() not in out:
            out += CAUTION

    return OutputCheck(out, out != text, flags)
