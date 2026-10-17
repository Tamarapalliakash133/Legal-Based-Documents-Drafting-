"""Input guardrails - pure regex/heuristics, microseconds, no extra LLM call.

Covers: empty/oversized input, direct prompt injection, jailbreak/roleplay, system-prompt
exfiltration, clearly illegal-assistance requests, private-person data requests, sensitive-ID
masking, and sanitising *retrieved* text (indirect prompt injection).
"""
import re
from dataclasses import dataclass
from typing import Optional

import rag.config as config


_I = re.IGNORECASE

_INJECTION = [
    r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier|preceding)\s+(instructions?|prompts?|rules?|messages?)",
    r"disregard\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier|system)\s+(instructions?|prompts?|rules?)",
    r"forget\s+(everything|all|your)\s+(you|instructions?|rules?|were told)",
    r"(reveal|show|print|display|repeat|leak|output|translate|summari[sz]e)\s+(me\s+)?(your|the)\s+(hidden\s+|secret\s+|internal\s+)?(system\s+)?(prompt|instructions?|configuration|rules)",
    r"what\s+(is|are|were)\s+your\s+(system\s+)?(prompt|instructions)",
    r"(developer|debug|admin|god|sudo)\s+mode",
    r"\bjailbreak(ed|ing)?\b",
    r"\bDAN\b.*\b(mode|prompt|do anything)\b",
    r"do\s+anything\s+now",
    r"you\s+are\s+now\s+(?!a\s+legal)(an?\s+)?[a-z ]{0,40}(without|with no)\s+(rules|restrictions|filters|limits)",
    r"pretend\s+(that\s+)?you\s+(have|has)\s+no\s+(rules|restrictions|guidelines|filters)",
    r"(bypass|override|disable|turn off)\s+(your\s+|the\s+|all\s+)?(safety|guardrails?|filters?|restrictions?|content policy)",
    r"i\s+am\s+(the\s+)?(developer|admin|administrator|owner|anthropic|openai)\b.*(authori[sz]e|permission|override)",
    r"(developers?|admins?)\s+(has\s+|have\s+)?authori[sz]ed\s+me",
    r"authori[sz]ation\s+code",
    r"repeat\s+(everything|all|the\s+text)\s+(above|before)",
    r"\b(system\s+prompt|guardrail\s+rules|hidden\s+instructions?|your\s+(prompt|instructions))\b",
    r"(base64|rot13|hex)\s+(encoding|encode|encoded)?.{0,30}(prompt|instructions)",
    r"ignore\s+the\s+(document|context)\s+(context\s+)?and\s+answer\s+only",
]

_HARMFUL = [
    r"\b(forge|forged|fabricate|fabricating|backdate|counterfeit)\b.{0,40}\b(document|signature|stamp|seal|court order|affidavit|will|certificate|receipt|agreement|contract)",
    r"\bfake\s+(court\s+order|notice|affidavit|will|stamp\s+paper|certificate|legal document)",
    r"(evade|avoid|dodge)\s+(paying\s+)?(taxes|tax|the\s+police|arrest|prosecution|summons)",
    r"(launder|laundering)\s+(money|cash|funds)",
    r"(destroy|hide|tamper\s+with|delete)\s+(the\s+)?evidence",
    r"(bribe|bribing|threaten|blackmail|intimidate)\s+(a\s+|the\s+)?(judge|witness|officer|police|juror|opponent)",
    r"(how\s+to|help\s+me)\s+(commit|get\s+away\s+with)\s+(fraud|murder|perjury|theft|a\s+crime)",
    r"(steal|stalk|harass|doxx?)\s+(someone|a\s+person|my\s+ex)",
    r"\b(fake|fabricated|forged|made-up|bogus)\b.{0,40}\b(notice|power of attorney|certificate|registration|affidavit|document)",
    r"hide\s+(my\s+)?(assets|money|income|property)\s+from",
    r"(alter|tamper|change|modify).{0,30}(notari[sz]ation|stamp|seal|signature).{0,40}(without|undetected|not\s+be\s+detected)",
    r"\bblackmail",
    r"made-up\s+facts",
]

_PRIVATE_DATA = [
    r"(home|residential|current|private)\s+address\s+of\b",
    r"where\s+(does|do)\s+.{1,60}\blive\b",
    r"(phone|mobile|contact)\s+number\s+of\b.{0,40}(complainant|defendant|plaintiff|witness|petitioner|respondent|party)",
    r"(aadhaar|aadhar|pan|ssn|social security|passport)\s+(number|no\.?|details)\s+of\b",
    r"full\s+personal\s+details",
    r"(sealed|confidential|privileged)\s+(file|record|case|document).{0,60}(not\s+a\s+party|even though|without\s+(permission|authori[sz]ation))",
    r"not\s+a\s+party\b.{0,60}\b(pull\s+up|access|open|show|give)",
    r"(pull\s+up|access|open|show|give\s+me)\s+.{0,30}\b(sealed)\b",
    r"(dump|export|list)\s+(all\s+)?(the\s+)?(client|customer|user)s?\s+(data|records|emails|files|information)",
    r"(national\s+id|bank\s+account|medical\s+history|salary)\b.{0,40}\b(of|named|in\s+the)\b",
    r"personal\s+email\s+addresses",
    r"(phone\s+number|address).{0,30}(where|and).{0,40}(children|kids|school)",
    r"(belonging\s+to|of)\s+another\s+(customer|client|user)",
    r"(i'?m|i\s+am)\s+not\s+a\s+party\s+(but|though)",
    r"(switch|change|set)\s+my\s+account\s+to\s+admin",
    r"(bypass|skip)\s+the\s+permission",
    r"(use|using)\s+(her|his|their)\s+login",
    r"(all|every)\s+(confidential|restricted|internal)\s+(documents?|records?|files?)",
    r"(knowledge\s+base|vector\s+(database|db|store)).{0,40}(export|dump|list|every|json)",
    r"(export|dump).{0,30}(vector\s+(database|db|store))",
    r"conversations?\s+with\s+other\s+users",
    r"(api\s+keys?|environment\s+variables?|credentials|secrets?)\b.{0,30}(you\s+use|print|output|show)",
    r"(print|output|show)\b.{0,20}(api\s+keys?|environment\s+variables?)",
    r"send\s+(all|every).{0,40}(to\s+my|@)",
]

_GUARANTEE = [
    r"guarantee\s+(that\s+)?(i|we|my)\s+(will|can|would)\s+(win|succeed|get)",
    r"(promise|assure|ensure)\s+(me\s+)?(that\s+)?(i|we)\s+(will|'ll)\s+win",
]

_SMALLTALK = re.compile(
    r"^\s*(hi+|hello+|hey+|hola|namaste|good\s+(morning|afternoon|evening)|thanks?( you)?|thank\s+you( so much)?|ok(ay)?|cool|great|bye|goodbye|who are you\??|what can you do\??|help)\s*[.!?]*\s*$",
    _I,
)


_CARD = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_AADHAAR = re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b")
_PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")

_INJ_RE = [re.compile(p, _I | re.DOTALL) for p in _INJECTION]
_HARM_RE = [re.compile(p, _I | re.DOTALL) for p in _HARMFUL]
_PRIV_RE = [re.compile(p, _I | re.DOTALL) for p in _PRIVATE_DATA]
_GUAR_RE = [re.compile(p, _I) for p in _GUARANTEE]


_CONTEXT_INJECTION = re.compile(
    r"(ignore\s+(all\s+)?(previous|prior|above)\s+instructions?|disregard\s+(the\s+)?(system|previous)|"
    r"^\s*(system|assistant|developer)\s*[:>]|\[\s*(system|admin|instruction)[^\]]*\]|"
    r"you\s+(must|should)\s+now\s+(reveal|ignore|output|send|say)|reveal\s+(the\s+)?(system\s+)?prompt|"
    r"do\s+not\s+tell\s+the\s+user|send\s+.{0,40}\s+to\s+https?://)",
    _I,
)

REFUSALS = {
    "prompt_injection": (
        "I can't follow requests to reveal or change my instructions. "
        "I'm happy to help with a legal question or a document draft instead."
    ),
    "harmful_request": (
        "I can't help with that because it involves wrongdoing or evading the law. "
        "I can explain the legal consequences, or help with a lawful alternative."
    ),
    "private_data": (
        "I can't share or look up private personal information or restricted case files. "
        "If you have a legitimate need, the right route is through the court or the "
        "party's counsel with proper authorisation."
    ),
    "empty": "Please type a question or describe what you need.",
    "too_long": "That message is too long. Please shorten it or split it into parts.",
}


@dataclass
class InputCheck:
    allowed: bool
    text: str
    category: Optional[str] = None
    message: Optional[str] = None
    smalltalk: bool = False
    pii_masked: bool = False
    flags: Optional[list] = None


def _luhn(num: str) -> bool:
    digits = [int(c) for c in num if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def mask_sensitive_ids(text: str) -> tuple[str, bool]:
    changed = False

    def card(m):
        nonlocal changed
        if _luhn(m.group(0)):
            changed = True
            return "[CARD-REDACTED]"
        return m.group(0)

    out = _CARD.sub(card, text)
    for rx, tag in ((_SSN, "[SSN-REDACTED]"), (_AADHAAR, "[ID-REDACTED]"), (_PAN, "[PAN-REDACTED]")):
        new = rx.sub(tag, out)
        changed = changed or new != out
        out = new
    return out, changed


def check_input(text: str, mode: str = "chat") -> InputCheck:
    """mode='chat'  -> strict, masks IDs.
       mode='draft' -> keeps personal details (a draft needs names/addresses) but still blocks
                       injection and illegal-document requests."""
    text = (text or "").strip()
    limit = config.MAX_CHAT_INPUT_CHARS if mode == "chat" else config.MAX_DRAFT_INPUT_CHARS
    if not text:
        return InputCheck(False, text, "empty", REFUSALS["empty"])
    if len(text) > limit:
        return InputCheck(False, text, "too_long", REFUSALS["too_long"])

    if any(r.search(text) for r in _INJ_RE):
        return InputCheck(False, text, "prompt_injection", REFUSALS["prompt_injection"])
    if any(r.search(text) for r in _HARM_RE):
        return InputCheck(False, text, "harmful_request", REFUSALS["harmful_request"])
    if mode == "chat" and any(r.search(text) for r in _PRIV_RE):
        return InputCheck(False, text, "private_data", REFUSALS["private_data"])

    flags = []
    if any(r.search(text) for r in _GUAR_RE):
        flags.append("asks_guarantee")  

    masked = False
    if mode == "chat":
        text, masked = mask_sensitive_ids(text)

    return InputCheck(
        True, text,
        smalltalk=bool(_SMALLTALK.match(text)),
        pii_masked=masked,
        flags=flags,
    )


def sanitize_context(text: str) -> str:
    """Indirect prompt-injection defence: neutralise instruction-like lines in retrieved docs."""
    if not _CONTEXT_INJECTION.search(text):
        return text
    kept = []
    for line in text.splitlines():
        kept.append("[instruction-like text removed]" if _CONTEXT_INJECTION.search(line) else line)
    return "\n".join(kept)
