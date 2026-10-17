"""Structured clinical-obligation extraction: schema, deadline parsing, evidence grounding and three extractors.

  A  RuleExtractor        deterministic clause decomposition + category/deadline/role/dependency rules (no ML, no LLM)
  B  LLMExtractor         small local GenAI (llama.cpp, grammar-constrained JSON) -- output accepted after schema parsing only
  C  ValidatedExtractor   B + schema validation, retries, evidence-grounding checks, rejection of unsupported deadlines/roles
  D  HybridExtractor      C + agreement check with the supervised action classifier (task_type correction when it is confident)

The LLM never touches workflow state: it only proposes records that must survive deterministic validation.
"""
from __future__ import annotations

import difflib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .action_rules import propose
from .ontology import CATEGORIES, ROLES

TaskType = Literal["appointment_referral", "laboratory_test", "imaging", "procedure", "medication_action", "patient_communication", "followup_review", "other_action"]
ISO_DUR = re.compile(r"^(PT\d+(?:\.\d+)?H|P\d+(?:\.\d+)?[DWMY])$")
I = re.IGNORECASE


# ----------------------------------------------------------------------------------------------- schema
class EvidenceSpan(BaseModel):
    start: int
    end: int
    text: str
    grounded: bool = True          # False when the text could not be located verbatim in the source note


class ExtractedObligation(BaseModel):
    obligation_id: str
    task_type: TaskType
    description: str = Field(min_length=3, max_length=400)
    due_interval: Optional[str] = None            # ISO-8601 duration (PT48H, P2D, P2W, P3M, P1Y); PT0H = "immediately"
    due_date: Optional[str] = None                # ISO date, only if an absolute date is stated
    responsible_role: Optional[str] = None        # canonical role from ontology.ROLES, only if stated
    dependencies: list[str] = Field(default_factory=list)
    conditional_requirements: Optional[str] = None
    evidence_span: EvidenceSpan
    source_note_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    missing_information: list[str] = Field(default_factory=list)
    extractor: str = "unknown"

    @field_validator("due_interval")
    @classmethod
    def _iso(cls, v):
        if v is not None and not ISO_DUR.match(v):
            raise ValueError(f"due_interval must be ISO-8601 duration like PT48H/P2D/P2W, got {v!r}")
        return v

    @field_validator("responsible_role")
    @classmethod
    def _role(cls, v):
        if v is not None and v not in ROLES:
            raise ValueError(f"responsible_role must be one of {ROLES}, got {v!r}")
        return v


@dataclass
class ExtractionResult:
    note_id: str
    extractor: str
    obligations: list[ExtractedObligation] = field(default_factory=list)
    schema_valid: bool = True            # raw output parsed and validated on the first attempt
    n_retries: int = 0
    latency_s: float = 0.0
    completion_tokens: int = 0
    rejected: list[dict] = field(default_factory=list)       # items removed by validation, with reasons
    repairs: list[dict] = field(default_factory=list)        # fields corrected/dropped by validation
    raw_text: str = ""


# ----------------------------------------------------------------------------------------------- deadlines
_NUMW = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
         "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "forty-eight": 48, "forty eight": 48, "seventy-two": 72, "seventy two": 72,
         "couple": 2}
_NUM = r"(?:\d+(?:\.\d+)?|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty[- ]eight|seventy[- ]two|forty)"
_UNIT = r"(?:hours?|hrs?|days?|weeks?|wks?|months?|years?)"
_PREP = r"(?:within(?:\s+the\s+next)?|in(?:\s+the\s+next)?|after|by|no\s+later\s+than|over\s+the\s+next|inside\s+of|inside|before|for\s+the\s+next)"
_RE_NUM = re.compile(rf"\b{_PREP}\s+(?:about\s+|approximately\s+|around\s+|another\s+)?({_NUM})(?:\s*(?:-|–|to|or)\s*({_NUM}))?[\s-]*({_UNIT})\b", I)
_RE_COUPLE = re.compile(rf"\b{_PREP}\s+(?:a\s+)?couple\s+of\s+({_UNIT})\b", I)
_RE_NEXT = re.compile(r"\bnext\s+(week|month|day|year)\b", I)
_RE_FIXED = re.compile(r"\b(tomorrow|today|tonight|this (?:morning|afternoon|evening)|as soon as possible|asap|immediately|right away|stat|urgently|emergently)\b", I)
_UNIT_ISO = {"h": "H", "d": "D", "w": "W", "m": "M", "y": "Y"}


@dataclass
class Deadline:
    iso: str
    start: int
    end: int
    text: str
    hours: float
    is_range: bool = False


def interval_hours(iso: str | None) -> float | None:
    if not iso:
        return None
    m = re.match(r"^PT(\d+(?:\.\d+)?)H$", iso)
    if m:
        return float(m.group(1))
    m = re.match(r"^P(\d+(?:\.\d+)?)([DWMY])$", iso)
    if not m:
        return None
    return float(m.group(1)) * {"D": 24, "W": 168, "M": 720, "Y": 8760}[m.group(2)]


def _num(tok: str) -> float:
    tok = tok.lower().replace("-", " ") if not tok[0].isdigit() else tok
    return float(tok) if tok[0].isdigit() else float(_NUMW.get(tok, _NUMW.get(tok.replace(" ", "-"), 0)))


def _iso(n: float, unit: str) -> str:
    u = unit.lower()[0]
    n = int(n) if float(n).is_integer() else n
    return f"PT{n}H" if u == "h" else f"P{n}{_UNIT_ISO[u]}"


def parse_deadline(text: str) -> Deadline | None:
    """First explicit relative deadline in `text` -> ISO-8601 duration (ranges use the upper bound). None if no deadline is stated."""
    cands: list[Deadline] = []
    for m in _RE_NUM.finditer(text):
        lo, hi, unit = m.group(1), m.group(2), m.group(3)
        n = max(_num(lo), _num(hi)) if hi else _num(lo)
        if n <= 0:
            continue
        iso = _iso(n, unit)
        cands.append(Deadline(iso, m.start(), m.end(), m.group(0), interval_hours(iso), bool(hi)))
    for m in _RE_COUPLE.finditer(text):
        iso = _iso(2, m.group(1))
        cands.append(Deadline(iso, m.start(), m.end(), m.group(0), interval_hours(iso)))
    for m in _RE_NEXT.finditer(text):
        unit = m.group(1)
        iso = {"week": "P1W", "month": "P1M", "day": "P1D", "year": "P1Y"}[unit.lower()]
        cands.append(Deadline(iso, m.start(), m.end(), m.group(0), interval_hours(iso)))
    for m in _RE_FIXED.finditer(text):
        w = m.group(1).lower()
        iso = "P1D" if w == "tomorrow" else ("PT24H" if w in ("today", "tonight") or w.startswith("this ") else "PT0H")
        cands.append(Deadline(iso, m.start(), m.end(), m.group(0), interval_hours(iso)))
    return min(cands, key=lambda d: d.start) if cands else None


# ----------------------------------------------------------------------------------------------- roles / conditionals
_ROLE_ALIASES = {
    "ordering_clinician": r"(?:treating\s+|ordering\s+|attending\s+|referring\s+)?(?:physician|doctor|clinician|provider|hospitalist|resident|pcp|primary\s+care(?:\s+(?:provider|physician|doctor))?)",
    "nurse": r"(?:nurse|nursing\s+staff|rn|care\s+nurse)",
    "pharmacy": r"(?:pharmacist|pharmacy)",
    "radiology": r"(?:radiologist|radiology(?:\s+department)?)",
    "specialist": r"(?:cardiologist|neurologist|oncologist|endocrinologist|nephrologist|gastroenterologist|surgeon|specialist|dermatologist|urologist)",
    "scheduler": r"(?:scheduler|scheduling\s+(?:team|staff)|front\s+desk|care\s+coordinator|referral\s+coordinator|coordinator)",
    "laboratory": r"(?:laboratory(?:\s+staff)?|lab\s+staff|phlebotomist|the\s+lab)",
    "patient": r"(?:patient|he|she|they)",
}
_AGENT_AFTER = r"\s+(?:should|must|will|to|is\s+to|needs?\s+to|shall|is\s+responsible|to\s+be\s+responsible)\b"
_ROLE_BY = {r: re.compile(rf"\b(?:by|with|ask|asks|have|having|from)\s+(?:the\s+)?(?:a\s+)?{p}\b", I) for r, p in _ROLE_ALIASES.items() if r != "patient"}
_ROLE_SUBJ = {r: re.compile(rf"\b{p}{_AGENT_AFTER}", I) for r, p in _ROLE_ALIASES.items()}
_COND_RE = re.compile(r"\b(?:if|unless|when|once|provided\s+that|in\s+case|as\s+needed|should\s+(?:abnormalit|symptom|it\s+|they\s+|the\s+\w+\s+persist)|should\s+\w+\s+persist)\b[^.;]*", I)


def find_role(clause: str, sentence: str | None = None) -> str | None:
    for scope in (clause,) + ((sentence,) if sentence and sentence != clause else ()):
        for r, rx in _ROLE_BY.items():
            if rx.search(scope):
                return r
        for r, rx in _ROLE_SUBJ.items():
            if r == "patient":
                continue
            if rx.search(scope):
                return r
    m = _ROLE_SUBJ["patient"].search(clause)
    if m and re.search(r"\bpatient\b", clause, I) and not re.search(r"\b(?:contact|notify|inform|call|tell|update|educat|counsel|instruct|advis)\w*\s+(?:the\s+)?patient", clause, I):
        return "patient"
    return None


def role_mentioned(role: str, scope: str) -> bool:
    if role == "patient":
        return bool(re.search(r"\bpatient\b|\bhe\b|\bshe\b|\bthey\b", scope, I))
    return bool(re.search(rf"\b{_ROLE_ALIASES[role]}\b", scope, I))


def find_conditional(clause: str) -> str | None:
    m = _COND_RE.search(clause)
    return m.group(0).strip(" ,.") if m else None


# ----------------------------------------------------------------------------------------------- text utilities
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n+")


def sentences(text: str) -> list[tuple[str, int, int]]:
    out, pos = [], 0
    for m in list(_SENT.finditer(text)) + [None]:
        end = m.start() if m else len(text)
        seg = text[pos:end]
        lead = len(seg) - len(seg.lstrip())
        s = seg.strip()
        if s:
            out.append((s, pos + lead, pos + lead + len(s)))
        pos = m.end() if m else len(text)
    return out


def find_span(source: str, snippet: str, fuzzy: float = 0.85) -> tuple[int, int, bool] | None:
    """Locate `snippet` in `source`: (start, end, exact). Exact (case-insensitive) first; then fuzzy over clause/sentence windows."""
    if not snippet or not snippet.strip():
        return None
    i = source.lower().find(snippet.strip().lower())
    if i >= 0:
        return i, i + len(snippet.strip()), True
    if fuzzy >= 1.0:
        return None
    best, bs = 0.0, None
    sn = norm(snippet)
    for s, a, b in sentences(source):
        pieces = [(s, a, b)] + [(p, a + s.find(p), a + s.find(p) + len(p)) for p in re.split(r"[,;]\s*", s) if len(p) > 8 and s.find(p) >= 0]
        for p, pa, pb in pieces:
            r = difflib.SequenceMatcher(None, norm(p), sn).ratio()
            if r > best:
                best, bs = r, (pa, pb)
    return (bs[0], bs[1], False) if bs and best >= fuzzy else None


# ----------------------------------------------------------------------------------------------- A: rule-based extractor
_CUE = re.compile(r"\b(?:repeat|obtain|order|check|draw|undergo|get|perform|recheck|review|reviewed|acknowledg\w*|interpret\w*|contact\w*|notif\w*|call\w*|inform\w*|tell|"
                  r"arrang\w*|schedul\w*|book\w*|refer\w*|follow[- ]?up|return|come back|reassess\w*|re-?evaluat\w*|start\w*|begin|initiat\w*|continue|stop|discontinu\w*|hold|"
                  r"increase|decrease|adjust\w*|prescrib\w*|refill|take|monitor\w*|admit\w*|discharge\w*|advis\w*|instruct\w*|counsel\w*|educat\w*|send|image|scan|"
                  r"x-?ray|mri|ct|ultrasound|echo\w*|biopsy|surgery|test\w*|labs?|blood\s*work)\b", I)
_DIRECTIVE = re.compile(r"\b(?:will|should|must|shall|needs?\s+to|need\s+for|to\s+be|is\s+to|are\s+to|recommend\w*|advis\w*|plan(?:ned|s)?|please|ask(?:ed)?|"
                        r"required|requires?|arrange\w*|ordered|scheduled|due)\b", I)
_IMPERATIVE = re.compile(r"^\s*(?:[-*\d.)]+\s*)?(?:repeat|obtain|order|check|draw|get|perform|recheck|review|contact|notify|call|inform|arrange|schedule|book|refer|"
                         r"start|begin|initiate|continue|stop|discontinue|hold|increase|decrease|adjust|prescribe|refill|take|monitor|admit|discharge|advise|instruct|counsel|"
                         r"educate|send|follow|return|reassess|re-?evaluate|ensure|confirm)\b", I)
_HIST = re.compile(r"\b(?:denies|denied|history of|was (?:started|referred|given|told|seen|admitted|prescribed|discharged)|had (?:a|an|been)|previously|last (?:year|month|week|visit)|has had|underwent)\b", I)
_SPLIT = re.compile(r"\s*;\s*|\s*,\s*(?:and\s+)?(?:then\s+)?(?=[a-z])|\s+and\s+then\s+|\s+and\s+(?=(?:the\s+)?(?:\w+\s+){0,3}(?:review|contact|notif|arrang|schedul|refer|repeat|obtain|order|start|continue|stop|follow|call|inform|check|perform|undergo|get|return|monitor|reassess|"
                    r"re-?evaluat|admit|discharge|advis|instruct|counsel|educat|send)\w*)|\s+then\s+|\s+followed by\s+|\s+after which\s+", I)
_COND_START = re.compile(r"^\s*(?:if|unless|when|once|provided|in case|as needed|should\s+(?:abnormalit|symptom|it|they|the))", I)
_REVIEW = re.compile(r"\b(?:review(?:ed|s|ing)?|acknowledg\w+|interpret\w*|sign(?:ed)?\s+off|look(?:ed)?\s+over)\b", I)
_RESULT_OBJ = re.compile(r"\b(?:results?|report|labs?|studies|imaging|findings|culture|pathology|films?|test)\b", I)
_NOTIFY = re.compile(r"\b(?:contact(?:ed)?|notif\w+|call(?:ed)?|inform\w*|tell|update|communicat\w+|reach\s+out|let\b[^.;]{0,20}\bknow)\b[^.;]{0,50}\b(?:patient|family|parent|caregiver|her|him|them)\b|"
                     r"\bpatient\b[^.;]{0,30}\b(?:contacted|informed|notified|called)\b", I)
_APPT = re.compile(r"\b(?:refer\w*|consult(?:ation)?\b|appointment|(?:arrang|schedul|book)\w*[^.;]{0,30}(?:visit|appointment|follow[- ]?up|clinic|consult)|follow[- ]?up[^.;]{0,25}(?:arranged|scheduled|booked)|"
                   r"(?:arranged|scheduled|booked)[^.;]{0,25}follow[- ]?up)\b", I)


def _classify_clause(clause: str) -> tuple[str, str] | None:
    """-> (task_type, step) or None."""
    if _REVIEW.search(clause) and _RESULT_OBJ.search(clause):
        return "followup_review", "review"
    if _NOTIFY.search(clause):
        return "patient_communication", "notify"
    if _APPT.search(clause):
        return "appointment_referral", "appointment"
    p = propose(clause, "plan")
    if p.labels:
        cat = p.labels[0]
        return cat, {"laboratory_test": "test", "imaging": "test", "procedure": "test"}.get(cat, "other")
    if re.search(r"\b(?:undergo|obtain|repeat|get|have)\b[^.;]{0,30}\b(?:blood|labs?|test\w*|cbc|panel)\b", clause, I):
        return "laboratory_test", "test"
    return None



def step_of(task_type: str, clause: str) -> str:
    """Workflow step implied by a clause: test | review | notify | appointment | other."""
    if _REVIEW.search(clause) and _RESULT_OBJ.search(clause):
        return "review"
    if task_type in ("laboratory_test", "imaging", "procedure"):
        return "test"
    if task_type == "patient_communication" or _NOTIFY.search(clause):
        return "notify"
    if task_type == "appointment_referral" or _APPT.search(clause):
        return "appointment"
    return "other"


def infer_dependencies(items: list[dict]) -> list[list[int]]:
    """Deterministic workflow dependencies from order + step type: review <- latest test; notification / appointment / conditional
    follow-up <- latest review (else latest test). items need keys task_type, clause (and optionally step)."""
    out = []
    for i, m in enumerate(items):
        st = m.get("step") or step_of(m["task_type"], m["clause"])
        prev = items[:i]
        steps = [p.get("step") or step_of(p["task_type"], p["clause"]) for p in prev]
        deps: list[int] = []
        if st == "review":
            deps = [j for j, x in enumerate(steps) if x == "test"][-1:]
        elif st in ("notify", "appointment") or (m["task_type"] == "followup_review" and st == "other"):
            reviews = [j for j, x in enumerate(steps) if x == "review"]
            tests = [j for j, x in enumerate(steps) if x == "test"]
            deps = reviews[-1:] or tests[-1:]
            if st == "appointment" and not find_conditional(m["clause"]) and not re.search(r"\b(?:after|following|once)\b", m["clause"], I):
                deps = reviews[-1:] if reviews else []
        out.append(deps)
    return out


class RuleExtractor:
    name = "A_rules"

    def extract(self, text: str, note_id: str = "note") -> ExtractionResult:
        t0 = time.perf_counter()
        obls: list[ExtractedObligation] = []
        meta: list[dict] = []
        for sent, s0, s1 in sentences(text):
            if sent.rstrip().endswith("?") or (_HIST.search(sent) and not _DIRECTIVE.search(sent) and not _IMPERATIVE.search(sent)):
                continue
            if not (_DIRECTIVE.search(sent) or _IMPERATIVE.search(sent)):
                continue
            # clause segmentation: pieces without an action cue (or starting with a conditional) join the previous piece
            pieces, pos = [], 0
            cuts = [m for m in _SPLIT.finditer(sent)]
            bounds = [0] + [m.end() for m in cuts]
            ends = [m.start() for m in cuts] + [len(sent)]
            for a, b in zip(bounds, ends):
                seg = sent[a:b]
                if not seg.strip():
                    continue
                if pieces and (_COND_START.match(seg) or not _CUE.search(seg)):
                    pieces[-1] = (pieces[-1][0], b)
                else:
                    pieces.append((a, b))
            for a, b in pieces:
                clause = sent[a:b].strip(" ,;").rstrip(".!")
                a2 = a + (len(sent[a:b]) - len(sent[a:b].lstrip(" ,;")))
                cls = _classify_clause(clause)
                if cls is None:
                    continue
                task_type, step = cls
                d = parse_deadline(clause)
                cstart = s0 + a2
                meta.append(dict(task_type=task_type, step=step, clause=clause, start=cstart, end=cstart + len(clause), sent=sent, deadline=d))
        for m, deps in zip(meta, infer_dependencies(meta)):
            m["deps"] = deps
        for i, m in enumerate(meta):
            role = find_role(m["clause"])
            cond = find_conditional(m["clause"])
            d = m["deadline"]
            missing = [x for x, v in (("deadline", d), ("responsible_role", role)) if not v]
            conf = 0.55 + (0.15 if d else 0) + (0.1 if role else 0) + (0.1 if _DIRECTIVE.search(m["clause"]) or _IMPERATIVE.search(m["clause"]) else 0)
            obls.append(ExtractedObligation(
                obligation_id=f"{note_id}:ob{i}", task_type=m["task_type"], description=re.sub(r"\s+", " ", m["clause"])[:300].strip().capitalize() or "obligation",
                due_interval=d.iso if d else None, responsible_role=role, dependencies=[f"{note_id}:ob{j}" for j in m["deps"]],
                conditional_requirements=cond, evidence_span=EvidenceSpan(start=m["start"], end=m["end"], text=text[m["start"]:m["end"]]),
                source_note_id=note_id, confidence=min(conf, 0.95), missing_information=missing, extractor=self.name))
        return ExtractionResult(note_id, self.name, obls, latency_s=time.perf_counter() - t0)


# ----------------------------------------------------------------------------------------------- grounding checks (used by C and by evaluation)
def unsupported_fields(ob: ExtractedObligation, source: str) -> list[str]:
    """Fields of `ob` that are asserted but not supported by `source` (deterministic, independent of any model)."""
    bad = []
    sp = ob.evidence_span
    located = source[sp.start:sp.end] if 0 <= sp.start < sp.end <= len(source) else None
    exact = located is not None and norm(located) == norm(sp.text)
    if not exact and find_span(source, sp.text, fuzzy=1.0) is None:
        bad.append("evidence_span")
    scope_text = located if exact else sp.text
    sent_scope = next((s for s, a, b in sentences(source) if a <= max(sp.start, 0) < b), scope_text) if exact else scope_text
    if ob.due_interval is not None:
        d = parse_deadline(sent_scope) or parse_deadline(scope_text)
        if d is None or abs((interval_hours(ob.due_interval) or -1) - d.hours) > 1e-6:
            bad.append("due_interval")
    if ob.responsible_role is not None and not role_mentioned(ob.responsible_role, sent_scope):
        bad.append("responsible_role")
    if ob.conditional_requirements is not None and norm(ob.conditional_requirements) not in norm(source):
        bad.append("conditional_requirements")
    return bad


def dependencies_valid(obls: list[ExtractedObligation]) -> list[str]:
    """Dependency problems: unknown id, self-dependency, forward reference (workflow must be acyclic and ordered)."""
    order = {o.obligation_id: i for i, o in enumerate(obls)}
    bad = []
    for o in obls:
        for d in o.dependencies:
            if d not in order or d == o.obligation_id or order[d] >= order[o.obligation_id]:
                bad.append(f"{o.obligation_id}->{d}")
    return bad


# ----------------------------------------------------------------------------------------------- B / C: LLM extractors
LLM_SCHEMA = {"type": "object", "properties": {"obligations": {"type": "array", "maxItems": 8, "items": {"type": "object", "properties": {
    "task_type": {"type": "string", "enum": CATEGORIES},
    "description": {"type": "string", "maxLength": 120},
    "due_interval": {"type": ["string", "null"], "maxLength": 12},
    "responsible_role": {"type": ["string", "null"], "enum": ROLES + [None]},
    "depends_on": {"type": "array", "maxItems": 3, "items": {"type": "integer"}},
    "conditional": {"type": ["string", "null"], "maxLength": 120},
    "evidence_span": {"type": "string", "maxLength": 240}},
    "required": ["task_type", "description", "due_interval", "responsible_role", "depends_on", "conditional", "evidence_span"]}}}, "required": ["obligations"]}

SYSTEM_PROMPT = (
    "You extract follow-up obligations from clinical instructions into JSON. Rules: (1) one obligation per distinct required action; "
    "(2) copy evidence_span verbatim from the text; (3) due_interval is an ISO-8601 duration (PT48H, P2D, P1W, P3M) ONLY if a deadline is explicitly stated for that action, else null; "
    "(4) responsible_role only if the text names who must do it, using one of " + ", ".join(ROLES) + ", else null; (5) depends_on lists 0-based indices of earlier obligations this one "
    "needs first (a result review depends on its test); (6) conditional is the condition text if the action only applies under a condition, else null; "
    "(7) task_type is one of " + ", ".join(CATEGORIES) + " (reviewing results = followup_review; arranging an appointment or referral = appointment_referral; "
    "telling the patient = patient_communication). Return {\"obligations\": []} when the text contains no required action.")
FEW_SHOT = [
    ("Obtain a basic metabolic panel within 24 hours. The nurse should call the patient with the result, and the physician will review it before discharge.",
     {"obligations": [
         {"task_type": "laboratory_test", "description": "Obtain a basic metabolic panel", "due_interval": "PT24H", "responsible_role": None, "depends_on": [], "conditional": None,
          "evidence_span": "Obtain a basic metabolic panel within 24 hours"},
         {"task_type": "patient_communication", "description": "Call the patient with the result", "due_interval": None, "responsible_role": "nurse", "depends_on": [0], "conditional": None,
          "evidence_span": "The nurse should call the patient with the result"},
         {"task_type": "followup_review", "description": "Physician reviews the result", "due_interval": None, "responsible_role": "ordering_clinician", "depends_on": [0], "conditional": None,
          "evidence_span": "the physician will review it before discharge"}]}),
    ("Refer to cardiology for echocardiography. Continue metoprolol 25 mg daily. She denies chest pain today.",
     {"obligations": [
         {"task_type": "appointment_referral", "description": "Refer to cardiology for echocardiography", "due_interval": None, "responsible_role": None, "depends_on": [], "conditional": None,
          "evidence_span": "Refer to cardiology for echocardiography"},
         {"task_type": "medication_action", "description": "Continue metoprolol 25 mg daily", "due_interval": None, "responsible_role": None, "depends_on": [], "conditional": None,
          "evidence_span": "Continue metoprolol 25 mg daily"}]}),
    ("He has a history of hypertension and was seen last year. Vitals are stable.", {"obligations": []}),
]


def _fewshot_messages() -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for text, out in FEW_SHOT:
        msgs += [{"role": "user", "content": text}, {"role": "assistant", "content": json.dumps(out)}]
    return msgs


def _build(items: list[dict], text: str, note_id: str, extractor: str, strict_roles: bool = False) -> tuple[list[ExtractedObligation], list[dict]]:
    """Convert raw LLM items to schema objects. Items that cannot satisfy the schema are returned in `rejected` (reason: schema)."""
    obls, rejected = [], []
    for i, it in enumerate(items):
        try:
            snippet = str(it.get("evidence_span") or "")
            loc = find_span(text, snippet, fuzzy=1.0)
            if loc:
                span = EvidenceSpan(start=loc[0], end=loc[1], text=text[loc[0]:loc[1]], grounded=True)
            else:
                span = EvidenceSpan(start=-1, end=-1, text=snippet or "(none)", grounded=False)
            dep_idx = [d for d in (it.get("depends_on") or []) if isinstance(d, int)]
            ob = ExtractedObligation(
                obligation_id=f"{note_id}:ob{i}", task_type=it.get("task_type"), description=(it.get("description") or snippet or "obligation")[:400],
                due_interval=it.get("due_interval"), responsible_role=it.get("responsible_role"),
                dependencies=[f"{note_id}:ob{d}" for d in dep_idx], conditional_requirements=it.get("conditional") or None,
                evidence_span=span, source_note_id=note_id, confidence=0.5, extractor=extractor)
            obls.append(ob)
        except Exception as e:  # noqa: BLE001
            rejected.append({"index": i, "reason": "schema_validation", "detail": str(e)[:160], "item": it})
    return obls, rejected


class LLMExtractor:
    """B: small local GenAI, output accepted as long as it parses against the schema (no grounding checks)."""
    name = "B_llm_raw"

    def __init__(self, backend, max_tokens: int = 500):
        self.backend, self.max_tokens = backend, max_tokens

    def _call(self, text: str, extra: list[dict] | None = None):
        msgs = _fewshot_messages() + [{"role": "user", "content": text}] + (extra or [])
        return self.backend.generate_json(msgs, LLM_SCHEMA, max_tokens=self.max_tokens)

    def extract(self, text: str, note_id: str = "note") -> ExtractionResult:
        res = ExtractionResult(note_id, self.name)
        r = self._call(text)
        res.latency_s, res.completion_tokens, res.raw_text = r.latency_s, r.completion_tokens, r.text
        try:
            items = json.loads(r.text)["obligations"]
        except Exception:  # noqa: BLE001
            res.schema_valid = False
            return res
        res.obligations, res.rejected = _build(items, text, note_id, self.name)
        res.schema_valid = not res.rejected
        return res


class ValidatedExtractor(LLMExtractor):
    """C: B + deterministic validation. Unsupported deadlines/roles/conditionals are dropped, ungrounded obligations rejected,
    invalid dependencies removed, and generation is retried (bounded) when nothing valid survives."""
    name = "C_llm_validated"

    def __init__(self, backend, max_tokens: int = 500, max_retries: int = 1, fuzzy_snap: float = 0.85, fill: bool = False):
        super().__init__(backend, max_tokens)
        self.max_retries, self.fuzzy_snap, self.fill = max_retries, fuzzy_snap, fill
        if fill:
            self.name = "C2_llm_validated_filled"

    def validate(self, obls: list[ExtractedObligation], text: str, rejected: list[dict], repairs: list[dict]) -> list[ExtractedObligation]:
        kept: list[ExtractedObligation] = []
        for ob in obls:
            if not ob.evidence_span.grounded:                      # try to snap to a near-verbatim passage before rejecting
                loc = find_span(text, ob.evidence_span.text, fuzzy=self.fuzzy_snap)
                if loc:
                    ob.evidence_span = EvidenceSpan(start=loc[0], end=loc[1], text=text[loc[0]:loc[1]], grounded=True)
                    repairs.append({"id": ob.obligation_id, "field": "evidence_span", "action": "snapped_to_source"})
                else:
                    rejected.append({"id": ob.obligation_id, "reason": "evidence_span_not_in_source", "detail": ob.evidence_span.text[:120]})
                    continue
            bad = unsupported_fields(ob, text)
            if "evidence_span" in bad:
                rejected.append({"id": ob.obligation_id, "reason": "evidence_span_not_in_source", "detail": ob.evidence_span.text[:120]})
                continue
            if "due_interval" in bad:                              # reject unsupported deadline: replace with the deterministic parse or null
                d = parse_deadline(ob.evidence_span.text)
                repairs.append({"id": ob.obligation_id, "field": "due_interval", "action": "replaced" if d else "dropped", "was": ob.due_interval, "now": d.iso if d else None})
                ob.due_interval = d.iso if d else None
            if "responsible_role" in bad:
                repairs.append({"id": ob.obligation_id, "field": "responsible_role", "action": "dropped", "was": ob.responsible_role})
                ob.responsible_role = None
            if "conditional_requirements" in bad:
                repairs.append({"id": ob.obligation_id, "field": "conditional_requirements", "action": "dropped", "was": ob.conditional_requirements})
                ob.conditional_requirements = None
            if ob.conditional_requirements is None:                # a conditional stated inside the span is legitimate evidence
                c = find_conditional(ob.evidence_span.text)
                if c and ob.task_type in ("patient_communication", "appointment_referral", "followup_review"):
                    ob.conditional_requirements = c
            if self.fill:                                          # complete fields ONLY through deterministic parsers over the cited evidence span
                d = parse_deadline(ob.evidence_span.text)
                if ob.due_interval is None and d:
                    ob.due_interval = d.iso
                    repairs.append({"id": ob.obligation_id, "field": "due_interval", "action": "filled_from_evidence_span", "now": d.iso})
                if ob.responsible_role is None:
                    r_ = find_role(ob.evidence_span.text)
                    if r_:
                        ob.responsible_role = r_
                        repairs.append({"id": ob.obligation_id, "field": "responsible_role", "action": "filled_from_evidence_span", "now": r_})
            ob.missing_information = [x for x, v in (("deadline", ob.due_interval), ("responsible_role", ob.responsible_role)) if not v]
            ob.confidence = min(0.9, 0.5 + 0.2 * bool(ob.due_interval) + 0.1 * bool(ob.responsible_role) + 0.1)
            kept.append(ob)
        # dedupe overlapping spans of the same type
        out: list[ExtractedObligation] = []
        for ob in sorted(kept, key=lambda o: (o.evidence_span.start, -o.evidence_span.end)):
            if any(o.task_type == ob.task_type and o.evidence_span.start <= ob.evidence_span.start and ob.evidence_span.end <= o.evidence_span.end and o.evidence_span.end - o.evidence_span.start >
                   1.5 * (ob.evidence_span.end - ob.evidence_span.start) for o in out):
                continue
            out.append(ob)
        ids = {o.obligation_id for o in out}
        for k, ob in enumerate(out):
            ob.dependencies = [d for d in ob.dependencies if d in ids and d != ob.obligation_id]
        if self.fill:
            inf = infer_dependencies([{"task_type": o.task_type, "clause": o.evidence_span.text} for o in out])
            for o, deps in zip(out, inf):
                o.dependencies = [out[j].obligation_id for j in deps]
            repairs.append({"field": "dependencies", "action": "inferred_by_workflow_rules"})
        # acyclic + ordered
        order = {o.obligation_id: i for i, o in enumerate(out)}
        for ob in out:
            keep = [d for d in ob.dependencies if order[d] < order[ob.obligation_id]]
            if len(keep) != len(ob.dependencies):
                repairs.append({"id": ob.obligation_id, "field": "dependencies", "action": "dropped_forward_or_cyclic", "was": ob.dependencies})
            ob.dependencies = keep
        return out

    def extract(self, text: str, note_id: str = "note") -> ExtractionResult:
        res = ExtractionResult(note_id, self.name)
        extra = None
        for attempt in range(self.max_retries + 1):
            r = self._call(text, extra)
            res.latency_s += r.latency_s
            res.completion_tokens += r.completion_tokens
            res.raw_text = r.text
            try:
                items = json.loads(r.text)["obligations"]
                obls, rej = _build(items, text, note_id, self.name)
            except Exception as e:  # noqa: BLE001
                if attempt == 0:
                    res.schema_valid = False
                res.n_retries = attempt + 1 if attempt < self.max_retries else attempt
                extra = [{"role": "assistant", "content": r.text[:300]}, {"role": "user", "content": f"Invalid JSON ({str(e)[:80]}). Return valid JSON only."}]
                continue
            if attempt == 0:
                res.schema_valid = not rej
            repairs: list[dict] = []
            rejected = list(rej)
            valid = self.validate(obls, text, rejected, repairs)
            res.obligations, res.rejected, res.repairs = valid, rejected, repairs
            has_cue = bool(_DIRECTIVE.search(text) or _IMPERATIVE.search(text))
            if valid or attempt == self.max_retries or not (items and has_cue):
                break
            res.n_retries = attempt + 1
            extra = [{"role": "assistant", "content": r.text[:400]},
                     {"role": "user", "content": "Every evidence_span must be copied exactly from the text, and deadlines/roles must be stated in the text. Try again."}]
        # rebuild stable sequential ids (two-phase mapping so renames cannot collide)
        mapping = {ob.obligation_id: f"{note_id}:ob{i}" for i, ob in enumerate(res.obligations)}
        for ob in res.obligations:
            ob.obligation_id = mapping[ob.obligation_id]
            ob.dependencies = [mapping[d] for d in ob.dependencies if d in mapping]
        return res


class HybridExtractor(ValidatedExtractor):
    """D: validated GenAI + supervised classifier. If the classifier is confident (prob >= tau) about a *different* task_type for an
    obligation's evidence span, the ML label overrides the LLM's; every override is logged."""
    name = "D_hybrid_llm_ml"

    def __init__(self, backend, classifier, tau: float = 0.6, **kw):
        super().__init__(backend, **kw)
        self.classifier, self.tau = classifier, tau
        self.name = "D_hybrid_llm_ml"

    def extract(self, text: str, note_id: str = "note") -> ExtractionResult:
        res = super().extract(text, note_id)
        for ob in res.obligations:
            label, p = self.classifier.predict_one(ob.evidence_span.text)
            if label in CATEGORIES and label != ob.task_type and p >= self.tau:
                res.repairs.append({"id": ob.obligation_id, "field": "task_type", "action": "overridden_by_classifier", "was": ob.task_type, "now": label, "p": round(p, 3)})
                ob.task_type = label
        res.extractor = self.name
        for ob in res.obligations:
            ob.extractor = self.name
        return res


class EnsembleExtractor:
    """E: union of the deterministic rule extractor and the validated GenAI extractor. Overlapping obligations are merged (rule fields win:
    they are deterministic and grounded); obligations found only by the LLM are kept because they passed validation."""
    name = "E_rules_plus_llm"

    def __init__(self, rules: RuleExtractor, llm: ValidatedExtractor):
        self.rules, self.llm = rules, llm

    def extract(self, text: str, note_id: str = "note") -> ExtractionResult:
        a = self.rules.extract(text, note_id)
        b = self.llm.extract(text, note_id)
        res = ExtractionResult(note_id, self.name, latency_s=a.latency_s + b.latency_s, completion_tokens=b.completion_tokens, schema_valid=b.schema_valid,
                               n_retries=b.n_retries, rejected=b.rejected, repairs=b.repairs, raw_text=b.raw_text)
        merged = list(a.obligations)
        for ob in b.obligations:
            ov = [m for m in merged if max(0, min(m.evidence_span.end, ob.evidence_span.end) - max(m.evidence_span.start, ob.evidence_span.start)) /
                  max(min(m.evidence_span.end - m.evidence_span.start, ob.evidence_span.end - ob.evidence_span.start), 1) >= 0.5]
            if not ov:
                merged.append(ob)
        merged.sort(key=lambda o: o.evidence_span.start)
        items = [{"task_type": o.task_type, "clause": o.evidence_span.text} for o in merged]
        deps = infer_dependencies(items)
        for i, o in enumerate(merged):
            o.obligation_id = f"{note_id}:ob{i}"
            o.dependencies = [f"{note_id}:ob{j}" for j in deps[i]]
            o.extractor = self.name
        res.obligations = merged
        return res
