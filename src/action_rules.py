"""Conservative rule-based proposer for clinical-action categories (silver labels).

This is a *candidate-label proposer*, NOT ground truth. Every output carries a rule id, matched
span, tense flag and a heuristic confidence. Items that are ambiguous are routed to the review queue.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

I = re.IGNORECASE

# ---- object vocabularies -------------------------------------------------------------------
LAB_OBJ = (r"labs?\b|laborator(?:y|ies)|blood\s*(?:work|tests?|counts?|cultures?|draw)|CBC\b|BMP\b|CMP\b|metabolic panel|"
           r"lipid(?: panel| profile)?|A1c|hemoglobin A1c|creatinine|electrolytes?|TSH\b|thyroid (?:panel|function|studies)|"
           r"liver (?:function|panel|enzymes)|LFTs?\b|cultures?\b|urinalysis|urine (?:test|studies|culture|drug screen)|INR\b|"
           r"troponins?|PSA\b|ferritin|iron studies|(?:serum|blood|drug|vitamin(?: [A-Z]\d*)?|lithium|potassium|sodium|glucose|magnesium|hormone|testosterone|cholesterol|thyroid|digoxin|vancomycin|INR|HbA1c)\s+levels?|stool (?:ova|test|studies|culture)|"
           r"ova and parasite|pregnancy test|pap smear|(?:serum|serology|titers?|panel|biomarker)\b|toxicology|ABGs?\b|blood gas|"
           r"d-dimer|HIV test|ESR\b|CRP\b|PT/INR|coagulation|cytology|pathology|screen(?:ing)? (?:test|labs)|quad screen|HbA1c|"
           r"hemoglobin|hematocrit|glucose (?:level|check|tolerance)|bilirubin|lipase|amylase|sedimentation|"
           r"(?:blood|serum|urine|genetic|diagnostic|confirmatory|screening|additional|further|repeat|laboratory|specific)\s+(?:tests?|testing|studies|workup)|work-?up|\bassays?\b|(?:tumor|cancer|tumour) markers?")
IMG_OBJ = (r"x-?rays?|radiographs?|radiography|MRI\b|magnetic resonance|CT\b|CAT scan|computed tomography|ultrasound|ultrasonograph\w*|"
           r"sonogra\w+|echocardiogra\w+|echo\b|mammogra\w+|DEXA|bone density|imaging|scans?\b|angiogra\w+|doppler|fluoroscop\w+|"
           r"PET\b|nuclear (?:scan|study)|barium|venogram|films?\b|CXR\b|KUB\b|HIDA")
PROC_OBJ = (r"surger(?:y|ies)|surgical|operation|operative|\w+ectomy|\w+otomy|\w+ostomy|\w+oscopy|\w+plasty|\w+pexy|\w+centesis|biops(?:y|ies)|"
            r"excision|ablation|catheteri[sz]ation|\bcath\b|stents?|\bPCI\b|bypass|CABG|resection|drainage|incision|aspiration|"
            r"intubat\w+|transfusions?|dialysis|chemotherapy|radiation therapy|radiotherapy|endoscop\w+|colonoscop\w+|EGD\b|"
            r"lumbar puncture|suture\w*|laceration repair|debridement|cardioversion|\bECG\b|\bEKG\b|electrocardiogram|stress test|holter|"
            r"procedure|reduction|injection|infusion|transplant\w*|vaccin\w+|immuni[sz]ation")
SPECIALTY = (r"cardiolog\w+|neurolog\w+|oncolog\w+|gastroenterolog\w+|nephrolog\w+|endocrinolog\w+|dermatolog\w+|psychiatr\w+|ophthalmolog\w+|"
             r"urolog\w+|orthop\w+|ENT\b|otolaryngolog\w+|rheumatolog\w+|pulmonolog\w+|hematolog\w+|surgeon|surgery|physical therapy|"
             r"PT\b|occupational therapy|\bOT\b|speech therap\w+|nutritionist|dietitian|social work\w*|case manag\w+|palliative|hospice|"
             r"rehab\w*|physiotherapy|genetic counsel\w+|specialist|obstetric\w*|gynecolog\w+|pain (?:clinic|management)|sleep (?:study|clinic)|"
             r"podiatr\w+|audiolog\w+|primary care|PCP\b|therapist|psycholog\w+|counsel(?:or|ing) services|clinic")
DRUG_SUFFIX = (r"[A-Za-z]+(?:pril|sartan|olol|dipine|statin|cillin|mycin|cycline|floxacin|azole|zepam|zolam|oxetine|pramine|triptyline|"
               r"vir\b|mab\b|prazole|tidine|thiazide|semide|gliptin|glitazone|formin|parin|xaban|barbital|sone\b|solone|caine|profen|"
               r"codone|morphine|fentanyl|cef\w*|amox\w*|penem|oxacin|dronate|lukast|terol|tropium|sulfa\w*|zosin|azepam|etine|"
               r"pam\b|lam\b|done\b|statin\b|dopa\b|trexate|platin|rubicin|tinib|nib\b|ximab|zumab)")
MED_OBJ = (r"\d+(?:\.\d+)?\s*(?:mg|mcg|g\b|grams?|units?|IU|mL|ml|meq)|tablets?|capsules?|pills?|daily\b|BID\b|TID\b|QID\b|PRN\b|\bq\d+h|"
           r"\bPO\b|\bIV\b|\bIM\b|subcutaneous\w*|intravenous\w*|intramuscular\w*|oral(?:ly)?\b|inhal\w+|medications?|medicines?|"
           r"antibiotics?|antihypertensives?|antivirals?|antifungals?|anticoagula\w+|statin|insulin|steroids?|analgesics?|inhalers?|"
           r"patch(?:es)?|cream|ointment|drops|doses?|dosing|therapy|regimen|prescription|vaccine|agent|treatment with|"
           r"aspirin|metformin|heparin|warfarin|prednisone|ibuprofen|acetaminophen|opioids?|NSAIDs?|diuretics?|beta[- ]blockers?|"
           + DRUG_SUFFIX)

PAST_NARR = (r"\b(?:denies|denied|denying|reports?|reported|states?|stated|complains?|history of|hx of|prior|previous(?:ly)?|"
             r"was (?:diagnosed|noted|found|seen|evaluated|admitted|referred|started|placed|treated|given|told|prescribed)|"
             r"had (?:a|an|been|undergone)|has had|have had|underwent|presented|presents with|presented with|ago\b|last (?:year|month|week|visit)|"
             r"since (?:the|his|her)|on (?:presentation|arrival|admission)|came in|brought (?:in|to))")
FUTURE_DIR = (r"\bwill\b|\bshall\b|\bto be\b|going to|plan(?:s|ned|ning)? (?:to|for|on)|recommend\w*|advis\w+|suggest\w*|\bshould\b|"
              r"\bneeds? to\b|\bmust\b|instruct\w+|\bask(?:ed)?\b|\bplease\b|let'?s|\bwe(?:'ll| will)\b|I(?:'ll| will)\b|"
              r"consider\w*|encourag\w+|\bneed for\b|due (?:for|in|on)|scheduled|arrang\w+|order(?:ed|s)?\b|placed|requested|set up|"
              r"proceed\w*|prescrib\w+|initiat\w+|\bstart(?:ed|s)?\b|\bresume\b|\bcontinue\b|\bdiscontinue\b|\brestart\b|\bhold\b")
IMPERATIVE_START = (r"^\s*(?:[-•*]\s*|\d+[.)]\s*)?(?:order|obtain|get|check|draw|repeat|recheck|send|perform|do|schedule|arrange|refer|"
                    r"start|begin|initiate|continue|stop|discontinue|hold|increase|decrease|adjust|change|switch|prescribe|refill|taper|add|"
                    r"resume|restart|follow|return|call|contact|monitor|avoid|encourage|advise|counsel|educate|discharge|admit|observe|"
                    r"recommend|consult|review|reassess|re-?evaluate|apply|use|take|keep|limit|refrain|elevate|rest|ice|proceed|"
                    r"plan|continue|trial|titrate|notify|inform|instruct|reassure|provide)\b")
TIMEFRAME = (r"(?:(?:in|within|after|over|for|every|at)\s+(?:the next\s+|about\s+|approximately\s+)?"
             r"(?:\d+\s*(?:-|to|or)\s*)?\d+\s*(?:hours?|hrs?|days?|weeks?|wks?|months?|years?)|"
             r"(?:in|within)\s+(?:a|one|two|three|four|five|six|several|a few|few)\s+(?:hours?|days?|weeks?|months?)|"
             r"(?:one|two|three|four|five|six)[- ](?:to[- ](?:one|two|three|four|five|six)[- ])?(?:hours?|days?|weeks?|months?)|"
             r"next\s+(?:visit|week|month|year|day)|tomorrow|today|tonight|this (?:week|afternoon|evening|morning)|"
             r"\bASAP\b|as soon as possible|immediately|\bstat\b|urgent(?:ly)?|emergent(?:ly)?|prior to discharge|before discharge|"
             r"at discharge|upon discharge|annually|yearly|monthly|weekly)")

# category -> list of (rule_id, compiled pattern). A category fires if any rule matches.
_DIR_LAB = rf"(?:order(?:ed|s|ing)?|obtain(?:ed)?|get|check(?:ed)?|draw|drawn|repeat(?:ed)?|recheck(?:ed)?|re-check|send|sent|perform(?:ed)?|request(?:ed)?|monitor(?:ed|ing)?|follow(?:ed)? (?:up )?with|do|run|trend|rule out|assess)"
RULES: dict[str, list[tuple[str, re.Pattern]]] = {
    "laboratory_test": [
        ("lab_verb_obj", re.compile(rf"\b{_DIR_LAB}\b[^.;:]{{0,60}}?\b(?:{LAB_OBJ})", I)),
        ("lab_obj_ordered", re.compile(rf"(?:{LAB_OBJ})[^.;:]{{0,40}}?\b(?:ordered|will be (?:ordered|obtained|checked|repeated|drawn)|to be (?:checked|repeated|obtained|drawn)|"
                                       rf"are (?:ordered|requested)|was (?:ordered|requested)|requested|pending)\b", I)),
        ("lab_will", re.compile(rf"\b(?:labs?|blood work|bloodwork)\b[^.;:]{{0,30}}\b(?:will|should)\b", I)),
    ],
    "imaging": [
        ("img_verb_obj", re.compile(rf"\b(?:order(?:ed|s|ing)?|obtain(?:ed)?|get|schedul\w+|repeat(?:ed)?|perform(?:ed)?|request(?:ed)?|arrang\w+|do|undergo|proceed with|plan(?:ned)? (?:for|on))\b[^.;:]{{0,60}}?\b(?:{IMG_OBJ})", I)),
        ("img_obj_ordered", re.compile(rf"(?:{IMG_OBJ})[^.;:]{{0,40}}?\b(?:ordered|will be (?:ordered|obtained|performed|scheduled|done|repeated)|to be (?:obtained|performed|done|repeated|scheduled)|"
                                       rf"are (?:ordered|requested)|was (?:ordered|requested)|requested|scheduled)\b", I)),
    ],
    "procedure": [
        ("proc_verb_obj", re.compile(rf"\b(?:schedul\w+|plan(?:ned)?|perform(?:ed)?|recommend\w*|undergo|proceed with|arrang\w+|will (?:require|need|undergo|have)|needs?|candidate for|"
                                     rf"set up for|prepar\w+ for)\b[^.;:]{{0,60}}?\b(?:{PROC_OBJ})", I)),
        ("proc_obj_sched", re.compile(rf"(?:{PROC_OBJ})[^.;:]{{0,40}}?\b(?:scheduled|will be (?:performed|done|scheduled)|to be (?:performed|done|scheduled)|is planned|was recommended)\b", I)),
    ],
    "appointment_referral": [
        ("referral", re.compile(r"\b(?:referred|referral|referrals|refers?|refer(?:ring)? (?:him|her|them|the patient|you|to))\b(?!\s+to (?:drug|the (?:above|following)|as))", I)),
        ("consult_spec", re.compile(rf"\b(?:consult(?:ation|ed|s|ing)?|see|seen by|follow(?:ed)?[- ]?up with|evaluation by|evaluated by|appointment with|visit with|establish care with)\b[^.;:]{{0,40}}?\b(?:{SPECIALTY})", I)),
        ("appt", re.compile(r"\b(?:schedul\w+|arrang\w+|book\w*|set up|make|made)\b[^.;:]{0,30}?\b(?:appointment|visit|follow[- ]?up)|\bappointment\b[^.;:]{0,40}?\b(?:scheduled|made|booked|arranged|set up)\b", I)),
    ],
    "medication_action": [
        ("med_verb_obj", re.compile(rf"\b(?:start(?:ed|s|ing)?|begin|began|initiat\w+|restart\w*|resum\w+|continue[sd]?|continuing|stop(?:ped)?|discontinu\w+|hold|held|titrat\w+|"
                                    rf"increas\w+|decreas\w+|adjust\w*|chang\w+|switch\w*|prescrib\w+|refill\w*|taper\w*|add(?:ed)?|treated with|given|administer\w*|trial of|placed on|"
                                    rf"take|use|apply|put on|start him on|start her on|on)\b[^.;:]{{0,70}}?(?:{MED_OBJ})", I)),
        ("med_obj_verb", re.compile(rf"(?:{MED_OBJ})[^.;:]{{0,40}}?\b(?:was|were|is|are|will be)\s+(?:started|initiated|prescribed|continued|discontinued|increased|decreased|adjusted|changed|given|held|stopped|added|refilled)\b", I)),
        ("med_review", re.compile(r"\bmedication (?:reconciliation|review|management|adjustment)\b", I)),
    ],
    "patient_communication": [
        ("comm_verb_patient", re.compile(r"\b(?:call(?:ed)?|contact(?:ed)?|notify|notified|inform(?:ed)?|counsel(?:ed|ing)?|educat\w+|instruct\w*|advis\w+|reassur\w+|let (?:him|her|them|the patient|you) know|communicat\w+|reviewed? (?:the )?(?:results?|plan)\s+with)\b[^.;:]{0,60}?\b(?:patient|family|parent|parents|mother|father|wife|husband|caregiver|you|him|her|them|results?|plan|importance|risks?|options?|about|regarding|to (?:avoid|keep|take|return|call|monitor|continue|follow|limit|stop|use|go|report|check|bring|come|seek|notify|contact|let))", I)),
        ("return_precaution", re.compile(r"\b(?:return|come back|call|contact|go to (?:the )?(?:ER|emergency)|seek (?:care|medical))\b[^.;:]{0,40}?\b(?:if|should|when|for)\b[^.;:]{0,40}?\b(?:worse|worsen\w*|persist\w*|symptoms?|fever|pain|bleeding|concern\w*|problems?|questions?|develop\w*|change\w*|not improv\w+|shortness|difficulty)", I)),
        ("results_to_patient", re.compile(r"\b(?:results?|findings)\b[^.;:]{0,40}\b(?:will be (?:called|given|sent|shared|communicated)|will (?:call|contact|notify))\b", I)),
    ],
    "followup_review": [
        ("followup_time", re.compile(rf"\b(?:follow[- ]?up|f/u|recheck|re-?check|re-?evaluat\w+|re-?assess\w*|revisit|return(?:\s+to\s+(?:the\s+)?(?:clinic|office|ED|ER))?|see (?:him|her|them|the patient|you)(?: again| back)?|come back|check[- ]?in|reviewed? (?:in|at))\b[^.;:]{{0,50}}?(?:{TIMEFRAME})", I)),
        ("followup_generic", re.compile(r"\b(?:will|should|to|plans? to|needs? to|advised to|instructed to|recommend\w* (?:that )?(?:he|she|they|the patient)?)\s*(?:follow[- ]?up|f/u|return|come back|be (?:re-?evaluated|re-?assessed|rechecked|seen))\b|\bfollow[- ]?up (?:visit|appointment|care|plan|in (?:the )?(?:clinic|office))|\b(?:to be|be) followed up\b|\bfollow[- ]?up (?:is|will be) (?:needed|required|arranged)", I)),
        ("review_results", re.compile(r"\b(?:review|reassess|re-?evaluat\w+|monitor|trend|follow)\b[^.;:]{0,30}?\b(?:results?|labs?|imaging|films?|response|progress|symptoms?|blood pressure|BP|weight|glucose|levels?|closely)\b", I)),
        ("next_visit", re.compile(r"\bnext (?:visit|appointment)\b|\bat (?:his|her|the) next\b", I)),
    ],
    "other_action": [
        ("admit_discharge", re.compile(r"\b(?:admit(?:ted)? (?:to|for)|will be admitted|admission to|discharge[d]? (?:home|to|with|on)|plan(?:ned)? (?:for )?discharge|transfer(?:red)? to|transferred)\b", I)),
        ("lifestyle_directive", re.compile(r"\b(?:encourag\w+|advis\w+|recommend\w*|instruct\w*|counsel\w*|should|avoid\w*|refrain\w* from|limit\w*|restrict\w*|maintain|follow|continue|start|begin|try)\b[^.;:]{0,50}?\b(?:diet\w*|exercise\w*|rest\b|ice\b|elevat\w+|weight loss|smoking|alcohol|salt|sodium|fluids?|hydrat\w+|activit(?:y|ies)|lifestyle|wound care|dressing|compression|physical therapy exercises|stretching|walking)\b", I)),
        ("inpatient_supportive", re.compile(r"\b(?:IV fluids?|intravenous (?:fluids|normal saline|saline)|oxygen|supportive (?:care|measures)|NPO\b|bed rest|observation|monitored? (?:closely|in the)|ICU|intensive care)\b[^.;:]{0,40}?\b(?:started|initiated|given|provided|administered|will|continue\w*|begun|placed|maintained)\b", I)),
    ],
}

NEG_GUARD = re.compile(r"\b(?:denies|denied|no (?:history|evidence|signs?) of|negative for|not (?:been )?(?:recommended|indicated|needed|required)|"
                       r"declin(?:e|ed|es)|refus(?:e|ed|es)|without (?:the need|any need))\b", I)
HIST_GUARD = re.compile(PAST_NARR, I)
RESULT_GUARD = re.compile(r"\b(?:reveal(?:s|ed|ing)?|show(?:s|ed|ing)?|demonstrat\w+|confirm(?:s|ed)|indicat(?:es|ed)|result(?:s|ed) (?:in|show|reveal|indicate)|found|diagnos(?:es|ed)|"
                          r"is positive|was positive|are positive|were positive|negative|normal|abnormal|elevated|consistent with)\b", I)
RISK_GUARD = re.compile(r"\b(?:risks? (?:include|of|are)|possible need|potential (?:risks?|complications?)|complications? (?:include|of)|side effects? (?:include|of)|may (?:cause|lead|result))\b", I)
QUESTION_OK = re.compile(r"^\s*(?:can|could|would|will) you (?:please )?(?:call|schedule|take|come|return|make|bring|continue|start|stop|get|go)\b", I)
FUT = re.compile(FUTURE_DIR, I)
IMP = re.compile(IMPERATIVE_START, I)
TIME = re.compile(TIMEFRAME, I)

CATEGORY_ORDER = ["laboratory_test", "imaging", "procedure", "appointment_referral", "medication_action",
                  "followup_review", "patient_communication", "other_action"]


@dataclass
class Proposal:
    labels: list[str] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)
    spans: dict[str, tuple[int, int]] = field(default_factory=dict)   # per-category evidence span in the sentence
    tense: str = "none"          # directive | historical | record | none
    confidence: float = 0.0
    due_phrase: str | None = None
    due_span: tuple[int, int] | None = None
    negated: bool = False
    needs_review: bool = False
    review_reason: str | None = None


def propose(text: str, section_kind: str = "narrative") -> Proposal:
    """section_kind: plan | narrative | dialogue_doctor | summary | synthea_plan"""
    p = Proposal()
    t = text.strip()
    if len(t) < 6:
        return p
    if section_kind == "dialogue_doctor" and len(t) > 300:
        p.tense = "none"          # unpunctuated ASR run-on: unreliable for span-level labels
        return p
    if t.endswith("?") and not QUESTION_OK.search(t):
        p.tense = "question"
        return p
    hits: dict[str, tuple[str, tuple[int, int]]] = {}
    for cat in CATEGORY_ORDER:
        for rid, rx in RULES[cat]:
            m = rx.search(t)
            if m:
                hits[cat] = (rid, m.span())
                break
    if not hits:
        return p
    if RISK_GUARD.search(t):
        p.tense = "record"
        return p
    if RESULT_GUARD.search(t) and section_kind != "dialogue_doctor" and not re.search(r"\b(?:will|should|recommend\w*|plan to)\b", t, I):
        p.tense = "record"       # order + result already documented => not an open instruction
        return p
    hist = bool(HIST_GUARD.search(t))
    fut = bool(FUT.search(t)) or bool(IMP.search(t))
    neg = bool(NEG_GUARD.search(t))
    in_plan = section_kind in ("plan", "dialogue_doctor", "synthea_plan")
    # tense logic: narrative sections need a forward-looking cue; plan sections accept order-style participles
    if neg and not fut:
        p.negated = True
        p.tense = "historical"
        return p
    if section_kind == "narrative" and hist and not (re.search(r"\bwill\b|\bshould\b|recommend|plan to|\bto be\b", t, I)):
        p.tense = "historical"
        return p
    if not fut and not in_plan:
        p.tense = "historical" if hist else "none"
        if hist:
            return p
        # a bare noun-phrase match with no directive cue: leave unlabeled, flag for review if multi-hit
        if len(hits) >= 2:
            p.needs_review, p.review_reason = True, "multiple_matches_without_directive_cue"
        return p
    p.tense = "directive"
    # drop 'other_action' when more specific labels exist and it is only a generic directive
    cats = [c for c in CATEGORY_ORDER if c in hits]
    if len(cats) > 1 and "other_action" in cats:
        cats.remove("other_action")
    # medication matches that are really just 'on <drug>' status statements in narrative text
    if "medication_action" in cats and section_kind == "narrative" and hist and len(cats) == 1:
        p.tense = "historical"
        return p
    p.labels = cats
    p.rules = [hits[c][0] for c in cats]
    p.spans = {c: hits[c][1] for c in cats}
    conf = 0.45
    conf += 0.20 if section_kind in ("plan", "synthea_plan") else 0.0
    conf += 0.10 if section_kind == "dialogue_doctor" else 0.0
    conf += 0.15 if IMP.search(t) or re.search(r"\bwill\b|recommend|order(?:ed)?\b|schedul", t, I) else 0.0
    conf += 0.10 if TIME.search(t) else 0.0
    conf -= 0.20 if hist else 0.0
    conf -= 0.10 if len(cats) >= 3 else 0.0
    conf -= 0.10 if len(t) > 400 else 0.0
    p.confidence = round(max(0.05, min(0.95, conf)), 2)
    m = TIME.search(t)
    if m:
        p.due_phrase, p.due_span = m.group(0), m.span()
    if p.confidence < 0.55:
        p.needs_review, p.review_reason = True, "low_rule_confidence"
    elif len(cats) >= 3:
        p.needs_review, p.review_reason = True, "three_or_more_labels"
    return p


_NUM = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "several": 7, "few": 3, "a few": 3}


def parse_due_days(phrase: str | None) -> float | None:
    """Parse a timeframe phrase to days. Returns None if not parseable (caller then imputes)."""
    if not isinstance(phrase, str) or not phrase:
        return None
    s = phrase.lower()
    if re.search(r"\b(stat|asap|as soon as possible|immediately|emergent\w*|today|tonight|this (?:morning|afternoon|evening)|prior to discharge|before discharge|at discharge|upon discharge)\b", s):
        return 0.25
    if "tomorrow" in s:
        return 1.0
    if re.search(r"urgent", s):
        return 1.0
    m = re.search(r"(\d+)\s*(?:-|to|or)\s*(\d+)\s*(hours?|hrs?|days?|weeks?|wks?|months?|years?)", s)
    if m:
        n = (int(m.group(1)) + int(m.group(2))) / 2
        unit = m.group(3)
    else:
        m = re.search(r"(\d+|a few|few|several|a|one|two|three|four|five|six|seven|eight)[- ]*(?:\w+[- ])?(hours?|hrs?|days?|weeks?|wks?|months?|years?)", s)
        if not m:
            if "next week" in s:
                return 7.0
            if "next month" in s:
                return 30.0
            if "next year" in s or "annual" in s or "yearly" in s:
                return 365.0
            if "weekly" in s:
                return 7.0
            if "monthly" in s:
                return 30.0
            if "next visit" in s or "next day" in s:
                return 1.0 if "day" in s else 30.0
            return None
        raw = m.group(1)
        n = float(raw) if raw.isdigit() else _NUM.get(raw)
        if n is None:
            return None
        unit = m.group(2)
    if unit.startswith(("hour", "hr")):
        return max(n / 24.0, 0.04)
    if unit.startswith("day"):
        return float(n)
    if unit.startswith(("week", "wk")):
        return float(n) * 7
    if unit.startswith("month"):
        return float(n) * 30
    if unit.startswith("year"):
        return float(n) * 365
    return None
