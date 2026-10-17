"""Evaluation corpora for obligation extraction.

1. CHALLENGE_SET - 40 hand-written instruction texts annotated BY THE SYSTEM DEVELOPER. This is NOT human-reviewed gold from
   independent clinicians: annotator = implementer, so scores on it carry annotator-bias risk. It is kept strictly separate from
   the rule development examples (rules were tuned on the dataset-pipeline text, not on these).
2. generate_template_docs(...) - synthetic documents whose ground truth is exact BY CONSTRUCTION. Two phrasing banks:
   'train' (used for nothing but sanity checks) and 'heldout' (novel sentence structures, reserved for testing).

Annotation conventions (docs/linuxone_setup.md): one obligation per distinct required action; task_type per ontology
(reviewing results = followup_review; arranging/booking an appointment, referral or 'follow up with <provider>' = appointment_referral;
revisit timing without a scheduling verb = followup_review; instructions to the patient = patient_communication); due_interval only
if a relative deadline is explicitly stated for that action (durations such as 'for ten days' are not deadlines; ranges use the upper
bound); role only if the text names who must act; dependency = the action needs an earlier one (review <- test, notify <- review).
"""
from __future__ import annotations

import random

LAB, IMG, PROC, REF, MED, COM, FUP, OTH = ("laboratory_test", "imaging", "procedure", "appointment_referral", "medication_action",
                                            "patient_communication", "followup_review", "other_action")


def E(task_type, span, due=None, role=None, deps=(), cond=None):
    return {"task_type": task_type, "span": span, "due_interval": due, "role": role, "depends_on": list(deps), "conditional": cond}


CHALLENGE_SET = [
    ("He was discharged in stable condition. Repeat the CBC in 48 hours and have the treating physician review the result. The nurse should call him if hemoglobin is below 8.",
     [E(LAB, "Repeat the CBC in 48 hours", "PT48H"), E(FUP, "have the treating physician review the result", None, "ordering_clinician", [0]),
      E(COM, "The nurse should call him if hemoglobin is below 8", None, "nurse", [1], "if hemoglobin is below 8")]),
    ("Chest x-ray to be repeated in one week to confirm resolution of the infiltrate. Follow up with the primary care provider after the report is available.",
     [E(IMG, "Chest x-ray to be repeated in one week", "P1W"), E(REF, "Follow up with the primary care provider after the report is available", None, None, [0])]),
    ("Start amoxicillin 500 mg three times daily for ten days. Return to the clinic in 2 weeks. Advise the patient to seek emergency care if breathing worsens.",
     [E(MED, "Start amoxicillin 500 mg three times daily for ten days"), E(FUP, "Return to the clinic in 2 weeks", "P2W"),
      E(COM, "Advise the patient to seek emergency care if breathing worsens", None, None, [], "if breathing worsens")]),
    ("MRI of the lumbar spine has been ordered and should be completed within 2 weeks. The neurosurgery team will review the images and contact the patient.",
     [E(IMG, "MRI of the lumbar spine has been ordered and should be completed within 2 weeks", "P2W"),
      E(FUP, "The neurosurgery team will review the images", None, "specialist", [0]), E(COM, "contact the patient", None, None, [1])]),
    ("Refer to cardiology for evaluation of the murmur.", [E(REF, "Refer to cardiology for evaluation of the murmur")]),
    ("The patient is asked to bring a list of home medications to the next visit. Colonoscopy is recommended in 3 months once the antibiotic course is complete.",
     [E(OTH, "The patient is asked to bring a list of home medications to the next visit", None, "patient"),
      E(PROC, "Colonoscopy is recommended in 3 months once the antibiotic course is complete", "P3M", None, [], "once the antibiotic course is complete")]),
    ("Check potassium and creatinine tomorrow morning and adjust the furosemide dose according to the result.",
     [E(LAB, "Check potassium and creatinine tomorrow morning", "P1D"), E(MED, "adjust the furosemide dose according to the result", None, None, [0])]),
    ("The patient reports mild headache for two days. No focal deficits on examination. Vital signs are stable.", []),
    ("She was referred to endocrinology last year and was started on metformin. She denies polyuria today.", []),
    ("Stop lisinopril today because of the cough.", [E(MED, "Stop lisinopril today because of the cough", "PT24H")]),
    ("Schedule a follow-up visit in 6 weeks to review the biopsy results.", [E(REF, "Schedule a follow-up visit in 6 weeks to review the biopsy results", "P6W")]),
    ("Please obtain an echocardiogram before the next clinic visit. The cardiologist will review the study and decide on further management.",
     [E(IMG, "Please obtain an echocardiogram before the next clinic visit"), E(FUP, "The cardiologist will review the study", None, "specialist", [0])]),
    ("Hold metformin for 48 hours after the contrast study.", [E(MED, "Hold metformin for 48 hours after the contrast study")]),
    ("The nurse will phone the patient in 3 days to check on symptoms.", [E(COM, "The nurse will phone the patient in 3 days to check on symptoms", "P3D", "nurse")]),
    ("Discussed risks and benefits of surgery. Patient understands and agrees. No questions at this time.", []),
    ("Repeat the thyroid panel in 6 weeks after the dose change. The endocrinologist will review it.",
     [E(LAB, "Repeat the thyroid panel in 6 weeks after the dose change", "P6W"), E(FUP, "The endocrinologist will review it", None, "specialist", [0])]),
    ("Refer to physical therapy twice weekly for 4 weeks; reassess strength in 4 weeks.",
     [E(REF, "Refer to physical therapy twice weekly for 4 weeks"), E(FUP, "reassess strength in 4 weeks", "P4W")]),
    ("A urine culture was sent from the emergency department. Please follow up the result and adjust antibiotics if the organism is resistant.",
     [E(FUP, "Please follow up the result"), E(MED, "adjust antibiotics if the organism is resistant", None, None, [0], "if the organism is resistant")]),
    ("Are you currently taking aspirin? Do you have any chest pain with exertion?", []),
    ("Pharmacist to reconcile the discharge medications before the patient leaves.", [E(MED, "Pharmacist to reconcile the discharge medications before the patient leaves", None, "pharmacy")]),
    ("Ultrasound of the right calf today to rule out DVT. Start anticoagulation if positive.",
     [E(IMG, "Ultrasound of the right calf today to rule out DVT", "PT24H"), E(MED, "Start anticoagulation if positive", None, None, [0], "if positive")]),
    ("Biopsy of the skin lesion in 2 weeks. Dermatology will call with an appointment.",
     [E(PROC, "Biopsy of the skin lesion in 2 weeks", "P2W"), E(REF, "Dermatology will call with an appointment", None, "specialist")]),
    ("No follow-up is needed.", []),
    ("Counsel the patient on smoking cessation and provide written materials.", [E(COM, "Counsel the patient on smoking cessation and provide written materials")]),
    ("Review the pending blood cultures daily until final. Continue vancomycin until the cultures are negative at 48 hours.",
     [E(FUP, "Review the pending blood cultures daily until final"), E(MED, "Continue vancomycin until the cultures are negative at 48 hours")]),
    ("Mammogram due in 1 year.", [E(IMG, "Mammogram due in 1 year", "P1Y")]),
    ("Arrange a home-health nursing visit within 3 days.", [E(REF, "Arrange a home-health nursing visit within 3 days", "P3D")]),
    ("Wound check at the clinic in 5 days. Remove sutures in 10 days.", [E(FUP, "Wound check at the clinic in 5 days", "P5D"), E(PROC, "Remove sutures in 10 days", "P10D")]),
    ("Please ensure the patient receives the influenza vaccine before discharge.", [E(PROC, "ensure the patient receives the influenza vaccine before discharge")]),
    ("The family declined admission and will seek care elsewhere. The patient left against medical advice.", []),
    ("Oncology to review the pathology within 1 week and communicate the plan to the patient.",
     [E(FUP, "Oncology to review the pathology within 1 week", "P1W", "specialist"), E(COM, "communicate the plan to the patient", None, None, [0])]),
    ("If the fever returns, call the clinic. Otherwise, follow up in 3 months.",
     [E(COM, "If the fever returns, call the clinic", None, None, [], "If the fever returns"), E(FUP, "follow up in 3 months", "P3M")]),
    ("Order a lipid panel and hemoglobin A1c with the next blood draw.", [E(LAB, "Order a lipid panel and hemoglobin A1c with the next blood draw")]),
    ("He should avoid NSAIDs and arrange a nephrology appointment within 10 days.",
     [E(OTH, "He should avoid NSAIDs"), E(REF, "arrange a nephrology appointment within 10 days", "P10D")]),
    ("Send the sputum culture today.", [E(LAB, "Send the sputum culture today", "PT24H")]),
    ("Titrate insulin glargine by 2 units every 3 days until fasting glucose is below 130.", [E(MED, "Titrate insulin glargine by 2 units every 3 days until fasting glucose is below 130")]),
    ("Echo from last month showed an EF of 55%. No new orders.", []),
    ("Mother states the child had a fever yesterday and was given acetaminophen.", []),
    ("Continue home oxygen at 2 L/min. Pulmonary rehab referral was placed in March.", [E(OTH, "Continue home oxygen at 2 L/min")]),
    ("Results of the stress test will be discussed with the patient at the next visit.", [E(COM, "Results of the stress test will be discussed with the patient at the next visit")]),
]


def challenge_docs() -> list[dict]:
    out = []
    for i, (text, obs) in enumerate(CHALLENGE_SET):
        o2, pos = [], 0
        for ob in obs:
            s = text.find(ob["span"])
            if s < 0:
                raise ValueError(f"challenge doc {i}: span not found: {ob['span']!r}")
            o2.append({**ob, "start": s, "end": s + len(ob["span"])})
        out.append({"doc_id": f"challenge-{i:02d}", "text": text, "obligations": o2, "source": "developer_annotated_challenge_set"})
    return out


# ----------------------------------------------------------------------------------------------- template corpora
LABS = ["CBC", "basic metabolic panel", "hemoglobin A1c", "lipid panel", "TSH", "urinalysis", "liver function tests", "INR", "serum creatinine", "troponin"]
IMGS = ["chest x-ray", "CT of the abdomen", "MRI of the brain", "renal ultrasound", "echocardiogram", "screening mammogram", "venous doppler"]
SPECS = ["cardiology", "neurology", "gastroenterology", "endocrinology", "dermatology", "nephrology", "ophthalmology"]
REASONS = ["chest pain", "persistent headaches", "abnormal liver enzymes", "poorly controlled diabetes", "a changing mole", "declining kidney function"]
DRUGS = ["lisinopril 10 mg daily", "metformin 500 mg twice daily", "atorvastatin 40 mg nightly", "amoxicillin 500 mg three times daily", "sertraline 50 mg daily"]
DEADLINES = [("within 48 hours", "PT48H"), ("in two days", "P2D"), ("within one week", "P1W"), ("in 2 weeks", "P2W"), ("in 3 months", "P3M"),
             ("within 24 hours", "PT24H"), ("in six weeks", "P6W"), ("within 72 hours", "PT72H"), ("in 10 days", "P10D")]
ROLE_PH = [("treating physician", "ordering_clinician"), ("attending physician", "ordering_clinician"), ("nurse", "nurse"), ("primary care provider", "ordering_clinician")]
IMG_ROLE_PH = [("radiologist", "radiology"), ("treating physician", "ordering_clinician"), ("cardiologist", "specialist")]
DISTRACT = ["She denies chest pain.", "He was seen last year for knee pain.", "Blood pressure was 128/76 on arrival.", "Past medical history includes diabetes and hypertension.",
            "Do you take any medications at home?", "The CT showed no acute findings.", "She was started on lisinopril last month.", "Vital signs are stable."]


def _cap(s: str) -> str:
    return s[0].upper() + s[1:]


def _lrn(r, bank):
    lab, (dl, iso), (role, rc) = r.choice(LABS), r.choice(DEADLINES), r.choice(ROLE_PH)
    if bank == "train":
        v = r.randrange(4)
        if v == 0:
            s = [f"Obtain a {lab} {dl}", f"The {role} should review the result", "Contact the patient if the result is abnormal"]
            return s, [E(LAB, s[0], iso), E(FUP, s[1], None, rc, [0]), E(COM, s[2], None, None, [1], "if the result is abnormal")]
        if v == 1:
            s = [f"Please repeat the {lab} {dl}", f"The {role} to review the findings", "Notify the patient if values remain abnormal"]
            return s, [E(LAB, s[0], iso), E(FUP, s[1], None, rc, [0]), E(COM, s[2], None, None, [1], "if values remain abnormal")]
        if v == 2:
            s = [f"Draw a {lab} {dl}, then have the {role} review it"]
            return s, [E(LAB, f"Draw a {lab} {dl}", iso), E(FUP, f"have the {role} review it", None, rc, [0])]
        s = [f"Recheck {lab} {dl}", f"Results go to the {role} for review"]
        return s, [E(LAB, s[0], iso), E(FUP, s[1], None, rc, [0])]
    v = r.randrange(5)
    if v == 0:
        s = [f"A repeat {lab} is required {dl}, after which the {role} must examine the result and inform the patient should any abnormality persist"]
        return s, [E(LAB, f"A repeat {lab} is required {dl}", iso), E(FUP, f"the {role} must examine the result", None, rc, [0]),
                   E(COM, "inform the patient should any abnormality persist", None, None, [1], "should any abnormality persist")]
    if v == 1:
        s = [f"Blood work ({lab}) is to be repeated {dl}", f"Review of the result falls to the {role}", "The patient is to be called when the result is outside the normal range"]
        return s, [E(LAB, s[0], iso), E(FUP, s[1], None, rc, [0]), E(COM, s[2], None, None, [1], "when the result is outside the normal range")]
    if v == 2:
        s = [f"{_cap(lab)} {dl} please", f"Result to be reviewed by the {role}; the nurse will phone the patient if it is abnormal"]
        return s, [E(LAB, s[0], iso), E(FUP, f"Result to be reviewed by the {role}", None, rc, [0]), E(COM, "the nurse will phone the patient if it is abnormal", None, "nurse", [1], "if it is abnormal")]
    if v == 3:
        s = [f"Ensure a {lab} is obtained {dl} and the {role} signs off on the result"]
        return s, [E(LAB, f"Ensure a {lab} is obtained {dl}", iso), E(FUP, f"the {role} signs off on the result", None, rc, [0])]
    s = [f"Follow-up bloods: {lab} {dl}", f"The {role} is responsible for reviewing it and, unless normal, contacting the patient"]
    return s, [E(LAB, s[0], iso), E(FUP, f"The {role} is responsible for reviewing it", None, rc, [0]), E(COM, "contacting the patient", None, None, [1], "unless normal")]


def _img(r, bank):
    img, (dl, iso), (role, rc) = r.choice(IMGS), r.choice(DEADLINES), r.choice(IMG_ROLE_PH)
    if bank == "train":
        if r.random() < 0.5:
            s = [f"Schedule a {img} {dl}", f"The {role} will review the report"]
            return s, [E(IMG, s[0], iso), E(FUP, s[1], None, rc, [0])]
        s = [f"Obtain a {img} {dl} and have the {role} review the report"]
        return s, [E(IMG, f"Obtain a {img} {dl}", iso), E(FUP, f"have the {role} review the report", None, rc, [0])]
    if r.random() < 0.5:
        s = [f"A {img} should be performed {dl}; the {role} is to interpret the findings"]
        return s, [E(IMG, f"A {img} should be performed {dl}", iso), E(FUP, f"the {role} is to interpret the findings", None, rc, [0])]
    s = [f"Imaging ({img}) is requested {dl}", f"The report must be acknowledged by the {role}"]
    return s, [E(IMG, s[0], iso), E(FUP, s[1], None, rc, [0])]


def _ref(r, bank):
    sp, rs, (dl, iso) = r.choice(SPECS), r.choice(REASONS), r.choice(DEADLINES)
    if bank == "train":
        v = r.randrange(3)
        if v == 0:
            s = f"Refer to {sp} for {rs}"
            return [s], [E(REF, s)]
        if v == 1:
            s = f"Please arrange a {sp} consultation {dl}"
            return [s], [E(REF, s, iso)]
        s = f"Schedule an appointment with {sp} {dl}"
        return [s], [E(REF, s, iso)]
    v = r.randrange(3)
    if v == 0:
        s = f"The patient needs to be seen by {sp} {dl} regarding {rs}"
        return [s], [E(REF, s, iso)]
    if v == 1:
        s = f"A referral to {sp} for {rs} is recommended"
        return [s], [E(REF, s)]
    s = f"Book a visit with {sp} {dl}"
    return [s], [E(REF, s, iso)]


def _fup(r, bank):
    (dl, iso) = r.choice(DEADLINES)
    if bank == "train":
        v = r.randrange(3)
        if v == 0:
            s = f"Follow up {dl}"
            return [s], [E(FUP, s, iso)]
        if v == 1:
            s = f"Return to clinic {dl} for reassessment"
            return [s], [E(FUP, s, iso)]
        s = f"Arrange a follow-up appointment {dl}"
        return [s], [E(REF, s, iso)]
    v = r.randrange(3)
    if v == 0:
        s = f"The patient should be reviewed in clinic {dl}"
        return [s], [E(FUP, s, iso)]
    if v == 1:
        s = f"A repeat visit is needed {dl}"
        return [s], [E(FUP, s, iso)]
    s = f"Recheck {dl} to reassess symptoms"
    return [s], [E(FUP, s, iso)]


def _med(r, bank):
    d = r.choice(DRUGS)
    if bank == "train":
        s = r.choice([f"Start {d}", f"Continue {d}", f"Increase the dose of {d}", f"Refill {d}"])
    else:
        s = r.choice([f"The patient is to begin {d}", f"{_cap(d)} should be continued", f"Please prescribe {d}", f"Discontinue {d}"])
    return [s], [E(MED, s)]


BLOCKS = [_lrn, _lrn, _img, _ref, _fup, _med]


def generate_template_docs(n: int, seed: int = 0, bank: str = "heldout") -> list[dict]:
    """Documents with exact ground truth. bank='heldout' uses sentence structures never used by bank='train'."""
    r = random.Random(f"{seed}-{bank}")
    docs = []
    for k in range(n):
        parts, all_obs, deps_off = [], [], 0
        for _ in range(r.randint(1, 3)):
            if r.random() < 0.4:
                parts.append((r.choice(DISTRACT), []))
            sents, obs = r.choice(BLOCKS)(r, bank)
            parts.append((". ".join(sents) + ".", obs))
        if r.random() < 0.5:
            parts.append((r.choice(DISTRACT), []))
        text, obligations, pos = "", [], 0
        for ptxt, obs in parts:
            base = len(obligations)
            text = (text + " " + ptxt) if text else ptxt
            for ob in obs:
                s = text.find(ob["span"], pos)
                if s < 0:
                    s = text.find(ob["span"])
                obligations.append({**ob, "depends_on": [d + base for d in ob["depends_on"]], "start": s, "end": s + len(ob["span"])})
                pos = s
        docs.append({"doc_id": f"tpl-{bank}-{k:04d}", "text": text, "obligations": obligations, "source": f"template_{bank}"})
    return docs
