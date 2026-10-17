import json, sys, time
from pathlib import Path
import llama_cpp
from llama_cpp import Llama
ROOT = Path(__file__).resolve().parent.parent
print("llama_cpp", llama_cpp.__version__); print(llama_cpp.llama_print_system_info().decode()[:400])
schema = {"type": "object", "properties": {"obligations": {"type": "array", "items": {"type": "object", "properties": {
    "task_type": {"type": "string", "enum": ["laboratory_test", "imaging", "procedure", "appointment_referral", "medication_action", "patient_communication", "followup_review", "other_action"]},
    "description": {"type": "string"}, "due_interval": {"type": ["string", "null"]}, "responsible_role": {"type": ["string", "null"]},
    "depends_on_index": {"type": ["integer", "null"]}, "conditional": {"type": ["string", "null"]}, "evidence_span": {"type": "string"}},
    "required": ["task_type", "description", "due_interval", "responsible_role", "depends_on_index", "conditional", "evidence_span"]}}}, "required": ["obligations"]}
text = "Patient should undergo repeat blood testing in two days. Results must be reviewed by the treating physician and follow-up arranged if abnormalities persist."
sys_p = ("You convert clinical instructions into a JSON list of follow-up obligations. Only use facts stated in the text. evidence_span must be copied verbatim from the text. "
         "Use null when a deadline, role or dependency is not stated. depends_on_index is the 0-based index of an earlier obligation this one depends on.")
for name in sys.argv[1:]:
    t = time.time(); llm = Llama(model_path=str(ROOT / "models/llm" / name), n_ctx=2048, n_threads=2, verbose=False); print(name, "load", round(time.time() - t, 2), "s", flush=True)
    for i in range(2):
        t = time.time()
        out = llm.create_chat_completion(messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": text}],
                                         response_format={"type": "json_object", "schema": schema}, temperature=0.0, max_tokens=400)
        dt = time.time() - t; u = out["usage"]
        print(f"{dt:.1f}s  prompt={u['prompt_tokens']} gen={u['completion_tokens']}  {u['completion_tokens']/dt:.1f} tok/s", flush=True)
    print(out["choices"][0]["message"]["content"][:700], flush=True)
