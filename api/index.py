"""
Leveling Up — Flask Backend
Fase 2: Mirror Mode + Weekly Identity Report

Endpoints:
  POST /identity/weekly-report   → Genera reporte semanal de identidad
  GET  /health                   → Health check

Setup:
  pip install flask google-genai python-dotenv
  export GMN_KEY=your_key
  python api/index.py
"""

import os
import json
from datetime import datetime
from collections import defaultdict
from flask import Flask, request, jsonify

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

app = Flask(__name__)

# -- Keys --
zai_key = os.environ["ZAI_KEY"]
opn_key = os.environ["OPN_KEY"]

# ── Gemini setup ──────────────────────────────────────────────────────────────
client = genai.Client(api_key=os.environ["GMN_KEY"])

REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "pattern_identified": {"type": "string"},
        "consistency_trend": {"type": "string"},
        "mirror_insight": {"type": "string"},
        "one_correction": {"type": "string"},
        "identity_alignment": {"type": "string"},
    },
    "required": [
        "headline",
        "pattern_identified",
        "consistency_trend",
        "mirror_insight",
        "one_correction",
        "identity_alignment",
    ],
}




# ── Health check ──────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": "1.0.0"})


@app.route("/identity/weekly-report", methods=["POST"])
def weekly_report():
    try:
        data = request.get_json(force=True)
 
        uid                = data.get("uid", "")
        identity_statement = data.get("identity_statement", "")
        roles              = data.get("roles", [])
        standards          = data.get("standards", [])
        week_entries       = data.get("week_entries", [])
 
        if not identity_statement or not week_entries:
            return jsonify({"error": "Missing identity_statement or week_entries"}), 400
 
        # ── FIX 2 & 3: Compute all numbers in Python, never in the LLM ──────
        computed = _compute_scores(roles, standards, week_entries)
 
        prompt = _build_report_prompt(
            identity_statement, roles, standards, week_entries, computed
        )
 
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=1200,
                response_mime_type="application/json",
                response_json_schema=REPORT_SCHEMA
            )
        )
 
        # With response_json_schema, the response text is already clean JSON
        report = json.loads(response.text)

 
        # Inject computed numbers — LLM never touches these
        report["uid"]          = uid
        report["overall_score"] = computed["overall_score"]
        report["role_scores"]  = computed["role_scores"]
        report["generated_at"] = datetime.utcnow().isoformat()
 
        return jsonify(report), 200
 
    except json.JSONDecodeError as e:
        return jsonify({"error": f"LLM returned invalid JSON: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
 
 
# ── Score computation (Python only, never LLM) ────────────────────────────────
def _compute_scores(roles, standards, week_entries):
    """
    Computes all numeric scores.
    Returns overall_score (0.0-1.0), daily_scores, and per-role scores.
    """
    role_map      = {r["id"]: r["name"] for r in roles}
    standard_map  = {s["id"]: s for s in standards}
 
    # Daily scores (0.0–1.0)
    daily_scores = []
    for entry in week_entries:
        total     = entry.get("standards_total", [])
        completed = entry.get("standards_completed", [])
        if total:
            daily_scores.append(len(completed) / len(total))
 
    overall_score = sum(daily_scores) / len(daily_scores) if daily_scores else 0.0
 
    # Per-role scores across the week
    role_completed = defaultdict(int)
    role_total     = defaultdict(int)
 
    for entry in week_entries:
        total_ids     = set(entry.get("standards_total", []))
        completed_ids = set(entry.get("standards_completed", []))
 
        for sid in total_ids:
            std = standard_map.get(sid)
            if not std:
                continue
            role_name = role_map.get(std.get("role_id", ""), "Unknown")
            role_total[role_name] += 1
            if sid in completed_ids:
                role_completed[role_name] += 1
 
    role_scores = {}
    for role_name, total in role_total.items():
        role_scores[role_name] = round(role_completed[role_name] / total, 2) if total else 0.0
 
    strongest_role = max(role_scores, key=role_scores.get) if role_scores else ""
    weakest_role   = min(role_scores, key=role_scores.get) if role_scores else ""
 
    return {
        "overall_score":  round(overall_score, 2),
        "daily_scores":   [round(s, 2) for s in daily_scores],
        "role_scores":    role_scores,
        "strongest_role": strongest_role,
        "weakest_role":   weakest_role
    }
 
 
# ── Prompt builder ────────────────────────────────────────────────────────────
def _build_report_prompt(identity_statement, roles, standards, week_entries, computed):
    """
    FIX 1: Standards grouped by role for real analytical context.
    FIX 3: Numbers pre-computed and injected — LLM told not to recalculate.
    FIX 4: Prompt is honest and confrontational, not generic.
    FIX 5: Asks explicitly for consistency pattern analysis.
    """
    role_map      = {r["id"]: r["name"] for r in roles}
    standard_map  = {s["id"]: s["title"] for s in standards}
 
    # FIX 1 — Group standards by role
    standards_by_role = defaultdict(list)
    for s in standards:
        role_name = role_map.get(s.get("role_id", ""), "Unknown")
        standards_by_role[role_name].append(s["title"])
 
    standards_section = "STANDARDS BY ROLE:\n" + "\n".join(
        f"  - {role}: {', '.join(stds)}"
        for role, stds in standards_by_role.items()
    )
 
    # FIX 2 — Daily breakdown with pre-computed scores
    days_lines = []
    for i, entry in enumerate(week_entries):
        date      = entry.get("date", f"Day {i+1}")
        completed = [standard_map.get(sid, sid) for sid in entry.get("standards_completed", [])]
        total     = [standard_map.get(sid, sid) for sid in entry.get("standards_total", [])]
        score     = computed["daily_scores"][i] if i < len(computed["daily_scores"]) else 0.0
 
        answers_text = ""
        for a in entry.get("evening_answers", []):
            q = a.get("question", "")
            ans = a.get("answer", "")
            if ans.strip():
                answers_text += f"\n    Q: {q}\n    A: {ans}"
 
        days_lines.append(
            f"  {date} | score: {score} ({len(completed)}/{len(total)} standards)\n"
            f"  Completed: {', '.join(completed) if completed else 'NONE'}\n"
            f"  Missing:   {', '.join(t for t in total if t not in completed) or 'none'}"
            + (f"\n  Journal:{answers_text}" if answers_text else "")
        )
 
    days_section = "\n\n".join(days_lines)
 
    role_scores_text = "\n".join(
        f"  - {role}: {int(score * 100)}%"
        for role, score in computed["role_scores"].items()
    )
 
    return f"""
You are a high-performance identity coach.
 
Your job is NOT to motivate.
Your job is to reveal truth, patterns, and misalignment.
 
Be honest, precise, and slightly confrontational when necessary.
Avoid generic advice. Base everything strictly on the data below.
Do NOT recalculate any scores. Use the precomputed values provided.
 
════════════════════════════════════════
IDENTITY: "{identity_statement}"
 
{standards_section}
 
PRE-COMPUTED SCORES (do NOT recalculate):
  Overall week score: {computed["overall_score"]} (0.0–1.0)
  Strongest role: {computed["strongest_role"]}
  Weakest role:   {computed["weakest_role"]}
  By role:
{role_scores_text}
 
DAILY BREAKDOWN:
{days_section}
════════════════════════════════════════
 
INSTRUCTIONS:
1. Identify consistency patterns: did performance improve, decline, or stay unstable across the week?
2. Find repeated failures in specific standards — name them explicitly.
3. Synthesize the Mirror Mode answers (if any) to reveal self-perception patterns.
4. Be specific — no generic phrases like "keep it up" or "you did great".
5. strongest_role and weakest_role must match the precomputed values exactly.
 
Respond ONLY with valid JSON, no markdown. Use this exact structure:
{{
  "headline": "<one powerful, honest sentence summarizing the week>",
  "pattern_identified": "<specific behavioral pattern from the data — name the exact standards that failed repeatedly>",
  "consistency_trend": "<improving | declining | unstable | consistent> — and explain why in one sentence>",
  "mirror_insight": "<what their journal answers reveal about their self-perception — be direct>",
  "one_correction": "<the single most impactful change for next week — specific and actionable>",
  "identity_alignment": "<honest verdict: are they living their declared identity? What is the gap?>"
}}
"""
 
 
if __name__ == "__main__":
    app.run(debug=True, port=5000)