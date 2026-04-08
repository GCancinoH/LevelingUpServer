"""
Leveling Up — Flask Backend
Fase 3: Mirror Mode + Weekly Identity Report + LLM-generated corrective quest.

Endpoints:
  POST /identity/weekly-report   → Genera reporte semanal de identidad + quest
  GET  /health                   → Health check

✔ Roles used analytically (standards grouped by role)
✔ Numbers computed in Python — LLM never calculates scores
✔ Honest, confrontational prompt
✔ Consistency pattern analysis
✔ Structured generated_quest output
✔ valid_standard_ids prevents LLM hallucination
✔ preferred_quest_type derived from failure pattern

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
        "generated_quest": {
            "type": "object",
            "properties": {
                "title":               {"type": "string"},
                "description":         {"type": "string"},
                "type":                {"type": "string"},
                "target_standard_ids": {"type": "array", "items": {"type": "string"}},
                "goal":                {"type": "integer"},
                "duration_days":       {"type": "integer"},
            },
            "required": ["title", "description", "type", "target_standard_ids", "goal", "duration_days"],
        },
    },
    "required": [
        "headline",
        "pattern_identified",
        "consistency_trend",
        "mirror_insight",
        "one_correction",
        "identity_alignment",
        "generated_quest",
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
 
        # Compute all numbers in Python — LLM never calculates scores
        computed             = _compute_scores(roles, standards, week_entries)
        preferred_quest_type = _derive_quest_type(computed, week_entries)
        valid_standard_ids   = [s["id"] for s in standards]
 
        prompt = _build_report_prompt(
            identity_statement, roles, standards, week_entries,
            computed, preferred_quest_type, valid_standard_ids
        )
 
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=1400,
                response_mime_type="application/json",
                response_json_schema=REPORT_SCHEMA
            )
        )
 
        # With response_json_schema, the response text is already clean JSON
        report = json.loads(response.text)
 
        # Inject computed numbers — LLM never touches these
        report["uid"]           = uid
        report["overall_score"] = computed["overall_score"]
        report["role_scores"]   = computed["role_scores"]
        report["generated_at"]  = datetime.utcnow().isoformat()
 
        # Validate and sanitize quest standard IDs — prevent LLM hallucination
        quest = report.get("generated_quest")
        if quest:
            quest["target_standard_ids"] = [
                sid for sid in quest.get("target_standard_ids", [])
                if sid in valid_standard_ids
            ]
            if not quest["target_standard_ids"]:
                quest["target_standard_ids"] = _most_failed(standards, week_entries)[:2]
 
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
 
    daily_scores   = []
    role_completed = defaultdict(int)
    role_total     = defaultdict(int)
 
    for entry in week_entries:
        total_ids     = entry.get("standards_total", [])
        completed_ids = set(entry.get("standards_completed", []))
 
        if total_ids:
            daily_scores.append(len(completed_ids) / len(total_ids))
 
        for sid in total_ids:
            std = standard_map.get(sid)
            if not std:
                continue
            role_name = role_map.get(std.get("role_id", ""), "Unknown")
            role_total[role_name] += 1
            if sid in completed_ids:
                role_completed[role_name] += 1
 
    overall_score = sum(daily_scores) / len(daily_scores) if daily_scores else 0.0
    role_scores = {
        role: round(role_completed[role] / total, 2)
        for role, total in role_total.items() if total
    }
 
    return {
        "overall_score":  round(overall_score, 2),
        "daily_scores":   [round(s, 2) for s in daily_scores],
        "role_scores":    role_scores,
        "strongest_role": max(role_scores, key=role_scores.get) if role_scores else "",
        "weakest_role":   min(role_scores, key=role_scores.get) if role_scores else ""
    }
 
 
def _derive_quest_type(computed, week_entries) -> str:
    """Derives quest type from failure frequency."""
    failures   = sum(1 for e in week_entries if
                     len(e.get("standards_completed", [])) < len(e.get("standards_total", [])))
    total_days = len(week_entries)
    if not total_days:
        return "CONSISTENCY"
    ratio = failures / total_days
    if ratio >= 0.8:    return "CONSISTENCY"  # failed almost every day
    elif ratio <= 0.28: return "ELIMINATION"  # only 1-2 failures
    else:               return "STREAK"       # inconsistent pattern
 
 
def _most_failed(standards, week_entries) -> list:
    """Returns standard IDs sorted by failure count descending."""
    counts = defaultdict(int)
    for entry in week_entries:
        completed = set(entry.get("standards_completed", []))
        for sid in entry.get("standards_total", []):
            if sid not in completed:
                counts[sid] += 1
    return sorted(counts, key=counts.get, reverse=True)
 
 
# ── Prompt builder ────────────────────────────────────────────────────────────
def _build_report_prompt(identity_statement, roles, standards, week_entries,
                         computed, preferred_quest_type, valid_standard_ids):
    """
    Builds the LLM prompt with:
    - Standards grouped by role (with IDs for anti-hallucination)
    - Pre-computed scores injected — LLM told not to recalculate
    - Honest, confrontational tone
    - Consistency pattern analysis
    - Quest generation instructions with valid_standard_ids guard
    """
    role_map      = {r["id"]: r["name"] for r in roles}
    standard_map  = {s["id"]: s["title"] for s in standards}
 
    # Group standards by role, include IDs for the LLM to reference in quest
    standards_by_role = defaultdict(list)
    for s in standards:
        role_name = role_map.get(s.get("role_id", ""), "Unknown")
        standards_by_role[role_name].append(f"{s['title']} (id:{s['id']})")
 
    standards_section = "STANDARDS BY ROLE:\n" + "\n".join(
        f"  [{role}]: {', '.join(stds)}"
        for role, stds in standards_by_role.items()
    )
 
    # Daily breakdown with pre-computed scores
    days_lines = []
    for i, entry in enumerate(week_entries):
        date      = entry.get("date", f"Day {i+1}")
        completed = [standard_map.get(sid, sid) for sid in entry.get("standards_completed", [])]
        missing   = [standard_map.get(sid, sid)
                     for sid in entry.get("standards_total", [])
                     if sid not in entry.get("standards_completed", [])]
        score     = computed["daily_scores"][i] if i < len(computed["daily_scores"]) else 0.0
 
        answers_text = "".join(
            f"\n    Q: {a.get('question', '')}\n    A: {a.get('answer', '')}"
            for a in entry.get("evening_answers", []) if a.get("answer", "").strip()
        )
        days_lines.append(
            f"  {date} | {score} ({len(completed)}/{len(entry.get('standards_total', []))})\n"
            f"  ✓ {', '.join(completed) or 'NONE'}\n"
            f"  ✗ {', '.join(missing) or 'none'}"
            + (f"\n  Journal:{answers_text}" if answers_text else "")
        )
 
    role_scores_text = "\n".join(
        f"  [{role}]: {int(score * 100)}%"
        for role, score in computed["role_scores"].items()
    )
 
    return f"""
You are a high-performance identity coach.
Your job is NOT to motivate. Reveal truth, patterns, and misalignment.
Be honest, precise, and slightly confrontational.
Base everything on the data. Do NOT recalculate scores.
 
════════════════════════════════════════
IDENTITY: "{identity_statement}"
 
{standards_section}
 
SCORES (precomputed — do NOT change):
  Overall: {computed["overall_score"]}
  Strongest: {computed["strongest_role"]}
  Weakest:   {computed["weakest_role"]}
{role_scores_text}
 
DAILY DATA:
{chr(10).join(days_lines)}
════════════════════════════════════════
 
INSTRUCTIONS:
1. Identify if performance improved, declined, or stayed unstable.
2. Name the EXACT standards that failed repeatedly.
3. Synthesize Mirror Mode journal answers if present.
4. Generate ONE corrective quest targeting the most-failed standards.
 
QUEST RULES:
- STREAK: complete standard X days in a row
- CONSISTENCY: complete standard N times out of durationDays
- ELIMINATION: zero failures in durationDays
- Preferred type this week: {preferred_quest_type}
- ONLY use these IDs (do NOT invent): [{', '.join(valid_standard_ids)}]
- goal must be an integer ≤ durationDays
- durationDays must be 5, 7, or 14
 
Respond ONLY with valid JSON, no markdown. Use this exact structure:
{{
  "headline": "<one powerful honest sentence>",
  "pattern_identified": "<specific — name exact failing standards>",
  "consistency_trend": "<improving|declining|unstable|consistent> — one sentence why",
  "mirror_insight": "<what journal reveals about self-perception>",
  "one_correction": "<single most impactful change — specific>",
  "identity_alignment": "<honest verdict on living the declared identity>",
  "generated_quest": {{
    "title": "<short mission title>",
    "description": "<exactly what to do>",
    "type": "<STREAK|CONSISTENCY|ELIMINATION>",
    "target_standard_ids": ["<valid IDs only>"],
    "goal": <int>,
    "duration_days": <5|7|14>
  }}
}}
"""
 
 
if __name__ == "__main__":
    app.run(debug=True, port=5000)