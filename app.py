"""
Leveling Up — Flask Backend
Fase 2: Mirror Mode + Weekly Identity Report

Endpoints:
  POST /identity/weekly-report   → Genera reporte semanal de identidad
  GET  /health                   → Health check

Setup:
  pip install flask google-generativeai python-dotenv
  export GEMINI_API_KEY=your_key
  python app.py
"""

import os
import json
from datetime import datetime
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from google import genai

load_dotenv()

app = Flask(__name__)

# Keys
zai_key = os.environ["ZAI_KEY"]
opn_key = os.environ["OPN_KEY"]

# ── Gemini setup ──────────────────────────────────────────────────────────────
client = genai.Client(api_key=os.environ["GMN_KEY"])


# ── Health check ──────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": "1.0.0"})


# ── Weekly Identity Report ─────────────────────────────────────────────────────
@app.route("/identity/weekly-report", methods=["POST"])
def weekly_report():
    """
    Expects JSON body:
    {
      "uid": "user_id",
      "identity_statement": "I am a disciplined athlete",
      "roles": [
        {"id": "r1", "name": "Athlete"},
        {"id": "r2", "name": "Trader"}
      ],
      "standards": [
        {"id": "s1", "title": "Train", "type": "TRAINING", "role_id": "r1", "xp": 25},
        {"id": "s2", "title": "Eat well", "type": "NUTRITION", "role_id": "r1", "xp": 20}
      ],
      "week_entries": [
        {
          "date": "2026-03-30",
          "standards_completed": ["s1"],
          "standards_total": ["s1", "s2"],
          "evening_answers": [
            {"question_id": "e_anchor_1", "question": "Did I act like the person I said I would be?", "answer": "Mostly yes..."},
            {"question_id": "e_mirror",   "question": "Mirror Mode...", "answer": "They would say..."}
          ]
        }
      ]
    }
    """
    try:
        data = request.get_json(force=True)

        uid                = data.get("uid", "")
        identity_statement = data.get("identity_statement", "")
        roles              = data.get("roles", [])
        standards          = data.get("standards", [])
        week_entries       = data.get("week_entries", [])

        if not identity_statement or not week_entries:
            return jsonify({"error": "Missing identity_statement or week_entries"}), 400

        prompt = _build_report_prompt(
            identity_statement, roles, standards, week_entries
        )

        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            config={
                "temperature": 0.7,
                "max_output_tokens": 1024,
                "response_mime_type": "application/json"
            }
        )

        raw = response.text.strip()

        # Gemini sometimes wraps in ```json ... ``` even with mime type set
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        report = json.loads(raw)
        report["uid"]         = uid
        report["generated_at"] = datetime.utcnow().isoformat()

        return jsonify(report), 200

    except json.JSONDecodeError as e:
        return jsonify({"error": f"LLM returned invalid JSON: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Prompt builder ─────────────────────────────────────────────────────────────
def _build_report_prompt(identity_statement, roles, standards, week_entries):
    role_names   = ", ".join(r["name"] for r in roles)
    standard_map = {s["id"]: s["title"] for s in standards}

    days_summary = []
    for entry in week_entries:
        date      = entry.get("date", "")
        completed = [standard_map.get(sid, sid) for sid in entry.get("standards_completed", [])]
        total     = [standard_map.get(sid, sid) for sid in entry.get("standards_total", [])]
        pct       = int(len(completed) / len(total) * 100) if total else 0

        answers_text = ""
        for a in entry.get("evening_answers", []):
            answers_text += f"\n  Q: {a.get('question','')}\n  A: {a.get('answer','')}"

        days_summary.append(
            f"Date: {date} | Identity score: {pct}% ({len(completed)}/{len(total)} standards)\n"
            f"Completed: {', '.join(completed) or 'none'}\n"
            f"Evening reflection:{answers_text}"
        )

    days_text = "\n\n".join(days_summary)

    return f"""
You are a coach analyzing a weekly identity journal for someone who declared:
IDENTITY: "{identity_statement}"
ROLES: {role_names}

Here is their week:
{days_text}

Based ONLY on the data above, generate a weekly identity report.
Respond ONLY with valid JSON, no markdown, no explanation. Use this exact structure:

{{
  "overall_score": <float 0.0-1.0, average of daily scores>,
  "headline": "<one powerful sentence summarizing the week — honest, not generic>",
  "strongest_role": "<role name where they showed up most consistently>",
  "weakest_role": "<role name that needs the most attention>",
  "pattern_identified": "<one specific behavioral pattern you detected from their journal answers — be specific, not generic>",
  "mirror_insight": "<synthesis of their Mirror Mode answers — what pattern emerges in how they see themselves>",
  "one_correction": "<the single most important thing to change next week — specific and actionable>",
  "identity_alignment": "<honest assessment: are they living their declared identity or not, and why>"
}}
"""


if __name__ == "__main__":
    app.run(debug=True, port=5000)