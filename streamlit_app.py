import json
import urllib.parse
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests
import streamlit as st


# =========================
# Config
# =========================
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

PREMIUM_MODEL = "openai/gpt-4o-mini"
BUDGET_MODEL = "mistralai/mistral-small-latest"

ALLOWED_PREFS = {
    "riding_style": ["trail", "enduro", "downhill", "cross-country"],
    "locking_mechanism": ["lock-on", "slip-on"],
    "thickness": ["thin", "medium", "thick", "medium-thick size xl"],
    "damping_level": ["low", "medium", "high"],
    "durability": ["low", "medium", "high"],
}
PREF_KEYS = list(ALLOWED_PREFS.keys())

DEMO_RESPONSES = [
    "Appreciate the details! Based on that, I’m thinking comfort + control—how rough are your usual trails?",
    "If your hands go numb, cushioning matters. Do you prefer softer rubber or more support?",
    "Got it. Quick check: lock-on hardware or more of a slip-on feel?",
    "Once I know thickness + durability priority, I can narrow to 2–3 top picks.",
]


# =========================
# Styling (dark UI like your screenshot)
# =========================
def inject_css():
    st.markdown(
        """
<style>
/* Page background + base text */
.stApp {
  background: radial-gradient(1200px 700px at 30% 10%, #1b2430 0%, #0b0f14 55%, #070a0d 100%);
  color: #EDEFF2;
}

/* Title spacing */
h1, h2, h3 { letter-spacing: -0.02em; }

/* Make main container narrower like a "lab" */
.block-container {
  max-width: 980px;
  padding-top: 40px;
}

/* Input areas */
textarea, input {
  border-radius: 14px !important;
}

/* Chat input bar */
.stChatInputContainer {
  border-radius: 18px !important;
}

/* Buttons */
div.stButton > button, .stDownloadButton > button {
  border-radius: 16px !important;
  padding: 0.65rem 1.1rem !important;
  font-weight: 600 !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  background: rgba(255,255,255,0.08) !important;
}
div.stButton > button:hover, .stDownloadButton > button:hover {
  background: rgba(255,255,255,0.12) !important;
}

/* "Primary" looking buttons via container class */
.primary-btn button {
  background: #ff5b5b !important;
  color: white !important;
  border: none !important;
}
.primary-btn button:hover {
  filter: brightness(0.95);
}

/* Expander */
div[data-testid="stExpander"] {
  border-radius: 18px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.04);
  overflow: hidden;
}
div[data-testid="stExpander"] summary {
  font-size: 1.05rem;
  font-weight: 650;
}

/* Code/json blocks */
pre {
  border-radius: 14px !important;
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )


# =========================
# Helpers
# =========================
def odi_search_link(product_name: str) -> str:
    q = urllib.parse.quote(product_name or "")
    return f"https://odigrips.com/search?q={q}"


def normalize_listish(cell):
    if pd.isna(cell):
        return []
    return [x.strip().lower() for x in str(cell).split(";") if x.strip()]


def score_row(row, prefs):
    score = 0
    if prefs.get("locking_mechanism"):
        if str(row.get("locking_mechanism", "")).strip().lower() == prefs["locking_mechanism"]:
            score += 4
        else:
            score -= 2

    if prefs.get("riding_style"):
        styles = normalize_listish(row.get("riding_style"))
        if prefs["riding_style"] in styles:
            score += 4

    if prefs.get("thickness"):
        if str(row.get("thickness", "")).strip().lower() == prefs["thickness"]:
            score += 2

    if prefs.get("damping_level"):
        if str(row.get("damping_level", "")).strip().lower() == prefs["damping_level"]:
            score += 2

    if prefs.get("durability"):
        if str(row.get("durability", "")).strip().lower() == prefs["durability"]:
            score += 1

    return score


def recommend(df, prefs, top_n=3):
    scored = df.copy()
    scored["__score"] = scored.apply(lambda r: score_row(r, prefs), axis=1)
    scored = scored.sort_values("__score", ascending=False)
    return scored.head(top_n).drop(columns="__score")


def load_default_dataframe():
    repo_root = Path(__file__).resolve().parents[0]
    candidates = [
        repo_root / "data" / "ODI_MTB_GRIPS.csv",
        repo_root / "ODI_MTB_GRIPS.csv",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p), p
    raise FileNotFoundError(
        "Could not find ODI_MTB_GRIPS.csv.\n"
        f"Looked for:\n- {candidates[0]}\n- {candidates[1]}\n\n"
        "Fix: add the CSV to your repo at one of those paths (recommended: data/ODI_MTB_GRIPS.csv)."
    )


def call_openrouter(messages, model, max_tokens, temperature=0.6):
    api_key = st.secrets.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY in Streamlit secrets.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=25)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# =========================
# Preference extraction
# =========================
def validate_pref_value(key: str, value: str) -> str:
    if not value:
        return ""
    normalized = value.strip().lower()
    if key == "riding_style" and normalized in {"xc", "x-c", "cross country", "crosscountry"}:
        normalized = "cross-country"
    if normalized in ALLOWED_PREFS[key]:
        return normalized
    return ""


def update_prefs(new_values: dict):
    for key in PREF_KEYS:
        val = new_values.get(key, "")
        valid = validate_pref_value(key, val)
        if valid:
            st.session_state.prefs[key] = valid


def heuristic_prefs(user_text: str) -> dict:
    text = user_text.lower()
    found = {k: "" for k in PREF_KEYS}
    synonyms = {
        "riding_style": {
            "rocky": "trail",
            "all-mountain": "trail",
            "xc": "cross-country",
            "cross country": "cross-country",
        },
        "thickness": {"chunky": "thick", "fat": "thick", "slim": "thin", "skinny": "thin"},
        "damping_level": {"plush": "high", "cushy": "high", "firm": "low"},
        "durability": {"long lasting": "high", "long-lasting": "high"},
    }

    for key, options in ALLOWED_PREFS.items():
        for option in options:
            if option in text:
                found[key] = option
        for hint, canonical in synonyms.get(key, {}).items():
            if hint in text:
                found[key] = canonical

    if "lock on" in text or "lock-on" in text:
        found["locking_mechanism"] = "lock-on"
    if "slip on" in text or "slip-on" in text:
        found["locking_mechanism"] = "slip-on"

    return found


def llm_extract_preferences(user_text: str, model: str, max_tokens: int) -> dict:
    instructions = (
        "You extract mountain bike grip preferences. "
        "Return strict JSON ONLY with the keys riding_style, locking_mechanism, thickness, damping_level, durability. "
        "Each value must be one of the allowed options or an empty string. "
        "Allowed values:\n"
        f"riding_style: {', '.join(ALLOWED_PREFS['riding_style'])}\n"
        f"locking_mechanism: {', '.join(ALLOWED_PREFS['locking_mechanism'])}\n"
        f"thickness: {', '.join(ALLOWED_PREFS['thickness'])}\n"
        f"damping_level: {', '.join(ALLOWED_PREFS['damping_level'])}\n"
        f"durability: {', '.join(ALLOWED_PREFS['durability'])}\n"
        "If unsure, use an empty string."
    )

    payload = {"existing_preferences": st.session_state.prefs, "user_message": user_text}
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": json.dumps(payload)},
    ]
    raw = call_openrouter(messages, model, max_tokens, temperature=0.2)
    return json.loads(raw)


# =========================
# Prompt Lab (UI-controlled system prompt)
# =========================
DEFAULT_TASK = "<Task> You are an ODI mountain bike grips expert who provides grip recommendations to users."
DEFAULT_PERSONA = "<Customer Persona> The user may be a beginner or an experienced mountain biker. Adjust explanation depth to their level."
DEFAULT_TONE = "<Tone> Respond in a professional and informative tone, similar to a helpful customer service representative."
DEFAULT_DATA_ACCESS = (
    "<Data Access>\n"
    "- You must rely only on the ODI grip dataset shown in the app.\n"
    "- Do NOT invent prices, features, or availability.\n"
    "- If info is missing, ask one focused follow-up question.\n"
    "- When you recommend, reference the exact product name."
)


def build_system_prompt_from_ui(task, persona, tone, data_access, low_cost=False):
    limit_note = "Keep replies under ~80 words.\n" if low_cost else ""
    return (
        f"{task}\n\n"
        f"{persona}\n\n"
        f"{tone}\n\n"
        f"{data_access}\n\n"
        f"{limit_note}"
        "Behavior rules:\n"
        "- Be conversational, not robotic.\n"
        "- Acknowledge the rider’s pain/comfort concerns.\n"
        "- Ask ONE clear follow-up if needed.\n"
        "- Never recommend non-ODI brands.\n"
    )


# =========================
# Session state
# =========================
def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hey! Tell me what you’re feeling on the bike (hand pain, numbness, trail type), and I’ll help narrow down ODI grips.",
            }
        ]
    if "prefs" not in st.session_state:
        st.session_state.prefs = {k: "" for k in PREF_KEYS}

    # Prompt lab fields
    if "ui_task" not in st.session_state:
        st.session_state.ui_task = DEFAULT_TASK
    if "ui_persona" not in st.session_state:
        st.session_state.ui_persona = DEFAULT_PERSONA
    if "ui_tone" not in st.session_state:
        st.session_state.ui_tone = DEFAULT_TONE
    if "ui_data_access" not in st.session_state:
        st.session_state.ui_data_access = DEFAULT_DATA_ACCESS

    # “locked in” system prompt snapshot
    if "confirmed_system_prompt" not in st.session_state:
        st.session_state.confirmed_system_prompt = build_system_prompt_from_ui(
            st.session_state.ui_task,
            st.session_state.ui_persona,
            st.session_state.ui_tone,
            st.session_state.ui_data_access,
            low_cost=False,
        )
    if "prompt_confirmed_at" not in st.session_state:
        st.session_state.prompt_confirmed_at = None

    # Modes
    if "low_cost_mode" not in st.session_state:
        st.session_state.low_cost_mode = False
    if "demo_mode_manual" not in st.session_state:
        st.session_state.demo_mode_manual = False
    if "demo_mode_forced" not in st.session_state:
        st.session_state.demo_mode_forced = False
    if "demo_notice_shown" not in st.session_state:
        st.session_state.demo_notice_shown = False
    if "demo_resp_idx" not in st.session_state:
        st.session_state.demo_resp_idx = 0


def reset_chat():
    st.session_state.messages = [
        {"role": "assistant", "content": "Fresh start. What kind of riding do you do, and what’s bothering your hands?"}
    ]
    st.session_state.prefs = {k: "" for k in PREF_KEYS}
    st.session_state.demo_mode_forced = False
    st.session_state.demo_notice_shown = False
    st.session_state.demo_resp_idx = 0


def enable_forced_demo():
    st.session_state.demo_mode_forced = True
    st.session_state.demo_notice_shown = False
    st.warning("OpenRouter had an issue, so the app switched to Demo mode (no API calls).")


def scripted_demo_response(user_text: str) -> str:
    idx = st.session_state.demo_resp_idx
    st.session_state.demo_resp_idx = (idx + 1) % len(DEMO_RESPONSES)
    snippet = user_text.strip()
    snippet = snippet if len(snippet) <= 80 else snippet[:77] + "..."
    base = DEMO_RESPONSES[idx]
    notice = ""
    if st.session_state.demo_mode_forced and not st.session_state.demo_notice_shown:
        notice = "Heads-up: live AI is paused, so you’re in Demo mode. "
        st.session_state.demo_notice_shown = True
    return f'{notice}{base} (Got it: "{snippet}").'


def filled_pref_count() -> int:
    return sum(1 for v in st.session_state.prefs.values() if v)


def build_pref_summary():
    parts = []
    for key, label in [
        ("riding_style", "style"),
        ("locking_mechanism", "locking"),
        ("thickness", "thickness"),
        ("damping_level", "damping"),
        ("durability", "durability"),
    ]:
        val = st.session_state.prefs.get(key, "")
        if val:
            parts.append(f"{label}: {val}")
    return ", ".join(parts)


# =========================
# Chat functions
# =========================
def llm_chat_response(model: str, max_tokens: int) -> str:
    system_prompt = st.session_state.confirmed_system_prompt
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(st.session_state.messages)
    return call_openrouter(messages, model, max_tokens, temperature=0.6)


def process_preferences(user_text: str, use_demo: bool, model: str, max_tokens: int):
    if use_demo:
        update_prefs(heuristic_prefs(user_text))
        return
    try:
        extracted = llm_extract_preferences(user_text, model, max_tokens)
        update_prefs(extracted)
    except Exception:
        enable_forced_demo()
        update_prefs(heuristic_prefs(user_text))


# =========================
# App
# =========================
st.set_page_config(page_title="AI Customer Support Lab", layout="wide")
inject_css()
init_state()

# Load CSV
try:
    df, csv_path = load_default_dataframe()
except Exception as e:
    st.error(str(e))
    st.stop()

# Header
st.title("AI Customer Support Lab")
st.caption("For a quick demo, check out this video.")

# “Select Class” (matches screenshot vibe)
colA, colB = st.columns([2, 1])
with colA:
    st.selectbox("Select Class", ["AI in Marketing class", "IBM 6010", "IBM 6500", "IBM 6540"], index=0)
with colB:
    st.caption(f"Dataset: `{csv_path}`")

# Prompt Lab editor
with st.expander("Structured Prompts", expanded=True):
    st.subheader("Task Definition")
    st.session_state.ui_task = st.text_area("", value=st.session_state.ui_task, height=90, key="task_area")

    st.subheader("Customer Persona")
    st.session_state.ui_persona = st.text_area("", value=st.session_state.ui_persona, height=90, key="persona_area")

    st.subheader("Tone & Language Style")
    st.session_state.ui_tone = st.text_area("", value=st.session_state.ui_tone, height=90, key="tone_area")

    st.subheader("Data Access")
    st.session_state.ui_data_access = st.text_area(
        "", value=st.session_state.ui_data_access, height=170, key="data_access_area"
    )

    # Optional: show a “data access sample” snippet like your screenshot
    st.markdown("**Data preview (1 example row)**")
    preview_row = df.iloc[0].to_dict() if len(df) else {}
    st.json(preview_row)

# Controls row (like your 3 buttons)
btn1, btn2, btn3 = st.columns([1, 1, 1])

with btn1:
    st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
    confirm = st.button("Confirm LLM Setup", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with btn2:
    clear = st.button("Clear Chat History", use_container_width=True)

with btn3:
    # Build transcript text for download
    transcript_lines = []
    for m in st.session_state.messages:
        role = m["role"].upper()
        transcript_lines.append(f"{role}: {m['content']}")
    transcript_text = "\n\n".join(transcript_lines)
    st.download_button(
        "Generate Transcript",
        data=transcript_text.encode("utf-8"),
        file_name=f"chat_transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=True,
    )

if confirm:
    st.session_state.confirmed_system_prompt = build_system_prompt_from_ui(
        st.session_state.ui_task,
        st.session_state.ui_persona,
        st.session_state.ui_tone,
        st.session_state.ui_data_access,
        low_cost=st.session_state.low_cost_mode,
    )
    st.session_state.prompt_confirmed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.success("LLM setup confirmed. The chat will now use your updated prompt.")

if clear:
    reset_chat()
    st.rerun()

# Sidebar (internal debug)
with st.sidebar:
    st.subheader("Internal Controls")
    st.toggle(
        "Low-cost mode",
        value=st.session_state.low_cost_mode,
        key="low_cost_mode",
        help="Uses budget model + shorter outputs.",
    )
    st.toggle(
        "Demo mode (no API calls)",
        value=st.session_state.demo_mode_manual,
        key="demo_mode_manual",
        help="Skips OpenRouter calls and uses scripted responses.",
    )
    if st.session_state.prompt_confirmed_at:
        st.caption(f"Prompt confirmed at: {st.session_state.prompt_confirmed_at}")

    st.markdown("### Current preferences")
    st.json(st.session_state.prefs)

    st.markdown("### Confirmed system prompt (snapshot)")
    st.code(st.session_state.confirmed_system_prompt, language="text")


# Model selection
use_demo_mode = st.session_state.demo_mode_manual or st.session_state.demo_mode_forced
model = BUDGET_MODEL if st.session_state.low_cost_mode else PREMIUM_MODEL
chat_max_tokens = 160 if st.session_state.low_cost_mode else 320
pref_max_tokens = 120 if st.session_state.low_cost_mode else 200

# Chat history render
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_text = st.chat_input("Which ODI grips would you recommend?")

if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})

    # preference extraction
    process_preferences(user_text, use_demo_mode, model, pref_max_tokens)

    # assistant response
    try:
        if use_demo_mode:
            assistant_reply = scripted_demo_response(user_text)
        else:
            assistant_reply = llm_chat_response(model, chat_max_tokens)
    except Exception:
        enable_forced_demo()
        assistant_reply = scripted_demo_response(user_text)

    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    st.rerun()

# Recommendations section
if filled_pref_count() >= 2:
    st.divider()
    st.subheader("Grip ideas based on what you've shared")
    summary = build_pref_summary()
    if summary:
        st.caption(f"Here's what I'm prioritizing: {summary}.")

    recs = recommend(df, st.session_state.prefs, top_n=3)
    for idx, rec in enumerate(recs.to_dict("records"), start=1):
        name = rec.get("name", "Unknown")
        with st.container():
            st.markdown(f"**{idx}. {name}**")
            st.write(
                f"- Style: {rec.get('riding_style', 'N/A')}\n"
                f"- Locking: {rec.get('locking_mechanism', 'N/A')}\n"
                f"- Thickness: {rec.get('thickness', 'N/A')}\n"
                f"- Damping: {rec.get('damping_level', 'N/A')}\n"
                f"- Durability: {rec.get('durability', 'N/A')}\n"
                f"- Pattern: {rec.get('grip_pattern', 'N/A')}\n"
                f"- Ergonomics: {rec.get('ergonomics', 'N/A')}\n"
                f"- Price: {rec.get('price', 'N/A')}\n"
                f"- Key features: {rec.get('key_features', 'N/A')}\n"
                f"- Colors: {rec.get('colors', 'N/A')}"
            )
            st.link_button("Search on ODI", odi_search_link(name), use_container_width=False)
