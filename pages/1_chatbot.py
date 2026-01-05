import json
import time
from typing import Optional, Dict, List, Any

import pandas as pd
import requests
import streamlit as st


# -----------------------
# State
# -----------------------
def init_state():
    if "chat" not in st.session_state:
        st.session_state.chat = []  # [{"role":"user|assistant","content":"..."}]
    if "llm_confirmed" not in st.session_state:
        st.session_state.llm_confirmed = False
    if "last_confirm_result" not in st.session_state:
        st.session_state.last_confirm_result = ""
    if "datasets" not in st.session_state:
        st.session_state.datasets = {}  # filename -> dataframe


# -----------------------
# Defaults (Structured Prompts)
# -----------------------
DEFAULT_TASK = "You are an expert ODI grip specialist helping users of all levels—beginner to expert—choose the most suitable grip from ODI's product range based strictly on the uploaded CSV dataset. Your job is to recommend the best grip for their specific riding style, comfort needs, hand size, and skill level."

DEFAULT_PERSONA = "The user may be a beginner or an experienced rider. Use simple, clear explanations for beginners (avoid technical jargon), and use more technical language if the user shows expertise or uses advanced terms. Always adapt based on how they describe their needs."

DEFAULT_TONE = "Respond in a professional, supportive, and informative tone—similar to a knowledgeable customer service expert in a high-end bike shop. Encourage beginners and build trust with experienced riders."

DEFAULT_DATA_RULES = """DATA RULES (STRICT):
- Use ONLY information retrieved from the embedded ODI product dataset.
- Do NOT browse the web, cite webpages, or use external reviews/knowledge.
- Do NOT invent features, prices, specs, availability, or “best overall” claims.
- If a detail is not in the retrieved context, say you’re not sure and ask a clarifying question instead.
- Only ODI grips are allowed. Never recommend competitor brands.
- Always recommend at least one specific product name from the retrieved context if a match is found.
"""

DEFAULT_SCOPE = """SCOPE:
This assistant supports ALL ODI grips in the dataset (e.g., MTB, BMX, Moto, Urban/Casual).
If the user asks about a category not supported, explain the limitation and ask follow-up.
Identify the riding category early (MTB vs BMX vs Moto vs Casual) because it strongly affects which grips fit.
"""

DEFAULT_PREF_SCHEMA = """
PREFERENCES (ONLY THESE AFFECT RECOMMENDATIONS):

Keys:
- riding_style
- locking_mechanism
- thickness
- damping_level
- durability

Allowed values:
riding_style: trail, enduro, downhill, cross-country, bmx, moto, urban, casual
locking_mechanism: lock-on, slip-on
thickness: thin, medium, thick, medium-thick size xl
damping_level: low, medium, high
durability: low, medium, high

Rules:
- Only set a preference if the user clearly indicates it.
- If unclear, leave it unset and ask ONE follow-up question.
"""

DEFAULT_MAPPING = """MAPPING HINTS (use only when intent is clear):

riding_style:
- “BMX / park / street tricks” -> bmx
- “Rocky trails / mixed terrain / all-mountain” -> trail
- “Enduro / aggressive trail / rough descents” -> enduro
- “Downhill / bike park / steep fast” -> downhill
- “XC / racing / long climbs” -> cross-country
- “Moto” -> moto
- “Commuting / city rides” -> urban
- “Casual cruising / e-bike comfort” -> casual

thickness:
- “small hands / slim / skinny” -> thin
- “chunky / fat / big / extra padding” -> thick
- “large hands / XL gloves” -> medium-thick size xl (only if user indicates XL/very large hands)

damping_level:
- “hands numb / vibration / shock absorption / rocky” -> high
- “balanced” -> medium
- “more trail feel / firm” -> low

locking_mechanism:
- “lock-on / clamps” -> lock-on
- “slip-on / push-on” -> slip-on

durability:
- “long-lasting / hard riding / abrasive trails” -> high
"""

DEFAULT_WORKFLOW = """WORKFLOW:
1) Welcome the user and ask what they ride + what problem they want to solve (comfort, control, numbness, hand size, etc.).
2) Identify riding_style early if possible.
3) Ask ONE focused follow-up question at a time to fill missing preferences.
4) Once enough preferences are collected, recommend grips based ONLY on the dataset.
5) Briefly explain why the suggested grips match the stated preferences, without adding unsupported details.
"""

DEFAULT_OUTPUT_RULES = """RESPONSE FORMAT:
- Keep replies short (2–6 sentences).
- Always include:
  (a) 1 quick acknowledgment of the user’s situation
  (b) 1 helpful suggestion or explanation based on their needs
  (c) ONE follow-up question if preferences are still missing

Avoid:
- long bullet lists
- multiple follow-up questions in one message
- technical jargon unless the user clearly uses it first
"""


# -----------------------
# Prompt + Dataset Context
# -----------------------
def build_dataset_context(
    datasets: Dict[str, pd.DataFrame],
    max_rows_each: int = 8
) -> str:
    """Compact dataset summary for grounding. Keeps prompt smaller than dumping full CSVs."""
    if not datasets:
        return "No dataset uploaded yet. You must ask the user to upload ODI product CSV files."

    lines = [f"{len(datasets)} ODI dataset file(s) loaded."]

    for fname, df in list(datasets.items())[:10]:
        lines.append(f"\n--- File: {fname}")
        lines.append(f"Columns: {list(df.columns)}")
        lines.append("Preview:")
        lines.append(df.head(max_rows_each).to_csv(index=False))

    if len(datasets) > 10:
        lines.append(f"\n(And {len(datasets) - 10} more files.)")

    return "\n".join(lines)


def build_system_prompt(
    task: str,
    persona: str,
    tone: str,
    data_rules: str,
    scope: str,
    pref_schema: str,
    mapping_guide: str,
    workflow: str,
    output_rules: str,
    dataset_context: str,
) -> str:
    return f"""
[Task Definition]
{task}

[Customer Persona]
{persona}

[Tone & Language Style]
{tone}

[Data Access & Grounding Rules]
{data_rules}

[Scope & Category Handling]
{scope}

[Preference Schema]
{pref_schema}

[Mapping Guide]
{mapping_guide}

[Conversation Workflow]
{workflow}

[Output Format Rules]
{output_rules}

[Uploaded Dataset Context]
{dataset_context}

IMPORTANT:
- If the dataset is missing or does not contain the requested category/specs, say so clearly and ask ONE follow-up question.
- Do NOT invent. Do NOT use outside knowledge.
""".strip()


# -----------------------
# OpenRouter LLM Call
# -----------------------
def call_llm_openrouter(
    api_key: str,
    model: str,
    system_prompt: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 600,
) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"

    payload: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system_prompt}] + messages,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "ODI Grips Chatbot (Streamlit)",
    }

    resp = requests.post(
        url,
        headers=headers,
        data=json.dumps(payload),
        timeout=60
    )

    if resp.status_code != 200:
        raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text}")

    data = resp.json()

    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(
            f"Unexpected OpenRouter response format: {json.dumps(data)[:1200]}"
        )


# -----------------------
# CSV Loading + Transcript
# -----------------------
def load_csv_uploaded_file(uploaded_file) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(uploaded_file)
    except Exception:
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="latin-1")
        except Exception:
            return None


def transcript_json() -> str:
    return json.dumps(
        {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "messages": st.session_state.chat,
        },
        indent=2,
    )


# =========================
# App
# =========================
init_state()

st.set_page_config(
    page_title="ODI Grips Chatbot",
    page_icon="🚵",
    layout="wide"
)

st.title("🚵 ODI Grips Chatbot")
st.caption("Structured prompts + dataset upload + transcript download")


# Sidebar: OpenRouter settings
with st.sidebar:
    st.header("LLM Settings (OpenRouter)")

    api_key = st.text_input(
        "OpenRouter API Key",
        value=st.secrets.get("OPENROUTER_API_KEY", ""),
        type="password",
    )

    model = st.text_input(
        "Model",
        value=st.secrets.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
    )

    if st.session_state.last_confirm_result:
        st.caption(st.session_state.last_confirm_result)

    st.success(
        "LLM confirmed ✅"
        if st.session_state.llm_confirmed
        else "LLM not confirmed ⚠️"
    )


# ---- Structured Prompts ----
st.subheader("Structured Prompts")

with st.expander("Structured Prompts", expanded=True):
    task = st.text_area("1) Task Definition", DEFAULT_TASK, height=90)
    persona = st.text_area("2) Customer Persona", DEFAULT_PERSONA, height=90)
    tone = st.text_area("3) Tone & Language Style", DEFAULT_TONE, height=90)

    st.markdown("#### Added grounding + control sections")

    data_rules = st.text_area(
        "4) Data Access & Grounding Rules",
        DEFAULT_DATA_RULES,
        height=180,
    )

    scope = st.text_area(
        "5) Scope & Category Handling",
        DEFAULT_SCOPE,
        height=150,
    )

    pref_schema = st.text_area(
        "6) Preference Schema",
        DEFAULT_PREF_SCHEMA,
        height=220,
    )

    mapping_guide = st.text_area(
        "7) Mapping Guide",
        DEFAULT_MAPPING,
        height=320,
    )

    workflow = st.text_area(
        "8) Conversation Workflow",
        DEFAULT_WORKFLOW,
        height=170,
    )

    output_rules = st.text_area(
        "9) Output Format Rules",
        DEFAULT_OUTPUT_RULES,
        height=190,
    )


# ---- Dataset Upload ----
st.subheader("Dataset Upload (ODI products)")
st.write("Upload **multiple CSV files** from your ODI products folder.")

csv_files = st.file_uploader(
    "Upload CSV files",
    type=["csv"],
    accept_multiple_files=True,
)

c1, c2 = st.columns([1, 1])

with c1:
    if st.button("Load Uploaded CSVs", use_container_width=True):
        if not csv_files:
            st.warning("No CSVs selected yet.")
        else:
            count = 0
            for f in csv_files:
                df = load_csv_uploaded_file(f)
                if df is not None:
                    st.session_state.datasets[f.name] = df
                    count += 1
            st.success(f"Loaded {count} CSV file(s).")

with c2:
    if st.button("Clear Loaded Data", use_container_width=True):
        st.session_state.datasets = {}
        st.toast("Datasets cleared.")


if st.session_state.datasets:
    st.markdown("### Loaded files")
    st.write(list(st.session_state.datasets.keys())[:40])

    preview_file = st.selectbox(
        "Preview a file",
        options=list(st.session_state.datasets.keys()),
    )

    st.dataframe(
        st.session_state.datasets[preview_file].head(25),
        use_container_width=True,
    )
else:
    st.info("No datasets loaded yet.")


# ---- Actions ----
st.subheader("Actions")
b1, b2, b3 = st.columns(3)

with b1:
    if st.button("✅ Confirm LLM Setup", use_container_width=True):
        if not api_key:
            st.session_state.llm_confirmed = False
            st.session_state.last_confirm_result = "Missing OpenRouter API key."
        else:
            try:
                ping_system = (
                    "You are a helpful assistant. "
                    "Reply exactly with 'LLM OK'."
                )
                ping_messages = [{"role": "user", "content": "LLM OK"}]

                out = call_llm_openrouter(
                    api_key=api_key,
                    model=model,
                    system_prompt=ping_system,
                    messages=ping_messages,
                    temperature=0.0,
                    max_tokens=10,
                )

                st.session_state.llm_confirmed = "LLM OK" in out
                st.session_state.last_confirm_result = f"Response: {out}"
                st.toast("LLM setup checked.")
            except Exception as e:
                st.session_state.llm_confirmed = False
                st.session_state.last_confirm_result = f"Error: {e}"

with b2:
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.chat = []
        st.toast("Chat cleared.")

with b3:
    st.download_button(
        "⬇️ Generate Transcript",
        data=transcript_json().encode("utf-8"),
        file_name="odi_chat_transcript.json",
        mime="application/json",
        use_container_width=True,
    )


st.divider()


# ---- Chat ----
st.subheader("Chat")

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_msg = st.chat_input("Ask: Which ODI grips would you recommend?")

if user_msg:
    st.session_state.chat.append({"role": "user", "content": user_msg})

    with st.chat_message("user"):
        st.markdown(user_msg)

    dataset_context = build_dataset_context(st.session_state.datasets)

    system_prompt = build_system_prompt(
        task=task,
        persona=persona,
        tone=tone,
        data_rules=data_rules,
        scope=scope,
        pref_schema=pref_schema,
        mapping_guide=mapping_guide,
        workflow=workflow,
        output_rules=output_rules,
        dataset_context=dataset_context,
    )

    with st.chat_message("assistant"):
        if not api_key:
            st.error("Add your OpenRouter API key in the sidebar first.")
        else:
            try:
                assistant_text = call_llm_openrouter(
                    api_key=api_key,
                    model=model,
                    system_prompt=system_prompt,
                    messages=st.session_state.chat,
                    temperature=0.2,
                    max_tokens=700,
                )

                st.markdown(assistant_text)
                st.session_state.chat.append(
                    {"role": "assistant", "content": assistant_text}
                )

            except Exception as e:
                st.error(f"LLM error: {e}")
