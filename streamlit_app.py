import json
import io
import time
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

# --- OPTIONAL: OpenAI SDK (recommended). If you prefer another provider, swap the `call_llm()` function. ---
# pip install openai
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# =========================
# Helpers
# =========================
def safe_json_loads(s: str) -> Tuple[Optional[Any], Optional[str]]:
    try:
        return json.loads(s), None
    except Exception as e:
        return None, str(e)


def df_preview(df: pd.DataFrame, max_rows: int = 10) -> str:
    if df is None or df.empty:
        return ""
    preview = df.head(max_rows).to_markdown(index=False)
    return preview


def dataset_to_compact_context(df: Optional[pd.DataFrame], max_rows: int = 12) -> str:
    """Turn uploaded dataset into a compact, LLM-friendly context string."""
    if df is None or df.empty:
        return ""
    cols = list(df.columns)
    ctx = []
    ctx.append("Uploaded dataset is available. Use it to answer questions when relevant.")
    ctx.append(f"Columns: {cols}")
    ctx.append("Top rows (preview):")
    ctx.append(df.head(max_rows).to_csv(index=False))
    return "\n".join(ctx)


def build_system_prompt(task: str, persona: str, tone: str, data_access: str) -> str:
    # Keep it explicit & structured for the model
    return f"""
You are ODI mountain bike grips assistant.

[Task Definition]
{task}

[Customer Persona]
{persona}

[Tone & Language Style]
{tone}

[Data Access / Product Facts]
{data_access}

[Rules]
- Recommend ODI grips and explain why in MTB terms (feel, damping, tack, flange, diameter, trail type).
- Ask 1-2 quick clarifying questions ONLY if needed (hand size, glove size, terrain, preference for flange, diameter).
- If dataset is provided, prefer it as the source of truth for product specs.
- Be concise but helpful: bullet points + a short recommendation summary.
""".strip()


def call_llm(
    api_key: str,
    model: str,
    system_prompt: str,
    messages: list,
    temperature: float = 0.4,
) -> str:
    """
    Uses OpenAI Python SDK if installed. You can swap this out for any provider.

    messages: list of dicts like [{"role":"user","content":"..."}, ...]
    """
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Add it to requirements.txt or replace call_llm().")

    client = OpenAI(api_key=api_key)

    # Chat Completions is still widely supported across many installs.
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "system", "content": system_prompt}] + messages,
    )
    return resp.choices[0].message.content


def init_state():
    if "chat" not in st.session_state:
        st.session_state.chat = []  # list of {"role": "...", "content":"..."}
    if "llm_confirmed" not in st.session_state:
        st.session_state.llm_confirmed = False
    if "uploaded_df" not in st.session_state:
        st.session_state.uploaded_df = None
    if "last_confirm_result" not in st.session_state:
        st.session_state.last_confirm_result = ""


# =========================
# Defaults
# =========================
DEFAULT_TASK = "You are an ODI mountain bike grips expert who provides grip recommendations to users."
DEFAULT_PERSONA = "The user is an experienced mountain biker. Use technical terms and slang."
DEFAULT_TONE = "Respond in a professional and informative tone, similar to a customer service representative."

DEFAULT_DATA_DICT: Dict[str, Any] = {
    "models": {
        "ODI Rogue": {
            "diameter_mm": 33,
            "feel": "chunky, cushy, high vibration damping",
            "best_for": ["big hands", "DH", "park", "people who want max cushion"],
            "notes": ["good shock absorption", "can feel bulky for small hands"],
        },
        "ODI Elite Pro": {
            "diameter_mm": 31,
            "feel": "tacky, balanced damping, slim-ish",
            "best_for": ["trail", "enduro", "all-mountain", "control without bulk"],
            "notes": ["popular all-rounder", "good wet grip"],
        },
        "ODI Ruffian": {
            "diameter_mm": 30,
            "feel": "thin, precise, firm",
            "best_for": ["XC", "slopestyle", "riders who like direct bar feel"],
            "notes": ["less damping", "great feedback"],
        },
    },
    "common_features": [
        "lock-on grip system",
        "different diameters change fatigue + control",
        "flange vs no-flange impacts hand position + comfort",
    ],
    "colors": ["Black", "Red", "Graphite", "Light Blue", "Gum Rubber", "Iridescent Purple"],
    "damping_level": {
        "thin": "more feedback, less cushion",
        "medium": "balanced",
        "thick": "more cushion, less trail buzz",
    },
}

DEFAULT_DATA_ACCESS_TEXT = json.dumps(DEFAULT_DATA_DICT, indent=2)


# =========================
# UI
# =========================
init_state()

st.set_page_config(page_title="ODI Grips Chatbot", page_icon="🚵", layout="wide")
st.title("🚵 ODI Grips Chatbot")
st.caption("Structured prompts + dataset upload + transcript download")

# ---- Sidebar: LLM Config ----
with st.sidebar:
    st.header("LLM Settings")

    api_key = st.text_input("API Key", value=st.secrets.get("OPENAI_API_KEY", ""), type="password")
    model = st.text_input("Model", value=st.secrets.get("OPENAI_MODEL", "gpt-4o-mini"))
    temperature = st.slider("Temperature", 0.0, 1.0, 0.4, 0.05)

    st.divider()

    # Buttons row (like your 3rd screenshot)
    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("✅ Confirm LLM Setup", use_container_width=True):
            if not api_key:
                st.session_state.llm_confirmed = False
                st.session_state.last_confirm_result = "Missing API key."
            else:
                try:
                    # Minimal ping message
                    ping_system = "You are a helpful assistant."
                    ping_messages = [{"role": "user", "content": "Reply with: LLM OK"}]
                    out = call_llm(api_key, model, ping_system, ping_messages, temperature=0.0)
                    st.session_state.llm_confirmed = "LLM OK" in out
                    st.session_state.last_confirm_result = f"Response: {out}"
                except Exception as e:
                    st.session_state.llm_confirmed = False
                    st.session_state.last_confirm_result = f"Error: {e}"

    with c2:
        if st.button("🧹 Clear Chat History", use_container_width=True):
            st.session_state.chat = []
            st.toast("Chat cleared.")

    with c3:
        # Generate transcript bytes for download
        transcript_obj = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": model,
            "temperature": temperature,
            "messages": st.session_state.chat,
        }
        transcript_text = json.dumps(transcript_obj, indent=2)

        st.download_button(
            "⬇️ Generate Transcript",
            data=transcript_text.encode("utf-8"),
            file_name="odi_chat_transcript.json",
            mime="application/json",
            use_container_width=True,
        )

    if st.session_state.last_confirm_result:
        if st.session_state.llm_confirmed:
            st.success("LLM confirmed.")
        else:
            st.warning("LLM not confirmed.")
        st.caption(st.session_state.last_confirm_result)


# ---- Main layout ----
left, right = st.columns([1.15, 0.85], gap="large")

with left:
    # Structured Prompts block
    with st.expander("Structured Prompts", expanded=True):
        # Optional: show your screenshot if you add it
        # Put your screenshot at: assets/structured_prompts.png
        try:
            st.image("assets/structured_prompts.png", caption="Structured Prompts UI reference", use_container_width=True)
        except Exception:
            pass

        task = st.text_area("Task Definition", value=DEFAULT_TASK, height=80)
        persona = st.text_area("Customer Persona", value=DEFAULT_PERSONA, height=80)
        tone = st.text_area("Tone & Language Style", value=DEFAULT_TONE, height=80)

        st.markdown("### Data Access (Editable Dictionary / JSON)")
        data_access_text = st.text_area(
            "Paste / edit JSON here",
            value=DEFAULT_DATA_ACCESS_TEXT,
            height=260,
        )

        parsed_data, json_err = safe_json_loads(data_access_text)
        if json_err:
            st.error(f"JSON error: {json_err}")
        else:
            st.success("JSON looks valid.")

    # Dataset upload
    with st.expander("Dataset Upload", expanded=True):
        uploaded = st.file_uploader("Upload a dataset (CSV / XLSX / JSON)", type=["csv", "xlsx", "json"])

        if uploaded is not None:
            try:
                if uploaded.name.lower().endswith(".csv"):
                    df = pd.read_csv(uploaded)
                elif uploaded.name.lower().endswith(".xlsx"):
                    df = pd.read_excel(uploaded)
                elif uploaded.name.lower().endswith(".json"):
                    raw = json.load(uploaded)
                    df = pd.json_normalize(raw) if isinstance(raw, (list, dict)) else pd.DataFrame(raw)
                else:
                    df = None

                st.session_state.uploaded_df = df
                if df is not None:
                    st.write("Preview:")
                    st.dataframe(df.head(25), use_container_width=True)
            except Exception as e:
                st.session_state.uploaded_df = None
                st.error(f"Could not read file: {e}")

        if st.session_state.uploaded_df is not None:
            st.caption("Dataset context will be included for the chatbot when relevant.")

with right:
    st.subheader("Chat")

    # Render chat history
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_msg = st.chat_input("Ask: Which ODI grips would you recommend?")

    if user_msg:
        # Add user message
        st.session_state.chat.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        # Build system prompt
        data_ctx = ""
        if st.session_state.uploaded_df is not None:
            data_ctx = dataset_to_compact_context(st.session_state.uploaded_df)

        # If JSON invalid, fall back to raw text; otherwise embed the parsed dict nicely.
        if json_err:
            data_access_for_prompt = data_access_text
        else:
            data_access_for_prompt = json.dumps(parsed_data, indent=2)

        if data_ctx:
            data_access_for_prompt = data_access_for_prompt + "\n\n[Uploaded Dataset Context]\n" + data_ctx

        system_prompt = build_system_prompt(task, persona, tone, data_access_for_prompt)

        # LLM call
        with st.chat_message("assistant"):
            if not api_key:
                st.error("Add your API key in the sidebar first.")
            else:
                try:
                    assistant_text = call_llm(
                        api_key=api_key,
                        model=model,
                        system_prompt=system_prompt,
                        messages=st.session_state.chat,  # includes the user's newest message already
                        temperature=temperature,
                    )
                    st.markdown(assistant_text)
                    st.session_state.chat.append({"role": "assistant", "content": assistant_text})
                except Exception as e:
                    st.error(f"LLM error: {e}")
