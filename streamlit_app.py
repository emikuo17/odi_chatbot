import os
import io
import time
import textwrap
from datetime import datetime

import pandas as pd
import streamlit as st

# Optional TF-IDF retrieval (recommended). If sklearn not installed, we fallback to keyword scoring.
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="AI Customer Support Lab",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------
# Dark UI CSS (roughly like your screenshot)
# ----------------------------
st.markdown(
    """
<style>
/* Main background */
.stApp {
    background: radial-gradient(1200px 800px at 20% 0%, #1b1f2a 0%, #0c0f14 45%, #07090d 100%);
    color: #eaeef7;
}

/* Title */
h1, h2, h3, h4, h5, h6, p, label, div {
    color: #eaeef7 !important;
}

/* Inputs */
textarea, input, .stTextInput input, .stTextArea textarea {
    background-color: rgba(255,255,255,0.06) !important;
    color: #eaeef7 !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
}

/* Selectbox */
div[data-baseweb="select"] > div {
    background-color: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
}

/* Expanders */
details {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 14px !important;
    padding: 8px 10px !important;
}

/* Buttons */
.stButton button {
    background: #ff5b5b !important;
    color: #ffffff !important;
    border: 0 !important;
    border-radius: 14px !important;
    padding: 10px 16px !important;
    font-weight: 700 !important;
}
.stButton button:hover {
    filter: brightness(0.92);
}

/* Chat bubbles */
.chat-bubble {
    border: 1px solid rgba(255,255,255,0.10);
    background: rgba(255,255,255,0.04);
    border-radius: 16px;
    padding: 12px 14px;
    margin: 8px 0;
}
.chat-user {
    background: rgba(255, 91, 91, 0.10);
}
.chat-role {
    font-size: 12px;
    opacity: 0.75;
    margin-bottom: 6px;
}

/* Footer-ish help text */
.small-note {
    font-size: 12px;
    opacity: 0.8;
}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------
# Helpers
# ----------------------------
def now_ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def df_to_row_text(df: pd.DataFrame) -> list[str]:
    """Convert each row to a single searchable string."""
    row_texts = []
    for _, r in df.iterrows():
        parts = []
        for c in df.columns:
            v = r.get(c, "")
            if pd.isna(v):
                v = ""
            parts.append(f"{c}: {str(v)}")
        row_texts.append(" | ".join(parts))
    return row_texts


def retrieve_relevant_rows(df: pd.DataFrame, query: str, k: int = 6) -> pd.DataFrame:
    """Return top-k relevant rows from df based on query."""
    if df is None or df.empty or not query.strip():
        return pd.DataFrame()

    texts = df_to_row_text(df)

    # TF-IDF (best)
    if SKLEARN_OK:
        try:
            vect = TfidfVectorizer(stop_words="english")
            X = vect.fit_transform(texts)
            q = vect.transform([query])
            sims = cosine_similarity(q, X).flatten()
            top_idx = sims.argsort()[::-1][:k]
            out = df.iloc[top_idx].copy()
            out.insert(0, "_score", [float(sims[i]) for i in top_idx])
            return out
        except Exception:
            pass

    # Fallback: simple keyword overlap scoring
    q_tokens = [t.lower() for t in query.split() if len(t) > 2]
    scores = []
    for t in texts:
        tl = t.lower()
        score = sum(1 for tok in q_tokens if tok in tl)
        scores.append(score)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    out = df.iloc[top_idx].copy()
    out.insert(0, "_score", [scores[i] for i in top_idx])
    return out


def build_system_prompt(task_def: str, persona: str, tone: str, data_rules: str, context_df: pd.DataFrame) -> str:
    """Create one consolidated system prompt."""
    context_block = ""
    if context_df is not None and not context_df.empty:
        # Keep context concise: show as bullet-like lines
        lines = []
        for _, row in context_df.iterrows():
            # drop score if present
            row_dict = row.to_dict()
            row_dict.pop("_score", None)
            # prefer "name" if it exists
            name = str(row_dict.get("name", "")).strip()
            if name:
                header = f"PRODUCT: {name}"
            else:
                header = "PRODUCT ROW"
            row_line = " | ".join([f"{k}={row_dict[k]}" for k in row_dict.keys()])
            lines.append(f"{header}\n{row_line}")
        context_block = "\n\n".join(lines)

    # Final prompt
    prompt = f"""
You are an AI Customer Support assistant.

[Task Definition]
{task_def.strip()}

[Customer Persona]
{persona.strip()}

[Tone & Language Style]
{tone.strip()}

[Data Access Rules]
{data_rules.strip()}

[Available Product Data Context]
{context_block if context_block else "No product rows retrieved for this question. If needed, ask 1 short follow-up question to clarify user needs."}

[Critical Rules]
- Only use the provided product data context when you recommend products.
- If product info is missing, ask one clear follow-up question (only one).
- Do not invent features, prices, sizes, or specs not shown in the data context.
- Keep answers practical and customer-support-like.
""".strip()

    return prompt


def call_llm(messages, model: str, temperature: float = 0.2) -> str:
    """
    Calls OpenAI using the new SDK if available.
    Requires OPENAI_API_KEY in environment.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return "⚠️ Missing OPENAI_API_KEY. Set it in your environment and rerun."

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ LLM call failed: {e}"


def format_transcript(chat_history: list[dict]) -> str:
    lines = []
    for m in chat_history:
        lines.append(f"[{m.get('time','')}] {m.get('role','').upper()}: {m.get('content','')}")
    return "\n\n".join(lines).strip()


# ----------------------------
# Session State
# ----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {role, content, time}
if "llm_confirmed" not in st.session_state:
    st.session_state.llm_confirmed = False
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "setup_snapshot" not in st.session_state:
    st.session_state.setup_snapshot = {}  # store confirmed prompt inputs


# ----------------------------
# Header
# ----------------------------
st.title("AI Customer Support Lab")
st.markdown('For a quick demo, check out this <a href="#" target="_blank">video</a>.', unsafe_allow_html=True)

# ----------------------------
# Top controls (class + model)
# ----------------------------
colA, colB, colC = st.columns([2.2, 1.3, 1.2])

with colA:
    selected_class = st.selectbox("Select Class", ["AI in Marketing class", "Other"], index=0)

with colB:
    model = st.selectbox(
        "Model",
        [
            "gpt-4o-mini",
            "gpt-4.1-mini",
            "gpt-4o",
        ],
        index=0,
        help="Pick the model your key has access to.",
    )

with colC:
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)


# ----------------------------
# Structured Prompts
# ----------------------------
with st.expander("Structured Prompts", expanded=True):
    st.subheader("Task Definition")
    task_def = st.text_area(
        label="",
        value=st.session_state.setup_snapshot.get("task_def", "<Task> You are an ODI mountain bike grips expert who provides grip recommendations to users."),
        height=90,
    )

    st.subheader("Customer Persona")
    persona = st.text_area(
        label=" ",
        value=st.session_state.setup_snapshot.get(
            "persona",
            "<Customer Persona> The user is an experienced mountain biker. Use technical terms and slang.",
        ),
        height=90,
    )

    st.subheader("Tone & Language Style")
    tone = st.text_area(
        label="  ",
        value=st.session_state.setup_snapshot.get(
            "tone",
            "<Tone> Respond in a professional and informative tone, similar to a customer service representative.",
        ),
        height=90,
    )

    st.subheader("Data Access")
    data_rules = st.text_area(
        label="   ",
        value=st.session_state.setup_snapshot.get(
            "data_rules",
            "<Data> Only recommend grip choices based on the uploaded product CSV. Do not recommend competitors.",
        ),
        height=80,
    )

    uploaded = st.file_uploader(
        "Upload product CSV (e.g., ODI_MTB_GRIPS.csv)",
        type=["csv"],
        accept_multiple_files=False,
    )

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.session_state.uploaded_df = df
            st.success(f"Loaded CSV with {df.shape[0]} rows and {df.shape[1]} columns.")
            with st.expander("Preview uploaded data (first 15 rows)", expanded=False):
                st.dataframe(df.head(15), use_container_width=True)
        except Exception as e:
            st.session_state.uploaded_df = None
            st.error(f"Failed to read CSV: {e}")

    # If user hasn't uploaded, but you want to auto-load local sample in dev:
    # (Comment out if you don't want this behavior)
    if st.session_state.uploaded_df is None and os.path.exists("ODI_MTB_GRIPS.csv"):
        try:
            st.session_state.uploaded_df = pd.read_csv("ODI_MTB_GRIPS.csv")
            st.info("Auto-loaded local ODI_MTB_GRIPS.csv (found in app directory). Upload to replace.")
        except Exception:
            pass


# ----------------------------
# Main Buttons row
# ----------------------------
b1, b2, b3 = st.columns([1, 1, 1])

with b1:
    if st.button("Confirm LLM Setup"):
        st.session_state.llm_confirmed = True
        st.session_state.setup_snapshot = {
            "task_def": task_def,
            "persona": persona,
            "tone": tone,
            "data_rules": data_rules,
            "model": model,
            "temperature": temperature,
            "class": selected_class,
            "confirmed_at": now_ts(),
        }
        st.success("LLM setup confirmed. Chatbot will use these settings.")

with b2:
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.llm_confirmed = False
        st.info("Chat cleared.")

with b3:
    transcript_text = format_transcript(st.session_state.chat_history)
    st.download_button(
        "Generate Transcript",
        data=transcript_text if transcript_text else "No chat history yet.",
        file_name=f"chat_transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
    )

st.markdown('<p class="small-note">Tip: Confirm LLM Setup before chatting (recommended).</p>', unsafe_allow_html=True)


# ----------------------------
# Chat Display
# ----------------------------
st.markdown("### Chat")

for m in st.session_state.chat_history:
    role = m["role"]
    content = m["content"]
    t = m.get("time", "")
    bubble_class = "chat-bubble chat-user" if role == "user" else "chat-bubble"
    st.markdown(
        f"""
<div class="{bubble_class}">
  <div class="chat-role">{role.upper()} • {t}</div>
  <div>{content.replace("\n","<br>")}</div>
</div>
""",
        unsafe_allow_html=True,
    )

# ----------------------------
# Chat Input
# ----------------------------
user_q = st.chat_input("Which ODI grips would you recommend?")

if user_q:
    # Save user message
    st.session_state.chat_history.append({"role": "user", "content": user_q, "time": now_ts()})

    # Use confirmed snapshot if confirmed, else current inputs
    if st.session_state.llm_confirmed and st.session_state.setup_snapshot:
        use_task = st.session_state.setup_snapshot["task_def"]
        use_persona = st.session_state.setup_snapshot["persona"]
        use_tone = st.session_state.setup_snapshot["tone"]
        use_data_rules = st.session_state.setup_snapshot["data_rules"]
        use_model = st.session_state.setup_snapshot.get("model", model)
        use_temp = st.session_state.setup_snapshot.get("temperature", temperature)
    else:
        use_task = task_def
        use_persona = persona
        use_tone = tone
        use_data_rules = data_rules
        use_model = model
        use_temp = temperature

    # Retrieve relevant product rows from uploaded CSV
    df = st.session_state.uploaded_df
    context_df = retrieve_relevant_rows(df, user_q, k=6) if df is not None else pd.DataFrame()

    # Build system prompt
    system_prompt = build_system_prompt(use_task, use_persona, use_tone, use_data_rules, context_df)

    # Build messages for LLM
    messages = [{"role": "system", "content": system_prompt}]

    # Include short history window (avoid huge prompt)
    history_window = st.session_state.chat_history[-10:]
    for m in history_window:
        messages.append({"role": m["role"], "content": m["content"]})

    # Call LLM
    assistant_text = call_llm(messages, model=use_model, temperature=use_temp)

    # Save assistant message
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_text, "time": now_ts()})

    # Rerun to display immediately
    st.rerun()
