# STREAMLIT APP with RAG EXTENSION + PA6 Batch Runner (Selectable LLMs)
# ✅ Keeps your original RAG pipeline + structured prompt controls + chat UI
# ✅ Adds (minimal changes):
#    1) Batch results include FIRST column: conversation_id = 1,2,3,...
#    2) After batch run: download BOTH CSV + JSON
#    3) Batch runner: choose which LLM(s) to run (1, 2, or all 3)
# ✅ UPDATE: LLM IDs changed to newer models on OpenRouter (GPT-4.1-mini, Claude Sonnet 4.5, Gemini 2.5 Flash)
# ✅ UPDATE: Batch output changed to LONG format:
#    {conversation_id, question, model, answer, rag_context}

import io
import json
import time
from typing import Dict, List

import pandas as pd
import requests
import streamlit as st

# RAG Libraries
import chromadb
from sentence_transformers import SentenceTransformer


# -----------------------
# Embedding Model (SAFE for Streamlit Cloud)
# -----------------------
@st.cache_resource
def get_embed_model():
    # Force CPU to avoid CUDA/MPS issues on Streamlit Cloud
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


# -----------------------
# State
# -----------------------
def init_state():
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "llm_confirmed" not in st.session_state:
        st.session_state.llm_confirmed = False
    if "last_confirm_result" not in st.session_state:
        st.session_state.last_confirm_result = ""
    if "datasets" not in st.session_state:
        st.session_state.datasets = {}
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None
    if "vector_db_ready" not in st.session_state:
        st.session_state.vector_db_ready = False

    # ✅ PA6 batch results
    if "batch_df" not in st.session_state:
        st.session_state.batch_df = None
    if "batch_last_run" not in st.session_state:
        st.session_state.batch_last_run = ""


# -----------------------
# RAG Functions
# -----------------------
def init_vector_db():
    """
    Use in-memory Chroma client to avoid Streamlit Cloud persistent DB corruption and
    embedding-dimension mismatch issues.
    Each time you upload/load, we rebuild cleanly.
    """
    client = chromadb.Client()  # ✅ in-memory
    st.session_state.vector_db = client.get_or_create_collection(name="odi-grips")
    st.session_state.vector_db_ready = True


def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_embed_model()
    return model.encode(texts).tolist()


def _row_to_product_card(row: pd.Series) -> str:
    """
    Product card formatting for embeddings + retrieval.
    IMPORTANT: includes colors so color queries (e.g., purple) can be retrieved.
    """
    name_key_candidates = ["product_name", "name", "title", "Product", "Product Name"]
    prod_name = None
    for k in name_key_candidates:
        if k in row.index and pd.notna(row[k]):
            prod_name = str(row[k]).strip()
            break

    preferred_keys = [
        "category", "Category",
        "riding_style", "Riding Style",
        "locking_mechanism", "Locking Mechanism",
        "Grip Attachment System", "grip attachment system",
        "thickness", "Thickness",
        "damping_level", "Damping Level",
        "durability", "Durability",
        "grip_pattern", "Grip Pattern",
        "pattern", "Pattern",
        "Feel", "feel",
        "traction", "Traction",
        "price", "Price",
        "ergonomics", "Ergonomics",
        "key_features", "Key Features",
        "differentiator", "Differentiator",
        "co_branding", "Co Branding",
        "Length", "length",
        # ✅ colors included
        "colors", "Colors",
        "color", "Color",
        "description", "Description",
        "notes", "Notes",
        "url", "URL", "link", "Link",
    ]

    lines = []
    if prod_name:
        lines.append(f"Product: {prod_name}")

    for k in preferred_keys:
        if k in row.index and pd.notna(row[k]):
            val = str(row[k]).strip()
            if val:
                label = str(k).replace("_", " ").title()
                lines.append(f"{label}: {val}")

    if len(lines) <= 1:
        extras = []
        for k, v in row.items():
            if pd.isna(v):
                continue
            if str(k) in preferred_keys:
                continue
            vv = str(v).strip()
            if vv:
                extras.append(f"{str(k)}: {vv}")
            if len(extras) >= 8:
                break
        if extras:
            lines.append("Other:")
            lines.extend(extras)

    return "\n".join(lines).strip()


def add_to_vector_db():
    if st.session_state.vector_db is None:
        init_vector_db()

    existing_ids = set()
    try:
        peek = st.session_state.vector_db.peek()
        for _id in peek.get("ids", []):
            existing_ids.add(_id)
    except Exception:
        pass

    all_texts, ids, metadata = [], [], []

    for fname, df in st.session_state.datasets.items():
        df = df.reset_index(drop=True)
        for i, row in df.iterrows():
            chunk_id = f"{fname}::row-{i}"
            if chunk_id in existing_ids:
                continue

            card = _row_to_product_card(row)
            if not card:
                continue

            all_texts.append(card)
            ids.append(chunk_id)
            metadata.append({"source": fname, "row_index": int(i)})

    if not all_texts:
        st.info("No rows to embed (empty dataset).")
        return

    embeddings = embed_texts(all_texts)

    if not (len(all_texts) == len(ids) == len(metadata) == len(embeddings)):
        st.error("Embedding batch length mismatch. Please check your CSV rows.")
        return

    st.session_state.vector_db.add(
        documents=all_texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadata,
    )
    st.success(f"Embedded {len(all_texts)} product rows.")


def rag_retrieve_context(query: str, top_k: int = 5) -> str:
    if st.session_state.vector_db is None or not st.session_state.vector_db_ready:
        return "No embedded product data available. Please load and embed CSVs first."

    embedded_query = embed_texts([query])[0]
    results = st.session_state.vector_db.query(
        query_embeddings=[embedded_query],
        n_results=top_k,
    )

    docs = results.get("documents", [[]])
    if not docs or not docs[0]:
        return "No matching context."

    return "\n\n---\n\n".join(docs[0])


# -----------------------
# Defaults (Structured Prompts)
# -----------------------
DEFAULT_TASK = (
    "You are an expert ODI grip specialist helping users of all levels—beginner to expert—"
    "choose the most suitable grip from ODI's product range based strictly on the uploaded CSV dataset. "
    "Your job is to recommend the best grip for their specific riding style, comfort needs, hand size, and skill level."
)

DEFAULT_PERSONA = (
    "The user may be a beginner or an experienced rider. Use simple, clear explanations for beginners "
    "(avoid technical jargon), and use more technical language if the user shows expertise or uses advanced terms. "
    "Always adapt based on how they describe their needs."
)

DEFAULT_TONE = (
    "Respond in a professional, supportive, and informative tone—similar to a knowledgeable customer service expert "
    "in a high-end bike shop. Encourage beginners and build trust with experienced riders."
)

DEFAULT_DATA_RULES = """DATA RULES (STRICT):
- Use ONLY information retrieved from the embedded ODI product dataset.
- Do NOT browse the web, cite webpages, or use external reviews/knowledge.
- Do NOT invent features, prices, specs, availability, or “best overall” claims.
- If a detail is not in the retrieved context, say you’re not sure and ask a clarifying question instead.
- Only ODI grips are allowed. Never recommend competitor brands.
- Always recommend at least one specific product name from the retrieved context if a match is found.
- If the user mentions a color (e.g., purple), prioritize products whose Colors include that color in the retrieved context.
"""

DEFAULT_SCOPE = """SCOPE:
This assistant supports ALL ODI grips in the dataset (e.g., MTB, BMX, Moto, Urban/Casual).
If the user asks about a category not supported, explain the limitation and ask follow-up.
Identify the riding category early (MTB vs BMX vs Moto vs Casual) because it strongly affects which grips fit.
"""

DEFAULT_PREF_SCHEMA = """PREFERENCES (ONLY THESE AFFECT RECOMMENDATIONS):

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
- Replies = 2–6 sentences
- Include:
  (a) acknowledgment
  (b) recommendation (include at least one named product when available)
  (c) one follow-up if needed
- Avoid long lists or multiple follow-ups
"""


def build_system_prompt(task, persona, tone, data_rules, scope, pref_schema, mapping_guide, workflow, output_rules, rag_context):
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

[RAG Context Retrieved for this Query]
{rag_context}

IMPORTANT:
- Use the RAG context above as your source of truth.
- If the RAG context is empty or does not contain a named product that matches, say so and ask ONE clarifying question.
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
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system_prompt}] + messages,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "ODI Grips Chatbot with RAG",
    }
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)

    if resp.status_code != 200:
        raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text}")

    data = resp.json()
    return data["choices"][0]["message"]["content"]


# -----------------------
# PA6 Batch Runner Helpers (Selectable LLMs)
# -----------------------
# ✅ Updated to newer OpenRouter models
BATCH_MODELS = {
    "GPT (OpenAI) — GPT-4.1 Mini": "openai/gpt-4.1-mini",
    "Claude (Anthropic) — Claude Sonnet 4.5": "anthropic/claude-sonnet-4.5",
    "Gemini (Google) — Gemini 2.5 Flash": "google/gemini-2.5-flash",
}


def safe_call_one(
    api_key: str,
    model_id: str,
    sys_prompt: str,
    question: str,
    temperature: float = 0.2,
    max_tokens: int = 600,
) -> str:
    try:
        msgs = [{"role": "user", "content": question}]
        return call_llm_openrouter(api_key, model_id, sys_prompt, msgs, temperature=temperature, max_tokens=max_tokens)
    except Exception as e:
        return f"ERROR: {e}"


def run_batch_eval(
    api_key: str,
    task: str,
    persona: str,
    tone: str,
    data_rules: str,
    scope: str,
    pref_schema: str,
    mapping_guide: str,
    workflow: str,
    output_rules: str,
    questions: List[str],
    selected_models: Dict[str, str],
    top_k: int = 5,
    temperature: float = 0.2,
    max_tokens: int = 600,
) -> pd.DataFrame:
    rows = []
    model_items = list(selected_models.items())
    total_calls = max(1, len(questions) * len(model_items))
    done = 0
    prog = st.progress(0)

    for idx, q in enumerate(questions, start=1):
        rag_context = rag_retrieve_context(q, top_k=top_k)
        sys_prompt = build_system_prompt(
            task, persona, tone, data_rules, scope, pref_schema, mapping_guide, workflow, output_rules, rag_context
        )

        # ✅ LONG format: one row per (question, model)
        for label, model_id in model_items:
            ans = safe_call_one(
                api_key=api_key,
                model_id=model_id,
                sys_prompt=sys_prompt,
                question=q,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            rows.append({
                "conversation_id": idx,
                "question": q,
                "llm": label,
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": ans},
                ],
            })

            done += 1
            prog.progress(min(1.0, done / total_calls))

    df = pd.DataFrame(rows)

    # Ensure conversation_id is first column
    if not df.empty and "conversation_id" in df.columns:
        df = df[["conversation_id"] + [c for c in df.columns if c != "conversation_id"]]

    return df


# -----------------------
# Streamlit UI Start
# -----------------------
init_state()

st.set_page_config(page_title="ODI Grips Chatbot (RAG)", page_icon="🚵", layout="wide")
st.title("🚵 ODI Grips Chatbot (with RAG)")

MODEL_PRESETS = {
    "GPT (OpenAI) — GPT-4.1 Mini": "openai/gpt-4.1-mini",
    "GPT (OpenAI) — GPT-4.1 (larger)": "openai/gpt-4.1",
    "Claude (Anthropic) — Claude Sonnet 4.5": "anthropic/claude-sonnet-4.5",
    "Gemini (Google) — Gemini 2.5 Flash": "google/gemini-2.5-flash",
    "Custom (type your own model ID)": "__custom__",
}

with st.sidebar:
    st.header("LLM Settings")
    api_key = st.text_input("OpenRouter API Key", type="password")

    model_choice = st.selectbox("Choose LLM (Chat mode)", list(MODEL_PRESETS.keys()), index=0)
    if MODEL_PRESETS[model_choice] == "__custom__":
        model = st.text_input("Model (OpenRouter ID)", value="openai/gpt-4.1-mini")
    else:
        model = MODEL_PRESETS[model_choice]
        st.caption(f"Using model: `{model}`")

    show_debug = st.checkbox("Show RAG Debug (retrieved context)", value=False)

st.subheader("📁 Upload CSV Files")
csv_files = st.file_uploader("Upload ODI product CSVs", type=["csv"], accept_multiple_files=True)

if st.button("🔄 Load & Embed CSVs"):
    st.session_state.vector_db = None
    st.session_state.vector_db_ready = False
    init_vector_db()

    st.session_state.datasets = {}
    for f in csv_files:
        df = pd.read_csv(f)
        df.columns = [c.strip() for c in df.columns]
        st.session_state.datasets[f.name] = df

    add_to_vector_db()
    st.success("CSV files loaded. Vector index rebuilt for RAG.")

    if show_debug and st.session_state.datasets:
        st.sidebar.markdown("### ✅ Loaded CSV Columns")
        for fname, df in st.session_state.datasets.items():
            st.sidebar.write(fname)
            st.sidebar.write(list(df.columns))

with st.expander("🧠 Structured Prompt Controls", expanded=True):
    st.subheader("Prompt Settings")
    task = st.text_area("Task", value=DEFAULT_TASK)
    persona = st.text_area("Persona", value=DEFAULT_PERSONA)
    tone = st.text_area("Tone", value=DEFAULT_TONE)
    data_rules = st.text_area("Data Rules", value=DEFAULT_DATA_RULES)
    scope = st.text_area("Scope", value=DEFAULT_SCOPE)
    pref_schema = st.text_area("Preferences", value=DEFAULT_PREF_SCHEMA)
    mapping_guide = st.text_area("Mapping", value=DEFAULT_MAPPING)
    workflow = st.text_area("Workflow", value=DEFAULT_WORKFLOW)
    output_rules = st.text_area("Format Rules", value=DEFAULT_OUTPUT_RULES)

st.subheader("Actions")
a1, a2, a3 = st.columns(3)

with a1:
    if st.button("✅ Confirm LLM Setup", use_container_width=True):
        if not api_key:
            st.session_state.llm_confirmed = False
            st.session_state.last_confirm_result = "Missing OpenRouter API key."
        else:
            try:
                ping_system = "You are a helpful assistant. Reply exactly with 'LLM OK'."
                ping_messages = [{"role": "user", "content": "LLM OK"}]
                out = call_llm_openrouter(api_key, model, ping_system, ping_messages, temperature=0.0, max_tokens=50)
                st.session_state.llm_confirmed = "LLM OK" in out
                st.session_state.last_confirm_result = f"Response: {out}"
                st.toast("LLM setup checked.")
            except Exception as e:
                st.session_state.llm_confirmed = False
                st.session_state.last_confirm_result = f"Error: {e}"

with a2:
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.chat = []
        st.toast("Chat history cleared.")

with a3:
    def transcript_json():
        return json.dumps(
            {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "messages": st.session_state.chat},
            indent=2,
            ensure_ascii=False,
        )

    st.download_button(
        "⬇️ Download Transcript",
        data=transcript_json().encode("utf-8"),
        file_name="odi_chat_transcript.json",
        mime="application/json",
        use_container_width=True,
    )

if st.session_state.last_confirm_result:
    st.info(st.session_state.last_confirm_result)

# -----------------------
# ✅ PA6 Batch Evaluation Section
# -----------------------
st.divider()
st.header("📊 PA6 Batch Evaluation (Run Questions × Selectable LLMs)")

colA, colB = st.columns([2, 1])
with colA:
    questions_text = st.text_area(
        "Paste your questions (one per line). Example: Colby 10 questions.",
        height=220,
        placeholder="Question 1...\nQuestion 2...\n..."
    )
with colB:
    top_k = st.number_input("RAG top_k", min_value=1, max_value=15, value=5, step=1)
    batch_temp = st.slider("Batch temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    batch_max_tokens = st.number_input("Batch max_tokens", min_value=100, max_value=2000, value=600, step=50)
    include_context_col = st.checkbox("Include rag_context column in downloads", value=True)

    selected_labels = st.multiselect(
        "Choose LLM(s) to run in batch",
        options=list(BATCH_MODELS.keys()),
        default=list(BATCH_MODELS.keys()),
    )
    selected_models = {label: BATCH_MODELS[label] for label in selected_labels}

questions = [q.strip() for q in questions_text.splitlines() if q.strip()]

run_disabled = (not api_key) or (len(questions) == 0) or (len(selected_models) == 0)
if (not api_key) or (len(questions) == 0):
    st.caption("To run batch: enter API key + paste at least 1 question. Also upload & embed CSVs for real RAG results.")
elif len(selected_models) == 0:
    st.caption("Select at least one LLM to run.")

if st.button("🚀 Run Batch", disabled=run_disabled, use_container_width=True):
    if st.session_state.vector_db is None or not st.session_state.vector_db_ready:
        st.warning("RAG is not ready. Please upload CSVs and click 'Load & Embed CSVs' first.")
    else:
        with st.spinner("Running batch… this will call the selected model(s) for each question."):
            df = run_batch_eval(
                api_key=api_key,
                task=task, persona=persona, tone=tone, data_rules=data_rules, scope=scope,
                pref_schema=pref_schema, mapping_guide=mapping_guide, workflow=workflow, output_rules=output_rules,
                questions=questions,
                selected_models=selected_models,
                top_k=int(top_k),
                temperature=float(batch_temp),
                max_tokens=int(batch_max_tokens),
            )

        if not include_context_col and "rag_context" in df.columns:
            df = df.drop(columns=["rag_context"])

        # Ensure conversation_id is still first
        if not df.empty and "conversation_id" in df.columns:
            df = df[["conversation_id"] + [c for c in df.columns if c != "conversation_id"]]

        st.session_state.batch_df = df
        st.session_state.batch_last_run = time.strftime("%Y-%m-%d %H:%M:%S")
        st.success(f"Batch finished at {st.session_state.batch_last_run}. Rows: {len(df)}")

if st.session_state.batch_df is not None:
    st.subheader("✅ Batch Results")
    st.caption(f"Last run: {st.session_state.batch_last_run}")
    st.dataframe(st.session_state.batch_df, use_container_width=True)

    csv_bytes = st.session_state.batch_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download CSV (.csv)",
        data=csv_bytes,
        file_name=f"pa6_batch_results_{time.strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    json_payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "row_count": int(len(st.session_state.batch_df)),
        "data": st.session_state.batch_df.to_dict(orient="records"),
    }
    json_bytes = json.dumps(json_payload, indent=2, ensure_ascii=False).encode("utf-8")
    st.download_button(
        "⬇️ Download JSON (.json)",
        data=json_bytes,
        file_name=f"pa6_batch_results_{time.strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True,
    )

# -----------------------
# Chat UI (unchanged)
# -----------------------
st.divider()
st.header("💬 Chat (Single LLM)")

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_msg = st.chat_input("Ask something...")
if user_msg:
    st.session_state.chat.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    context = rag_retrieve_context(user_msg)

    if show_debug:
        with st.expander("🔎 Retrieved RAG Context (Debug)", expanded=True):
            st.text(context)

    sys_prompt = build_system_prompt(
        task, persona, tone, data_rules, scope, pref_schema, mapping_guide, workflow, output_rules, context
    )

    with st.chat_message("assistant"):
        try:
            reply = call_llm_openrouter(api_key, model, sys_prompt, st.session_state.chat)
            st.markdown(reply)
            st.session_state.chat.append({"role": "assistant", "content": reply})
        except Exception as e:
            st.error(str(e))
