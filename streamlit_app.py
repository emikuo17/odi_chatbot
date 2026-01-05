# STREAMLIT APP with RAG EXTENSION (minimal fixes for issues 1–5)

import json
import time
from typing import Optional, Dict, List, Any

import pandas as pd
import requests
import streamlit as st

# RAG Libraries
import chromadb
from sentence_transformers import SentenceTransformer

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
    if "embed_model" not in st.session_state:
        st.session_state.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    if "vector_db_ready" not in st.session_state:
        st.session_state.vector_db_ready = False

# -----------------------
# RAG Functions
# -----------------------
def init_vector_db():
    # FIX (Issue #2): Use persistent client so embeddings survive reruns/restarts
    client = chromadb.PersistentClient(path="chroma_store")

    # FIX (Issue #2): Use get_or_create_collection to avoid recreating each time
    st.session_state.vector_db = client.get_or_create_collection(name="odi-grips")
    st.session_state.vector_db_ready = True


def embed_texts(texts: List[str]) -> List[List[float]]:
    return st.session_state.embed_model.encode(texts).tolist()


def _row_to_product_card(row: pd.Series) -> str:
    """
    FIX (Issue #5): Cleaner "product card" chunk formatting improves retrieval quality.
    Uses whichever columns exist; avoids noisy "k: v, k: v" blobs.
    """
    # Try common name keys if present (doesn't break if missing)
    name_key_candidates = ["product_name", "name", "title", "Product", "Product Name"]
    prod_name = None
    for k in name_key_candidates:
        if k in row.index and pd.notna(row[k]):
            prod_name = str(row[k]).strip()
            break

    # Compact key fields if they exist in your schema
    preferred_keys = [
        "category", "Category",
        "riding_style", "Riding Style",
        "locking_mechanism", "Locking Mechanism",
        "thickness", "Thickness",
        "damping_level", "Damping Level",
        "durability", "Durability",
        "pattern", "Pattern",
        "compound", "Compound",
        "material", "Material",
        "weight", "Weight",
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
                # Normalize label to avoid duplicate label variants
                label = str(k).replace("_", " ").title()
                lines.append(f"{label}: {val}")

    # Fallback: include a few remaining fields if we found nothing (avoid empty chunks)
    if len(lines) <= 1:
        extras = []
        for k, v in row.items():
            if pd.isna(v):
                continue
            if str(k) in preferred_keys:
                continue
            extras.append(f"{str(k)}: {str(v).strip()}")
            if len(extras) >= 8:
                break
        if extras:
            lines.append("Other:")
            lines.extend(extras)

    return "\n".join(lines).strip()


def add_to_vector_db():
    # Ensure vector DB exists
    if st.session_state.vector_db is None:
        init_vector_db()

    # FIX (Issue #1): Use stable IDs to avoid duplicates/collisions
    # FIX (Issue #1): Avoid re-embedding same IDs on repeated "Load & Embed"
    existing_ids = set()
    try:
        # Chroma may not support listing all IDs in every config;
        # we guard this to keep it compatible.
        peek = st.session_state.vector_db.peek()
        for _id in peek.get("ids", []):
            existing_ids.add(_id)
    except Exception:
        pass

    all_texts, ids, metadata = [], [], []

    for fname, df in st.session_state.datasets.items():
        df = df.reset_index(drop=True)
        for i, row in df.iterrows():
            chunk_id = f"{fname}::row-{i}"  # stable
            if chunk_id in existing_ids:
                continue

            card = _row_to_product_card(row)
            if not card:
                continue

            all_texts.append(card)
            ids.append(chunk_id)
            metadata.append({"source": fname, "row_index": int(i)})

    if not all_texts:
        st.info("No new rows to embed (already embedded or empty).")
        return

    embeddings = embed_texts(all_texts)
    st.session_state.vector_db.add(
        documents=all_texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadata,
    )
    st.success(f"Embedded {len(all_texts)} new product rows.")


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

    # Present as clearly separated product cards
    return "\n\n---\n\n".join(docs[0])

# -----------------------
# Defaults (Structured Prompts)
# -----------------------
# FIX (Issue #3): remove accidental string concatenation duplicates
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

# -----------------------
# Prompt Assembly
# -----------------------
def build_system_prompt(task, persona, tone, data_rules, scope, pref_schema, mapping_guide, workflow, output_rules, rag_context):
    # FIX (Issue #4): Do not over-restrict when retrieval is empty; guide fallback behavior
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
def call_llm_openrouter(api_key: str, model: str, system_prompt: str, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 600) -> str:
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
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"Unexpected response format: {json.dumps(data)[:1200]}")

# -----------------------
# Streamlit UI Start
# -----------------------
init_state()

st.set_page_config(page_title="ODI Grips Chatbot (RAG)", page_icon="🚵", layout="wide")
st.title("🚵 ODI Grips Chatbot (with RAG)")

with st.sidebar:
    st.header("LLM Settings")
    api_key = st.text_input("OpenRouter API Key", type="password")
    model = st.text_input("Model", value="openai/gpt-4o-mini")

st.subheader("📁 Upload CSV Files")
csv_files = st.file_uploader("Upload ODI product CSVs", type=["csv"], accept_multiple_files=True)

# FIX (Issue #1/#2): make embedding button idempotent + show status
if st.button("🔄 Load & Embed CSVs"):
    st.session_state.datasets = {}
    for f in csv_files:
        df = pd.read_csv(f)
        st.session_state.datasets[f.name] = df

    # ensure persistent DB exists
    if st.session_state.vector_db is None:
        init_vector_db()

    add_to_vector_db()
    st.success("CSV files loaded. Vector index updated for RAG.")

# Prompt settings
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

# ---- Actions Row (LLM Setup / Clear / Export) ----
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
                out = call_llm_openrouter(api_key, model, ping_system, ping_messages, temperature=0.0, max_tokens=10)
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
        return json.dumps({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "messages": st.session_state.chat}, indent=2)

    st.download_button(
        "⬇️ Download Transcript",
        data=transcript_json().encode("utf-8"),
        file_name="odi_chat_transcript.json",
        mime="application/json",
        use_container_width=True,
    )

# Chat section
st.subheader("💬 Chat")
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_msg = st.chat_input("Ask something...")
if user_msg:
    st.session_state.chat.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    context = rag_retrieve_context(user_msg)

    # NOTE: your original code called build_system_prompt(); keep that name if you prefer.
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
