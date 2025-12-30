# STREAMLIT APP with RAG EXTENSION (from your original base)

import json
import time
from typing import Optional, Dict, List, Any

import pandas as pd
import requests
import streamlit as st

# RAG Libraries
import chromadb
from chromadb.utils import embedding_functions
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

# -----------------------
# RAG Functions
# -----------------------
def init_vector_db():
    client = chromadb.Client()
    st.session_state.vector_db = client.create_collection(name="odi-grips")

def embed_texts(texts: List[str]) -> List[List[float]]:
    return st.session_state.embed_model.encode(texts).tolist()

def add_to_vector_db():
    if st.session_state.vector_db is None:
        init_vector_db()
    all_texts, ids, metadata = [], [], []
    idx = 0
    for fname, df in st.session_state.datasets.items():
        for _, row in df.iterrows():
            text = ", ".join([f"{k}: {v}" for k, v in row.items() if pd.notna(v)])
            all_texts.append(text)
            ids.append(f"chunk-{idx}")
            metadata.append({"source": fname})
            idx += 1
    embeddings = embed_texts(all_texts)
    st.session_state.vector_db.add(documents=all_texts, embeddings=embeddings, ids=ids, metadatas=metadata)

def rag_retrieve_context(query: str, top_k: int = 5) -> str:
    if st.session_state.vector_db is None:
        return "No embedded product data available."
    embedded_query = embed_texts([query])[0]
    results = st.session_state.vector_db.query(query_embeddings=[embedded_query], n_results=top_k)
    return "\n\n".join(results["documents"][0]) if results["documents"] else "No matching context."

# -----------------------
# Defaults (Structured Prompts)
# -----------------------
DEFAULT_TASK = "You are an ODI mountain bike grips expert who provides grip recommendations to users."
DEFAULT_PERSONA = "The user is an experienced mountain biker. Use technical terms and slang."
DEFAULT_TONE = "Respond in a professional and informative tone, similar to a customer service representative."
DEFAULT_DATA_RULES = """DATA RULES (STRICT):
- Use ONLY information retrieved from the embedded ODI product dataset.
- Do NOT browse the web, cite webpages, or use external reviews/knowledge.
- Do NOT invent features, prices, specs, availability, or “best overall” claims.
- If a detail is not in the retrieved context, say you’re not sure and ask a clarifying question instead.
- Only ODI grips are allowed. Never recommend competitor brands.
"""
DEFAULT_SCOPE = """SCOPE:
This assistant supports ALL ODI grips in the dataset (e.g., MTB, BMX, Moto, Urban/Casual).
If the user asks about a category not supported, explain the limitation and ask follow-up.
"""
DEFAULT_PREF_SCHEMA = """PREFERENCES:
- riding_style: trail, enduro, downhill, cross-country, bmx, moto, urban, casual
- locking_mechanism: lock-on, slip-on
- thickness: thin, medium, thick, medium-thick size xl
- damping_level: low, medium, high
- durability: low, medium, high
"""
DEFAULT_MAPPING = """MAPPING HINTS:
- “large hands / XL gloves” -> medium-thick size xl
- “shock absorption / vibration” -> high damping
"""
DEFAULT_WORKFLOW = """WORKFLOW:
1) Ask what bike and problem the user has
2) Identify riding_style early if possible
3) Ask ONE follow-up if preferences unclear
4) Recommend based ONLY on matched chunks
5) Explain match briefly and clearly
"""
DEFAULT_OUTPUT_RULES = """RESPONSE FORMAT:
- Replies = 2–6 sentences
- Include:
  (a) acknowledgment
  (b) recommendation
  (c) one follow-up if needed
- Avoid long lists or multiple follow-ups
"""

# -----------------------
# Prompt Assembly
# -----------------------
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
- Only use above context. Never invent.
- Ask follow-up if info is missing.
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
    data = resp.json()
    return data["choices"][0]["message"]["content"]

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
if st.button("🔄 Load & Embed CSVs"):
    st.session_state.datasets = {}
    for f in csv_files:
        df = pd.read_csv(f)
        st.session_state.datasets[f.name] = df
    add_to_vector_db()
    st.success("Files embedded for RAG.")

# Prompt settings
st.subheader("Structured Prompt Controls")
task = st.text_area("Task", value=DEFAULT_TASK)
persona = st.text_area("Persona", value=DEFAULT_PERSONA)
tone = st.text_area("Tone", value=DEFAULT_TONE)
data_rules = st.text_area("Data Rules", value=DEFAULT_DATA_RULES)
scope = st.text_area("Scope", value=DEFAULT_SCOPE)
pref_schema = st.text_area("Preferences", value=DEFAULT_PREF_SCHEMA)
mapping_guide = st.text_area("Mapping", value=DEFAULT_MAPPING)
workflow = st.text_area("Workflow", value=DEFAULT_WORKFLOW)
output_rules = st.text_area("Format Rules", value=DEFAULT_OUTPUT_RULES)

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
    sys_prompt = build_system_prompt(task, persona, tone, data_rules, scope, pref_schema, mapping_guide, workflow, output_rules, context)

    with st.chat_message("assistant"):
        try:
            reply = call_llm_openrouter(api_key, model, sys_prompt, st.session_state.chat)
            st.markdown(reply)
            st.session_state.chat.append({"role": "assistant", "content": reply})
        except Exception as e:
            st.error(str(e))
