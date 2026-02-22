# STREAMLIT APP with RAG + Batch Runner (BATCH-ONLY)
# ✅ Minimal change: removed single chat UI + removed "Chat LLM (Single Chat Mode)" selector
# ✅ Keeps: OpenRouter API key input + Show RAG Debug + CSV upload/embed + batch runner
# ✅ Batch runs BOTH Prompt A and Prompt B:
#    Loop: question -> model -> prompt_id in {A,B}
# ✅ Batch CSV download outputs an R-friendly LONG format:
#    conversation_id, prompt_id, llm, llm_label, role, content, product_recommended
# ✅ JSON download includes prompt_id too
#
# ✅ MINIMAL CHANGE (THIS EDIT):
# - Update preferred_fields to match your test CSV columns
# - Use `name` as the product name column (affects product cards + product_recommended extraction)

import io
import json
import time
import re
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
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


# -----------------------
# State
# -----------------------
def init_state():
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

    if "batch_df" not in st.session_state:
        st.session_state.batch_df = None
    if "batch_last_run" not in st.session_state:
        st.session_state.batch_last_run = ""

    if "embed_status_lines" not in st.session_state:
        st.session_state.embed_status_lines = []


# -----------------------
# RAG Functions
# -----------------------
def init_vector_db():
    client = chromadb.Client()  # in-memory
    st.session_state.vector_db = client.get_or_create_collection(name="odi-grips")
    st.session_state.vector_db_ready = True


def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_embed_model()
    return model.encode(texts).tolist()


def _row_to_product_card(row: pd.Series) -> str:
    row_dict = {}
    for k, v in row.items():
        kk = str(k).strip()
        vv = "" if v is None else str(v).strip()
        row_dict[kk] = vv
        row_dict[kk.lower()] = vv

    def get_val(*keys: str) -> str:
        for key in keys:
            if key in row_dict and row_dict[key].strip() != "":
                return row_dict[key].strip()
            lk = key.lower()
            if lk in row_dict and row_dict[lk].strip() != "":
                return row_dict[lk].strip()
        return ""

    # ✅ MINIMAL CHANGE: use `name` as product name for your test CSV
    product_name = get_val("name", "Name", "product_name", "Product Name", "title", "Title")

    # ✅ MINIMAL CHANGE: preferred_fields updated to your column schema
    preferred_fields = [
        ("Name", ["name", "Name"]),
        ("Length", ["length", "Length"]),
        ("Key Features", ["key_features", "Key Features"]),
        ("Thickness", ["thickness", "Thickness"]),
        ("Colors", ["colors", "Colors"]),
        ("Durability", ["durability", "Durability"]),
        ("Damping Level", ["damping_level", "Damping Level"]),
        ("Grip Pattern", ["grip_pattern", "Grip Pattern"]),
        ("Feel", ["feel", "Feel"]),
        ("Traction", ["traction", "Traction"]),
        ("Price", ["price", "Price"]),
        ("Ergonomics", ["ergonomics", "Ergonomics"]),
        ("Grip Attachment System", ["grip attachment system", "Grip Attachment System"]),
        ("Co-Branding", ["co_branding", "Co Branding", "co-branding", "Co-Branding"]),
        ("Locking Mechanism", ["locking_mechanism", "Locking Mechanism"]),
        ("Riding Style", ["riding_style", "Riding Style"]),
        ("Differentiator", ["differentiator", "Differentiator"]),
    ]

    lines = []
    if product_name:
        lines.append(f"Product: {product_name}")

    for label, keys in preferred_fields:
        val = get_val(*keys)
        if val:
            lines.append(f"{label}: {val}")

    if len(lines) == 0:
        extras = []
        for k, v in row_dict.items():
            if k != k.lower():
                continue
            if v.strip():
                extras.append(f"{k}: {v.strip()}")
            if len(extras) >= 12:
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
    total_rows = 0
    skipped_existing = 0
    skipped_truly_empty = 0

    for fname, df in st.session_state.datasets.items():
        df = df.reset_index(drop=True)
        total_rows += len(df)

        for i, row in df.iterrows():
            chunk_id = f"{fname}::row-{i}"
            if chunk_id in existing_ids:
                skipped_existing += 1
                continue

            card = _row_to_product_card(row)
            if not card.strip():
                skipped_truly_empty += 1
                continue

            all_texts.append(card)
            ids.append(chunk_id)
            metadata.append({"source": fname, "row_index": int(i)})

    st.info(
        f"Loaded rows: {total_rows} | Embedded: {len(all_texts)} | "
        f"Skipped existing: {skipped_existing} | Skipped empty rows: {skipped_truly_empty}"
    )

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
# Defaults (5 boxes)
# -----------------------
DEFAULT_TASK = """<TASK>
You are an expert ODI grip specialist. Recommend the most suitable ODI grip based strictly on the uploaded CSV dataset and the retrieved RAG context.
""".strip()

DEFAULT_DATA_RULES = """<DATA ACCESS (STRICT)>
- Use ONLY information from the retrieved RAG context.
- Do NOT use outside knowledge.
- Do NOT invent specifications or claims.
- If the context is insufficient, still choose the closest match from the RAG context and briefly state what information is missing.
""".strip()

DEFAULT_STYLE = """<STYLE>
Use clear and direct language.
""".strip()

DEFAULT_DECISION_RULE = """<DECISION RULE>
- You MUST recommend EXACTLY ONE product for every question.
- The recommended product_name MUST appear in the current RAG context.
- If the match is weak, still pick the closest match from the RAG context and briefly state what information is missing.
- Do NOT ask clarifying questions.
""".strip()

DEFAULT_OUTPUT_RULES = """<OUTPUT RULES>
Return:
Recommended Product: <exact product_name from RAG context>
Reason: <1–2 sentences based only on retrieved specs>
""".strip()


def build_system_prompt(
    task: str,
    data_rules: str,
    style: str,
    decision_rule: str,
    output_rules: str,
    rag_context: str
) -> str:
    return f"""
{task}

{data_rules}

{style}

{decision_rule}

{output_rules}

<RAG Context Retrieved for this Query>
{rag_context}

IMPORTANT:
- Use the RAG context above as your source of truth.
- The recommended product name must appear in the RAG context.
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
# Batch Runner Helpers
# -----------------------
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


# -----------------------
# helper: extract product recommendation from assistant text
# -----------------------
def _build_product_name_set() -> set:
    names = set()
    for _fname, _df in st.session_state.datasets.items():
        cols_lc = [c.lower() for c in _df.columns]

        # ✅ MINIMAL CHANGE: accept `name` as the product name column
        if "name" in cols_lc:
            col = _df.columns[cols_lc.index("name")]
        elif "product_name" in cols_lc:
            col = _df.columns[cols_lc.index("product_name")]
        else:
            continue

        for v in _df[col].astype(str).fillna("").tolist():
            vv = v.strip()
            if vv:
                names.add(vv)
    return names


def extract_product_recommended(assistant_text: str) -> str:
    if not assistant_text:
        return ""
    if not st.session_state.get("datasets"):
        return ""

    product_names = _build_product_name_set()
    if not product_names:
        return ""

    text_lc = assistant_text.lower()

    for name in sorted(product_names, key=len, reverse=True):
        if name.lower() in text_lc:
            return name

    m = re.search(r"(?im)^\s*Recommended\s*Product\s*:\s*(.+)\s*$", assistant_text)
    if m:
        return m.group(1).strip()

    m2 = re.search(r"(?im)^\s*Product\s*:\s*(.+)\s*$", assistant_text)
    if m2:
        return m2.group(1).strip()

    return ""


# -----------------------
# convert batch_df (wide) -> R-friendly long CSV (includes prompt_id)
# -----------------------
def batch_df_to_r_long(df: pd.DataFrame, label_map: Dict[str, str]) -> pd.DataFrame:
    cols = ["conversation_id", "prompt_id", "llm", "llm_label", "role", "content", "product_recommended"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)

    out_rows = []
    for _, r in df.iterrows():
        conv_id = r.get("conversation_id", "")
        prompt_id = r.get("prompt_id", "")
        llm = r.get("model", "")
        llm_label = label_map.get(llm, "")

        msgs = r.get("messages", [])
        if not isinstance(msgs, list):
            msgs = []

        assistant_text = ""
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "assistant":
                assistant_text = m.get("content", "") or ""
                break
        prod = extract_product_recommended(assistant_text)

        for m in msgs:
            if not isinstance(m, dict):
                continue
            out_rows.append({
                "conversation_id": conv_id,
                "prompt_id": prompt_id,
                "llm": llm,
                "llm_label": llm_label,
                "role": m.get("role", ""),
                "content": m.get("content", ""),
                "product_recommended": prod if m.get("role") == "assistant" else "",
            })

    long_df = pd.DataFrame(out_rows)
    for c in cols:
        if c not in long_df.columns:
            long_df[c] = ""
    return long_df[cols]


# -----------------------
# run_batch_eval loops prompt_id in {A,B}
# -----------------------
def run_batch_eval(
    api_key: str,
    prompts: Dict[str, Dict[str, str]],
    questions: List[str],
    selected_models: Dict[str, str],
    top_k: int = 5,
    temperature: float = 0.2,
    max_tokens: int = 600,
) -> pd.DataFrame:
    rows = []
    model_items = list(selected_models.items())
    prompt_items = list(prompts.items())

    total_calls = max(1, len(questions) * len(model_items) * len(prompt_items))
    done = 0
    prog = st.progress(0)

    for idx, q in enumerate(questions, start=1):
        rag_context = rag_retrieve_context(q, top_k=top_k)

        for _label, model_id in model_items:
            for prompt_id, p in prompt_items:
                sys_prompt = build_system_prompt(
                    task=p["task"],
                    data_rules=p["data_rules"],
                    style=p["style"],
                    decision_rule=p["decision_rule"],
                    output_rules=p["output_rules"],
                    rag_context=rag_context
                )

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
                    "prompt_id": prompt_id,
                    "model": model_id,
                    "messages": [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": ans},
                    ],
                })

                done += 1
                prog.progress(min(1.0, done / total_calls))

    df = pd.DataFrame(rows)
    if not df.empty and "conversation_id" in df.columns:
        df = df[["conversation_id"] + [c for c in df.columns if c != "conversation_id"]]
    return df


# -----------------------
# Streamlit UI Start (BATCH ONLY)
# -----------------------
init_state()

st.set_page_config(page_title="ODI Grips Batch Eval (RAG)", page_icon="🚵", layout="wide")
st.title("🚵 ODI Grips Batch Evaluation (with RAG)")

PING_MODEL = "openai/gpt-4.1-mini"

with st.sidebar:
    st.header("LLM Settings")
    api_key = st.text_input("OpenRouter API Key", type="password")
    show_debug = st.checkbox("Show RAG Debug (retrieved context)", value=False)

if show_debug and st.session_state.datasets:
    st.sidebar.markdown("### ✅ Loaded CSV Columns")
    for fname, df in st.session_state.datasets.items():
        st.sidebar.write(fname)
        st.sidebar.write(list(df.columns))

st.subheader("📁 Upload CSV Files")
csv_files = st.file_uploader("Upload ODI product CSVs", type=["csv"], accept_multiple_files=True)

if st.button("🔄 Load & Embed CSVs"):
    st.session_state.vector_db = None
    st.session_state.vector_db_ready = False
    init_vector_db()

    st.session_state.datasets = {}
    for f in csv_files:
        df = pd.read_csv(
            f,
            dtype=str,
            engine="python",
            keep_default_na=False
        )
        df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
        df = df.fillna("")
        st.session_state.datasets[f.name] = df

    st.session_state.embed_status_lines = []
    add_to_vector_db()
    st.session_state.embed_status_lines.append("✅ CSV files loaded. Vector index rebuilt for RAG.")

if st.session_state.get("embed_status_lines"):
    for line in st.session_state.embed_status_lines:
        if line.startswith("❌"):
            st.error(line)
        else:
            st.success(line)

with st.expander("🧠 Structured Prompt Controls", expanded=True):
    st.subheader("Prompt Settings")

    colP1, colP2 = st.columns(2)

    with colP1:
        st.markdown("### Prompt A")
        task_A = st.text_area("Task (A)", value=DEFAULT_TASK, height=90)
        data_rules_A = st.text_area("Data Access (Strict) (A)", value=DEFAULT_DATA_RULES, height=140)
        style_A = st.text_area("Style (A)", value=DEFAULT_STYLE, height=100)
        decision_rule_A = st.text_area("Decision Rule (A)", value=DEFAULT_DECISION_RULE, height=170)
        output_rules_A = st.text_area("Output Rules (A)", value=DEFAULT_OUTPUT_RULES, height=90)

    with colP2:
        st.markdown("### Prompt B")
        task_B = st.text_area("Task (B)", value=DEFAULT_TASK, height=90)
        data_rules_B = st.text_area("Data Access (Strict) (B)", value=DEFAULT_DATA_RULES, height=140)
        style_B = st.text_area("Style (B)", value=DEFAULT_STYLE, height=100)
        decision_rule_B = st.text_area("Decision Rule (B)", value=DEFAULT_DECISION_RULE, height=170)
        output_rules_B = st.text_area("Output Rules (B)", value=DEFAULT_OUTPUT_RULES, height=90)

st.subheader("Actions")
a1 = st.columns(1)[0]
with a1:
    if st.button("✅ Confirm LLM Setup", use_container_width=True):
        if not api_key:
            st.session_state.llm_confirmed = False
            st.session_state.last_confirm_result = "Missing OpenRouter API key."
        else:
            try:
                ping_system = "You are a helpful assistant. Reply exactly with 'LLM OK'."
                ping_messages = [{"role": "user", "content": "LLM OK"}]
                out = call_llm_openrouter(api_key, PING_MODEL, ping_system, ping_messages, temperature=0.0, max_tokens=50)
                st.session_state.llm_confirmed = "LLM OK" in out
                st.session_state.last_confirm_result = f"Response: {out}"
                st.toast("LLM setup checked.")
            except Exception as e:
                st.session_state.llm_confirmed = False
                st.session_state.last_confirm_result = f"Error: {e}"

if st.session_state.last_confirm_result:
    st.info(st.session_state.last_confirm_result)

# -----------------------
# Batch Evaluation Section (only)
# -----------------------
st.divider()
st.header("📊 Batch Evaluation (Run Questions × Selectable LLMs × Prompt A/B)")

colA, colB = st.columns([2, 1])
with colA:
    questions_text = st.text_area(
        "Paste your questions (one per line).",
        height=220,
        placeholder="Question 1...\nQuestion 2...\n..."
    )
with colB:
    top_k = st.number_input("RAG top_k", min_value=1, max_value=15, value=5, step=1)
    batch_temp = st.slider("Batch temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    batch_max_tokens = st.number_input("Batch max_tokens", min_value=100, max_value=2000, value=600, step=50)

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
        prompts = {
            "A": {
                "task": task_A,
                "data_rules": data_rules_A,
                "style": style_A,
                "decision_rule": decision_rule_A,
                "output_rules": output_rules_A,
            },
            "B": {
                "task": task_B,
                "data_rules": data_rules_B,
                "style": style_B,
                "decision_rule": decision_rule_B,
                "output_rules": output_rules_B,
            },
        }

        with st.spinner("Running batch… this will call (Prompt A + Prompt B) for each LLM and each question."):
            df = run_batch_eval(
                api_key=api_key,
                prompts=prompts,
                questions=questions,
                selected_models=selected_models,
                top_k=int(top_k),
                temperature=float(batch_temp),
                max_tokens=int(batch_max_tokens),
            )

        st.session_state.batch_df = df
        st.session_state.batch_last_run = time.strftime("%Y-%m-%d %H:%M:%S")
        st.success(f"Batch finished at {st.session_state.batch_last_run}. Rows: {len(df)}")

if st.session_state.batch_df is not None:
    st.subheader("✅ Batch Results")
    st.caption(f"Last run: {st.session_state.batch_last_run}")
    st.dataframe(st.session_state.batch_df, use_container_width=True)

    model_id_to_label = {v: k for k, v in BATCH_MODELS.items()}
    batch_long_df = batch_df_to_r_long(st.session_state.batch_df, model_id_to_label)

    csv_bytes = batch_long_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download CSV (.csv) — R-friendly (LONG)",
        data=csv_bytes,
        file_name=f"batch_results_long_{time.strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    json_bytes = json.dumps(
        st.session_state.batch_df.to_dict(orient="records"),
        indent=2,
        ensure_ascii=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download JSON (.json)",
        data=json_bytes,
        file_name=f"batch_results_{time.strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True,
    )
