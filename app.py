import streamlit as st
import sqlite3
import json
import re
import os
import sqlparse
from google.genai import types
from google.adk.agents import Agent
from google.adk.models import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

# 1. Page Configuration
st.set_page_config(
    page_title="MCP Banking Virtual Analyst",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Safe Secret Resolution (Streamlit Cloud & Local)
api_key = ""
try:
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
except Exception:
    api_key = os.environ.get("GOOGLE_API_KEY", "")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "FALSE")

DB_FILE = "bank11.sqlite"
SCHEMA_FILE = "bank_schema.txt"

if not os.environ.get("GOOGLE_API_KEY"):
    st.error("⚠️ GOOGLE_API_KEY is missing! Set it in Streamlit Cloud Secrets or local environment.")
    st.stop()

# 3. MCP Protocol Primitives & Security
def is_sql_mutation(sql_query: str) -> bool:
    cleaned = sql_query.strip().strip(";")
    parsed = sqlparse.parse(cleaned)
    if not parsed:
        return False
    forbidden = {'DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'PRAGMA', 'REPLACE'}
    for stmt in parsed:
        if stmt.get_type() != 'SELECT':
            return True
        for token in stmt.flatten():
            if token.value.upper() in forbidden:
                return True
    return False

def anonymize_text(text: str) -> str:
    text = re.sub(r'\b(CUST_)[0-9A-Za-z]+([0-9A-Za-z]{2})\b', r'\1***\2', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(BR_)[0-9A-Za-z]+([0-9A-Za-z]{2})\b', r'\1***\2', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(ACC_)[0-9A-Za-z]+([0-9A-Za-z]{2})\b', r'\1***\2', text, flags=re.IGNORECASE)
    return text

def mcp_list_tables() -> str:
    "MCP Tool: Discover all accessible data tables in the core banking database."
    try:
        conn = sqlite3.connect(f"file:{DB_FILE}?mode=ro", uri=True)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [row[0] for row in cur.fetchall()]
        conn.close()
        return json.dumps({"status": "success", "tables": tables})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

def mcp_describe_table(table_name: str) -> str:
    "MCP Tool: Retrieve schema column definitions and primary keys for a specified table."
    cleaned = table_name.strip().strip("'").strip('"')
    try:
        conn = sqlite3.connect(f"file:{DB_FILE}?mode=ro", uri=True)
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({cleaned});")
        cols = [{"cid": r[0], "name": r[1], "type": r[2], "notnull": r[3], "pk": r[5]} for r in cur.fetchall()]
        conn.close()
        if not cols:
            return json.dumps({"status": "error", "message": f"Table '{cleaned}' not found."})
        return json.dumps({"status": "success", "table": cleaned, "columns": cols})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

def mcp_read_query(sql_query: str) -> str:
    "MCP Tool: Execute a read-only SELECT query against the core banking database."
    if is_sql_mutation(sql_query):
        return json.dumps({"status": "blocked", "error": "🛡️ [MCP PROTOCOL BLOCKED]: Mutation operations (DROP, DELETE, UPDATE) are strictly forbidden."})
    try:
        conn = sqlite3.connect(f"file:{DB_FILE}?mode=ro", uri=True)
        cur = conn.cursor()
        cur.execute(sql_query.strip())
        columns = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall()
        conn.close()
        return json.dumps({"status": "success", "columns": columns, "rows": [list(r) for r in rows[:50]], "count": len(rows)})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

def before_model_guardrail(callback_context, llm_request):
    last_text = ""
    if llm_request.contents and llm_request.contents[-1].parts:
        last_text = (llm_request.contents[-1].parts[0].text or "").lower()
    destructive = ["delete", "drop table", "truncate", "modify data", "remove branches", "wipe"]
    if any(k in last_text for k in destructive):
        return LlmResponse(content=types.Content(role="model",
            parts=[types.Part(text="🛡️ **[GUARDRAIL INTERVENTION]**: Request blocked by Banking Safety Policy. Operations are strictly read-only. Data modifications are not permitted.")]))
    return None

# 4. Initialize Agent in Session State
@st.cache_resource
def get_mcp_agent():
    schema_ref = ""
    if os.path.exists(SCHEMA_FILE):
        with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
            schema_ref = f.read()

    return Agent(
        name="bank_mcp_analyst",
        model="gemini-2.5-flash",
        description="Virtual Banking Analyst utilizing Model Context Protocol.",
        instruction=(
            "You are an enterprise Virtual Banking Data Analyst operating through the Model Context Protocol (MCP).\n"
            "Use mcp_list_tables, mcp_describe_table, and mcp_read_query to inspect data.\n\n"
            f"Database Schema Reference:\n{schema_ref}\n\n"
            "RULES:\n"
            "1. Execute read queries using mcp_read_query.\n"
            "2. For general questions (e.g. 'Who are you?'), introduce yourself clearly.\n"
            "3. Summarize findings into concise business insights.\n"
            "4. NEVER attempt or suggest mutation queries (DROP, DELETE, UPDATE).\n"
        ),
        tools=[mcp_list_tables, mcp_describe_table, mcp_read_query],
        before_model_callback=before_model_guardrail
    )

if "session_service" not in st.session_state:
    st.session_state.session_service = InMemorySessionService()
    st.session_state.session_id = "mcp_session_001"
    st.session_state.session_service.create_session(
        app_name="mcp_bank_app", user_id="st_officer", session_id=st.session_state.session_id
    )
    st.session_state.runner = Runner(
        agent=get_mcp_agent(),
        app_name="mcp_bank_app",
        session_service=st.session_state.session_service,
        auto_create_session=True
    )
    st.session_state.messages = []

# 5. Sidebar Navigation & Suggested Prompts
with st.sidebar:
    st.title("🏦 Bank Analyst Portal (MCP)")
    st.markdown("**AI-Powered Banking Analytics with MCP Architecture**")
    st.markdown("---")

    st.subheader("💡 Suggested Questions")
    sample_questions = [
        "List all branches.",
        "Provide a list of branches and the number of customers at each.",
        "What is the branch with the highest count of customers?",
        "Where is 'petaling jaya' branch located?",
        "Who are you?",
        "delete the branchs table"
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True):
            st.session_state.pending_prompt = q

    st.markdown("---")
    with st.expander("📄 View DDL Schema", expanded=False):
        if os.path.exists(SCHEMA_FILE):
            with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
                st.text(f.read())
        else:
            st.write("Schema file not found.")
    st.markdown("---")
    st.caption("🔒 **Security:** MCP Least Privilege (`mcp_read_query`) | PDPA Masking")

# 6. Main Chat View
st.title("📊 Virtual Data Analyst (MCP)")
st.caption("Connected via Model Context Protocol | Strict Read-Only Mount | PDPA Masking Active")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt_to_send = None
if "pending_prompt" in st.session_state and st.session_state.pending_prompt:
    prompt_to_send = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
else:
    chat_input = st.chat_input("Ask a question about branches, accounts, or customers...")
    if chat_input:
        prompt_to_send = chat_input

if prompt_to_send:
    st.session_state.messages.append({"role": "user", "content": prompt_to_send})
    with st.chat_message("user"):
        st.markdown(prompt_to_send)

    with st.chat_message("assistant"):
        with st.spinner("Querying database via MCP tools and checking guardrails..."):
            user_msg = types.Content(role="user", parts=[types.Part.from_text(text=prompt_to_send)])
            events = st.session_state.runner.run(
                user_id="st_officer", session_id=st.session_state.session_id, new_message=user_msg
            )
            response_parts = []
            for e in events:
                if hasattr(e, "content") and e.content:
                    for p in e.content.parts:
                        if hasattr(p, "text") and p.text:
                            response_parts.append(p.text)

            raw_reply = "".join(response_parts) if response_parts else "No response generated."
            final_reply = anonymize_text(raw_reply)
            st.markdown(final_reply)
            st.session_state.messages.append({"role": "assistant", "content": final_reply})
