import os
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

load_dotenv("/Users/aashnakunkolienker/projects/fastapi-qchat-quadsci/.env")

from openai import OpenAI
from tavily import TavilyClient

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="External Signals · QuadSci",
    page_icon="🔍",
    layout="centered",
)

st.markdown("""
<style>
  .stApp { background-color: #0d1117; }
  .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 780px; }

  h1 { color: #58a6ff !important; font-family: 'SF Mono', monospace !important;
       font-size: 1.15rem !important; letter-spacing: 0.04em; margin-bottom: 0 !important; }
  .subtitle { color: #8b949e; font-family: monospace; font-size: 0.76rem; margin-bottom: 1.5rem; }

  /* user bubble */
  [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stMarkdown p {
    color: #e6edf3; font-family: 'SF Mono', monospace; font-size: 0.87rem;
    background: #161b22; border-left: 2px solid #58a6ff;
    padding: 0.5rem 0.75rem; border-radius: 0 6px 6px 0;
  }

  /* agent bubble */
  [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown p {
    color: #cdd9e5; font-size: 0.9rem; line-height: 1.68;
  }

  /* meta line */
  .meta { color: #484f58; font-family: monospace; font-size: 0.7rem;
          margin-top: -0.4rem; margin-bottom: 0.6rem; }

  /* status/log box */
  [data-testid="stStatus"] { background: #161b22 !important; border: 1px solid #21262d !important; }
  [data-testid="stStatus"] p, [data-testid="stStatus"] li {
    font-family: 'SF Mono', monospace !important; font-size: 0.75rem !important; color: #8b949e !important;
  }

  .stChatInputContainer { background: #161b22 !important; border-top: 1px solid #21262d; }
  .stChatInputContainer textarea {
    background: #161b22 !important; color: #e6edf3 !important;
    font-family: 'SF Mono', monospace !important; font-size: 0.87rem !important;
    border: 1px solid #30363d !important; border-radius: 6px !important;
  }
  .stChatInputContainer textarea::placeholder { color: #484f58 !important; }
  #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── header ────────────────────────────────────────────────────────────────────
st.markdown("# External Signals — Test Agent")
st.markdown(
    '<div class="subtitle">Property of QuadSci</div>',
    unsafe_allow_html=True,
)
with st.expander("ℹ️ Info"):
    st.markdown(
        """
**Model stack**
- **Search:** Tavily web search API — basic depth (1 credit) for specific questions, advanced depth (2 credits/query) for open-ended research
- **Routing:** GPT-5.4 decides whether to run a focused or broad search based on your question
- **Response:** GPT-5.4 synthesizes search results into a conversational answer

**How it works**
Each question goes through three steps: identify the company → search the web for recent signals → generate a response. Conversation history is kept so follow-ups work naturally.

**This is a test agent.** Results are sourced from public web data via Tavily and may not be exhaustive.
        """
    )

# ── clients ───────────────────────────────────────────────────────────────────
@st.cache_resource
def get_clients():
    tv_key = os.environ.get("TAVILY_API_KEY") or st.secrets.get("TAVILY_API_KEY", "")
    oa_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
    return TavilyClient(api_key=tv_key), OpenAI(api_key=oa_key)

tavily, llm = get_clients()

LLM_MODEL = "gpt-5.4"
RECENCY_DAYS = 180

# ── query templates ───────────────────────────────────────────────────────────
def _year_window():
    y = datetime.now().year
    return f"{y - 1} {y}"

BROAD_QUERIES_POSITIVE = {
    "growth_momentum": '"{company}" revenue growth funding expansion new customers product launch partnership acquisition wins {years}',
    "leadership_strategy": '"{company}" CEO executive leadership strategy innovation market expansion hiring investment {years}',
}
BROAD_QUERIES_NEGATIVE = {
    "financial_risk": '"{company}" layoffs restructuring debt downgrade revenue decline cost cutting financial challenges losses {years}',
    "churn_signals": '"{company}" customer churn contract loss market share declining leadership turnover workforce reduction reorganization {years}',
}
BROAD_QUERIES_NEUTRAL = {
    "financial_workforce": '"{company}" earnings revenue financial performance layoffs workforce restructuring acquisition merger {years}',
    "leadership_strategy": '"{company}" CEO executive leadership change strategy partnership product launch regulatory {years}',
}

def pick_query_set(growth_context):
    gc = (growth_context or "").lower()
    if any(k in gc for k in ("churn", "contraction", "decline")):
        return BROAD_QUERIES_NEGATIVE
    if any(k in gc for k in ("high growth", "med growth", "medium growth")):
        return BROAD_QUERIES_POSITIVE
    return BROAD_QUERIES_NEUTRAL

# ── Tavily ────────────────────────────────────────────────────────────────────
def _search(query, depth, max_results=5):
    return tavily.search(query=query, search_depth=depth,
                         max_results=max_results, include_answer=True, days=RECENCY_DAYS)

def _fmt(resp):
    parts = []
    if resp.get("answer"):
        parts.append(resp["answer"])
    seen = set()
    for r in resp.get("results", []):
        url = r.get("url", "")
        if url in seen:
            continue
        seen.add(url)
        title = r.get("title", "").strip()
        content = re.sub(r"Logo for [^\n]+", "", r.get("content", ""))
        content = re.sub(r"\n{3,}", "\n\n", content).strip()[:700]
        if content:
            parts.append(f"Source: {title} ({url})\n{content}")
    return "\n\n".join(parts)

def run_search(company, search_mode, topic, growth_context, log):
    """
    Returns (tool_output, total_credits, log_lines).
    basic  = 1 credit/query  (targeted mode, 1 query)
    advanced = 2 credits/query (broad mode, 2 parallel queries)
    """
    years = _year_window()
    total_credits = 0

    if search_mode == "basic" and topic:
        query = f'"{company}" {topic} {years}'
        log.write(f"🔍 **Basic search** (1 credit)\n`{query}`")
        t0 = time.time()
        resp = _search(query, "basic", max_results=6)
        elapsed = time.time() - t0
        n = len(resp.get("results", []))
        total_credits = 1
        log.write(f"↳ {n} results in {elapsed:.1f}s · 1 credit used")
        return _fmt(resp), total_credits

    else:
        query_set = pick_query_set(growth_context)
        queries = [(lbl, tmpl.format(company=company, years=years)) for lbl, tmpl in query_set.items()]
        log.write(f"🔍 **Advanced search** (2 credits/query · {len(queries)} parallel queries)")
        for lbl, q in queries:
            log.write(f"`{lbl}:` {q}")
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=2) as ex:
            futures = [ex.submit(_search, q, "advanced", 4) for _, q in queries]
            responses = [f.result() for f in futures]
        elapsed = time.time() - t0
        total_credits = len(queries) * 2
        total_results = sum(len(r.get("results", [])) for r in responses)
        log.write(f"↳ {total_results} results across {len(queries)} queries in {elapsed:.1f}s · {total_credits} credits used")

        sections = []
        for (lbl, _), r in zip(queries, responses):
            fmt = _fmt(r)
            if fmt:
                sections.append(f"── {lbl.replace('_', ' ').title()} ──\n{fmt}")
        return "\n\n".join(sections), total_credits

# ── LLM calls ────────────────────────────────────────────────────────────────
def decide(company, question, history, log):
    history_txt = "\n".join(
        f"{'User' if m['role']=='user' else 'Agent'}: {m['content'][:200]}"
        for m in history[-6:]
    )
    log.write("🤖 Deciding search strategy...")
    resp = llm.chat.completions.create(
        model=LLM_MODEL, temperature=0,
        messages=[
            {"role": "system", "content": (
                "You are a routing agent. Given a company, question, and recent conversation, decide:\n"
                "1. search_mode: 'basic' (specific topic) or 'advanced' (open-ended, broad)\n"
                "2. topic: if basic, a short search phrase. Empty if advanced.\n"
                "3. growth_context: 'High Growth', 'Medium Growth', 'Stable', 'Contraction', or 'Unknown'.\n"
                "Respond in JSON only: {\"search_mode\": ..., \"topic\": \"...\", \"growth_context\": \"...\"}"
            )},
            {"role": "user", "content": f"Company: {company}\nQuestion: {question}\n\nRecent conversation:\n{history_txt}"},
        ],
    )
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", resp.choices[0].message.content.strip())
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        d = {"search_mode": "advanced", "topic": "", "growth_context": "Unknown"}
    mode = d.get("search_mode", "advanced")
    topic = d.get("topic", "")
    gc = d.get("growth_context", "Unknown")
    log.write(f"↳ mode: **{mode}** · topic: `{topic or '—'}` · growth context: {gc}")
    return d

def extract_company(question, history, log):
    history_txt = "\n".join(
        f"{'User' if m['role']=='user' else 'Agent'}: {m['content'][:300]}"
        for m in history[-4:]
    )
    log.write("🏢 Identifying company...")
    resp = llm.chat.completions.create(
        model=LLM_MODEL, temperature=0,
        messages=[
            {"role": "system", "content": (
                "Extract the company the user is asking about. "
                "If it's a follow-up with no new company named, infer from conversation. "
                "Reply with ONLY the company name. If truly unknown reply 'Unknown'."
            )},
            {"role": "user", "content": f"Question: {question}\n\nConversation:\n{history_txt}"},
        ],
    )
    company = resp.choices[0].message.content.strip()
    log.write(f"↳ {company}")
    return company

def respond(company, question, tool_output, history, log):
    log.write("✍️ Generating response...")
    history_msgs = [{"role": m["role"], "content": m["content"]} for m in history[-8:]]
    messages = [
        {"role": "system", "content": (
            "You are a conversational research agent. Write like a sharp colleague texting a quick briefing — "
            "plain paragraphs only, no bullet points, no headers, no bold labels, no lists. "
            "2-3 short paragraphs max. Lead with the single most interesting finding. "
            "Use specific numbers, dates, names. Reference prior conversation for follow-ups. "
            "Do not mention tools, search modes, or Tavily. "
            "If data is thin, say so in one sentence and share what you did find."
        )},
    ] + history_msgs + [
        {"role": "user", "content": f"[Company: {company}]\n[Question: {question}]\n\nExternal signals:\n{tool_output[:4500]}"},
    ]
    resp = llm.chat.completions.create(model=LLM_MODEL, temperature=0.3, messages=messages)
    return resp.choices[0].message.content.strip()

# ── session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_company" not in st.session_state:
    st.session_state.last_company = None

# ── render history ────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            st.markdown(f'<div class="meta">{msg["meta"]}</div>', unsafe_allow_html=True)

# ── chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("ask about any company…"):
    with st.chat_message("user"):
        st.markdown(prompt)

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

    with st.chat_message("assistant"):
        with st.status("Searching external signals…", expanded=True) as status:
            t0 = time.time()

            company = extract_company(prompt, history, status)

            if company == "Unknown" and st.session_state.last_company:
                company = st.session_state.last_company
            elif company != "Unknown":
                st.session_state.last_company = company

            if company == "Unknown":
                status.update(label="Couldn't identify a company", state="error")
                reply = "Which company are you asking about?"
                meta = ""
            else:
                d = decide(company, prompt, history, status)
                mode = d.get("search_mode", "advanced")
                topic = d.get("topic", "")
                gc = d.get("growth_context", "Unknown")

                tool_output, credits = run_search(company, mode, topic, gc, status)

                reply = respond(company, prompt, tool_output, history, status)
                elapsed = time.time() - t0

                credit_label = f"{credits} credit{'s' if credits != 1 else ''}"
                status.update(
                    label=f"✓ {company} · {mode} search · {credit_label} · {elapsed:.1f}s",
                    state="complete",
                    expanded=False,
                )
                meta = f"{company} · {mode} · {credit_label} · {elapsed:.1f}s"

        st.markdown(reply)
        if meta:
            st.markdown(f'<div class="meta">{meta}</div>', unsafe_allow_html=True)

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": reply, "meta": meta})
