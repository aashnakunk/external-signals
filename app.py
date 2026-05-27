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
  .subtitle { color: #8b949e; font-family: monospace; font-size: 0.76rem; margin-bottom: 0.25rem; }

  [data-testid="chatAvatarIcon-user"] svg,
  [data-testid="chatAvatarIcon-user"] img,
  [data-testid="chatAvatarIcon-assistant"] svg,
  [data-testid="chatAvatarIcon-assistant"] img { display: none !important; }

  [data-testid="chatAvatarIcon-user"]::after {
    content: "you"; color: #58a6ff;
    font-family: 'SF Mono', monospace; font-size: 0.75rem; font-weight: 600;
  }
  [data-testid="chatAvatarIcon-assistant"]::after {
    content: "agent"; color: #3fb950;
    font-family: 'SF Mono', monospace; font-size: 0.75rem; font-weight: 600;
  }

  [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stMarkdown p {
    color: #e6edf3; font-family: 'SF Mono', monospace; font-size: 0.87rem;
    background: #161b22; border-left: 2px solid #58a6ff;
    padding: 0.5rem 0.75rem; border-radius: 0 6px 6px 0;
  }

  [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown p {
    color: #cdd9e5; font-size: 0.9rem; line-height: 1.68;
  }

  .meta { color: #484f58; font-family: monospace; font-size: 0.7rem;
          margin-top: -0.4rem; margin-bottom: 0.6rem; }

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

# ── auth ──────────────────────────────────────────────────────────────────────
def _check_password():
    if st.session_state.get("authenticated"):
        return True
    with st.form("auth_form"):
        st.markdown("### External Signals · QuadSci")
        pwd = st.text_input("Password", type="password", placeholder="enter password")
        submitted = st.form_submit_button("Enter", use_container_width=True)
    if submitted:
        correct = st.secrets.get("APP_PASSWORD", "")
        if correct and pwd == correct:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False

if not _check_password():
    st.stop()

# ── header ────────────────────────────────────────────────────────────────────
col_title, col_bin = st.columns([9, 1])
with col_title:
    st.markdown("# External Signals — Test Agent")
    st.markdown('<div class="subtitle">Property of QuadSci</div>', unsafe_allow_html=True)
with col_bin:
    if st.button("🗑️", help="Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_company = None
        st.rerun()

with st.popover("ℹ️  info", use_container_width=False):
    st.markdown(
        "**Search:** Tavily web search API\n\n"
        "- **1 credit** — basic depth, 1 focused query (specific questions)\n"
        "- **2 credits** — advanced depth, 1 query (moderately specific)\n"
        "- **4 credits** — advanced depth, 2 parallel queries (open-ended)\n\n"
        "**Routing:** GPT-5.4 picks the search tier based on your question.\n\n"
        "**Response:** GPT-5.4 synthesizes results through a CS lens.\n\n"
        "_Test agent — public web data only._"
    )

# ── clients ───────────────────────────────────────────────────────────────────
@st.cache_resource
def get_clients():
    tv_key = os.environ.get("TAVILY_API_KEY") or st.secrets.get("TAVILY_API_KEY", "")
    oa_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
    return TavilyClient(api_key=tv_key), OpenAI(api_key=oa_key, max_retries=3)

tavily, llm = get_clients()

LLM_MODEL = "gpt-5.4"
RECENCY_DAYS = 180
TAVILY_TIMEOUT = 30

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
    for attempt in range(3):
        try:
            return tavily.search(query=query, search_depth=depth,
                                 max_results=max_results, include_answer=True,
                                 days=RECENCY_DAYS, timeout=TAVILY_TIMEOUT)
        except Exception:
            if attempt == 2:
                return {"answer": "", "results": []}
            time.sleep(2 ** attempt)  # 1s, 2s

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
    years = _year_window()

    # 1 credit — basic depth, 1 focused query
    if search_mode == "basic" and topic:
        query = f'"{company}" {topic} {years}'
        log.write(f"basic search · {query}")
        t0 = time.time()
        resp = _search(query, "basic", max_results=6)
        elapsed = time.time() - t0
        result = _fmt(resp)
        log.write(f"{len(resp.get('results', []))} results · 1 credit · {elapsed:.1f}s")
        return result or "No results found — search may be temporarily unavailable.", 1

    # 2 credits — advanced depth, 1 query
    if search_mode == "advanced" and topic:
        query = f'"{company}" {topic} {years}'
        log.write(f"advanced search · {query}")
        t0 = time.time()
        resp = _search(query, "advanced", max_results=6)
        elapsed = time.time() - t0
        result = _fmt(resp)
        log.write(f"{len(resp.get('results', []))} results · 2 credits · {elapsed:.1f}s")
        return result or "No results found — search may be temporarily unavailable.", 2

    # 4 credits — advanced depth, 2 parallel queries
    query_set = pick_query_set(growth_context)
    queries = [(lbl, tmpl.format(company=company, years=years)) for lbl, tmpl in query_set.items()]
    log.write(f"broad search · {len(queries)} parallel queries")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = [ex.submit(_search, q, "advanced", 4) for _, q in queries]
        responses = [f.result() for f in futures]
    elapsed = time.time() - t0
    total_results = sum(len(r.get("results", [])) for r in responses)
    log.write(f"{total_results} results · 4 credits · {elapsed:.1f}s")

    sections = [
        f"── {lbl.replace('_', ' ').title()} ──\n{_fmt(r)}"
        for (lbl, _), r in zip(queries, responses)
        if _fmt(r)
    ]
    return "\n\n".join(sections) or "No results found — search may be temporarily unavailable.", 4

# ── LLM calls ────────────────────────────────────────────────────────────────
def extract_company(question, history, log):
    history_txt = "\n".join(
        f"{'User' if m['role']=='user' else 'Agent'}: {m['content'][:300]}"
        for m in history[-4:]
    )
    log.write("identifying company...")
    try:
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
    except Exception:
        company = "Unknown"
    log.write(company)
    return company

def decide(company, question, history, log):
    history_txt = "\n".join(
        f"{'User' if m['role']=='user' else 'Agent'}: {m['content'][:200]}"
        for m in history[-6:]
    )
    log.write("deciding search strategy...")
    try:
        resp = llm.chat.completions.create(
            model=LLM_MODEL, temperature=0,
            messages=[
                {"role": "system", "content": (
                    "You are a routing agent. Given a company, question, and recent conversation, decide:\n"
                    "1. search_mode — pick one:\n"
                    "   'basic': very specific question with a clear topic (e.g. 'did they raise funding?', 'any layoffs?') → 1 credit\n"
                    "   'advanced': moderately specific follow-up or named topic needing depth (e.g. 'tell me about their AI strategy', 'what happened with their CEO?') → 2 credits\n"
                    "   'broad': open-ended, no specific topic (e.g. 'what's happening at X?', 'any news?') → 4 credits\n"
                    "2. topic: for basic/advanced, a short search phrase. Empty if broad.\n"
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
    except Exception:
        d = {"search_mode": "advanced", "topic": "", "growth_context": "Unknown"}
    log.write(f"mode: {d.get('search_mode', 'advanced')} · topic: {d.get('topic') or '—'}")
    return d

def respond(company, question, tool_output, history, log):
    log.write("generating response...")
    history_msgs = [{"role": m["role"], "content": m["content"]} for m in history[-8:]]
    messages = [
        {"role": "system", "content": (
            "You are an external signals agent. Summarize recent news about the company — "
            "financials, funding, layoffs, leadership changes, partnerships, product launches, strategic moves. "
            "Just report what's happening. No advice, no recommendations, no 'what this means for your account', "
            "no actionable insights, no CS framing. "
            "Write in plain conversational paragraphs — no bullet points, no headers, no bold labels. "
            "2-3 paragraphs max. Use specific numbers, dates, names. "
            "Never mention tools, Tavily, or search modes. If a topic has no clear public news, say so briefly."
        )},
    ] + history_msgs + [
        {"role": "user", "content": f"Company: {company}\nQuestion: {question}\n\nWhat you found from your own research:\n{tool_output[:4500]}"},
    ]
    try:
        resp = llm.chat.completions.create(model=LLM_MODEL, temperature=0.3, messages=messages)
        return resp.choices[0].message.content.strip()
    except Exception:
        return "The response couldn't be generated right now — the AI service may be temporarily unavailable. Please try again in a moment."

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

# ── resolve prompt (typed or clicked) ────────────────────────────────────────
if prompt := st.chat_input("ask about any company…"):
    with st.chat_message("user"):
        st.markdown(prompt)

    full_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

    # If the agent just asked "which company?", stitch original question + company clarification together
    effective_prompt = prompt
    if (len(full_history) >= 2
            and full_history[-1]["role"] == "assistant"
            and "which company" in full_history[-1]["content"].lower()):
        effective_prompt = f"{full_history[-2]['content']} (company: {prompt})"

    with st.chat_message("assistant"):
        with st.status("Searching external signals…", expanded=True) as status:
            t0 = time.time()

            company = extract_company(effective_prompt, full_history, status)
            if company == "Unknown" and st.session_state.last_company:
                company = st.session_state.last_company
            elif company != "Unknown":
                st.session_state.last_company = company

            # only pass history relevant to the current company to avoid cross-company confusion
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
                if m.get("company", "").lower() == company.lower()
            ]

            if company == "Unknown":
                status.update(label="Couldn't identify a company", state="error")
                reply = "Which company are you asking about?"
                meta = ""
            else:
                d = decide(company, effective_prompt, history, status)
                mode = d.get("search_mode", "advanced")
                topic = d.get("topic", "")
                gc = d.get("growth_context", "Unknown")

                tool_output, credits = run_search(company, mode, topic, gc, status)
                reply = respond(company, effective_prompt, tool_output, history, status)
                elapsed = time.time() - t0

                credit_label = f"{credits} credit{'s' if credits != 1 else ''}"
                status.update(
                    label=f"✓ {company} · {mode} · {credit_label} · {elapsed:.1f}s",
                    state="complete",
                    expanded=False,
                )
                meta = f"{company} · {mode} · {credit_label} · {elapsed:.1f}s"

        st.markdown(reply)
        if meta:
            st.markdown(f'<div class="meta">{meta}</div>', unsafe_allow_html=True)

    st.session_state.messages.append({"role": "user", "content": prompt, "company": company})
    st.session_state.messages.append({"role": "assistant", "content": reply, "meta": meta, "company": company})
