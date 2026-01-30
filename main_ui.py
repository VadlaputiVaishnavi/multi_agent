



import streamlit as st
import os
import requests
import warnings
from typing import Dict, Any

# Corrected Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
# Note: Ensure you are using the correct agent imports for your version
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_classic import hub
from streamlit_lottie import st_lottie  # Fixed typo here

# 1. Page Configuration
st.set_page_config(page_title="Multi-Agent AI", page_icon="🤖", layout="wide")
warnings.filterwarnings("ignore")

# 2. Secret & Environment Setup
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("🔑 API Key not found! Add it to Streamlit Secrets.")
    st.stop()

# 3. Persistent Data Initialization
if "results" not in st.session_state:
    st.session_state.results = None
if "user_query" not in st.session_state:
    st.session_state.user_query = ""

# 4. Initialize LLM & Tools
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

@st.cache_resource
def setup_agent():
    search = DuckDuckGoSearchRun()
    wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    arxiv = ArxivQueryRun()
    tools = [search, wiki, arxiv]
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

executor = setup_agent()

# 5. Helper for Animations
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except: return None

lottie_ai = load_lottieurl("https://lottie.host/8040f796-031e-4509-9528-7634f1a238b1/mXU5P96H3o.json")

# --- UI LAYOUT ---
st.title("🤖 Multi-Agent AI System")

# Hero Section
if not st.session_state.results:
    if lottie_ai: st_lottie(lottie_ai, height=250)
    
    # Hidden label for accessibility
    query_input = st.text_input("Search Input", placeholder="What should your agents research?", label_visibility="collapsed")
    
    if st.button("🚀 Launch Agents", use_container_width=True) and query_input:
        st.session_state.user_query = query_input
        with st.status("Agents working...", expanded=True) as status:
            try:
                # Step 1: Research
                res = executor.invoke({"input": f"Research: {query_input}"})
                res_text = res.get("output", "No data found.")
                
                # Step 2: Synthesis
                summary = llm.invoke(f"Summarize this in 100 words: {res_text}").content
                email = llm.invoke(f"Draft a professional email based on: {summary}").content
                
                # Save to session state
                st.session_state.results = {
                    "research": res_text,
                    "summary": summary,
                    "email": email
                }
                status.update(label="✅ Analysis Complete", state="complete")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

# Result Section (Persists because of session_state)
else:
    st.markdown(f"### 📋 Results for: *{st.session_state.user_query}*")
    tab1, tab2, tab3 = st.tabs(["🔍 Research", "📄 Summary", "✉️ Email"])
    
    with tab1: st.info(st.session_state.results["research"])
    with tab2: st.success(st.session_state.results["summary"])
    with tab3: st.code(st.session_state.results["email"], language="markdown")
    
    if st.button("🔄 New Research"):
        st.session_state.results = None
        st.rerun()