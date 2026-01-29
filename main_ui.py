



import streamlit as st
import os
import warnings
from typing import Dict, Any

# LangChain & Gemini Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_classic import hub

# 1. Page Configuration
st.set_page_config(page_title="Multi-Agent AI", page_icon="🤖", layout="wide")
warnings.filterwarnings("ignore")

# 2. Secret & Environment Setup
# This line works for both local 'secrets.toml' and Streamlit Cloud 'Secrets'
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("🔑 API Key not found! Please check your .streamlit/secrets.toml file.")
    st.stop()

# 3. Initialize the Brain (LLM)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0
)

# 4. Tools & Agents Setup
search = DuckDuckGoSearchRun()
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
arxiv = ArxivQueryRun()
tools = [search, wiki, arxiv]

# Pull the standard ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")

# Create the Research Agent
research_agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
research_executor = AgentExecutor(
    agent=research_agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True
)

# 5. The Orchestrator Logic
def run_orchestrator(query: str) -> Dict[str, Any]:
    # Agent 1: Research
    research_output = research_executor.invoke({"input": f"Research the topic: {query}"})
    text_data = research_output.get("output", "Search failed.")

    # Agent 2: Summarizer (Chain)
    summary_prompt = PromptTemplate.from_template("Summarize this in 100 words: {text}")
    summary_res = llm.invoke(summary_prompt.format(text=text_data))

    # Agent 3: Email (Chain)
    email_prompt = PromptTemplate.from_template("Write a professional email about: {text}")
    email_res = llm.invoke(email_prompt.format(text=summary_res.content))

    return {
        "research": text_data,
        "summary": summary_res.content,
        "email": email_res.content
    }

# 6. Streamlit User Interface
st.title("🤖 Multi-Agent AI System")
st.caption("One query triggers Research, Summarization, and Email Composition.")

user_query = st.text_input("What would you like to research today?", placeholder="e.g. Advancements in Quantum Computing")

if st.button("🚀 Execute Agents"):
    if not user_query:
        st.warning("Please enter a topic first.")
    else:
        with st.spinner("Agents are collaborating..."):
            try:
                data = run_orchestrator(user_query)
                
                # Layout Results in Tabs
                tab1, tab2, tab3 = st.tabs(["🔍 Research", "📄 Summary", "✉️ Email"])
                
                with tab1:
                    st.markdown("### Detailed Research Results")
                    st.info(data["research"])
                
                with tab2:
                    st.markdown("### Concise Summary")
                    st.success(data["summary"])
                    
                with tab3:
                    st.markdown("### Generated Email Draft")
                    st.code(data["email"], language="markdown")
                    st.button("📋 Copy to Clipboard", on_click=lambda: st.write("Copied! (Simulated)"))

            except Exception as e:
                st.error(f"Execution Error: {str(e)}")