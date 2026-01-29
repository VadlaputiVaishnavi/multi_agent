import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from typing import Dict, Any
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import (
    DuckDuckGoSearchRun,
    WikipediaQueryRun,
    ArxivQueryRun
)
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic import hub
from langchain_core.prompts import PromptTemplate

load_dotenv()

# ---------------- LLM ----------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# ---------------- TOOLS ----------------
search = DuckDuckGoSearchRun()
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
arxiv = ArxivQueryRun()
tools = [search, wiki, arxiv]

# ---------------- RESEARCH AGENT ----------------
react_prompt = hub.pull("hwchase17/react")

research_agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt
)

research_executor = AgentExecutor(
    agent=research_agent,
    tools=tools,
    verbose=True
)

# ---------------- SUMMARY PROMPT ----------------
SUMMARY_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""
Summarize the following content in 100–150 words.
Be clear, factual, and structured.

{text}
"""
)

# ---------------- EMAIL PROMPT ----------------
EMAIL_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""
Write a professional email based on the following summary.
Tone: clear, polite, and formal.

{text}
"""
)

# ---------------- CRITIC PROMPT ----------------
CRITIC_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""
Act as a critic and fact-checker.

Tasks:
1. Verify factual consistency
2. Highlight key insights


Content:
{text}

Return output in sections:
- Fact Check
- Insights
- Titles
- Sources
"""
)

# ---------------- ORCHESTRATOR ----------------
def run_orchestrator(query: str) -> Dict[str, Any]:

    research = research_executor.invoke({
        "input": f"Research the topic: {query}"
    })

    research_text = research.get("output", research)

    summary = llm.invoke(
        SUMMARY_PROMPT.format(text=research_text)
    )

    email = llm.invoke(
        EMAIL_PROMPT.format(text=summary.content)
    )

    critic = llm.invoke(
        CRITIC_PROMPT.format(text=research_text)
    )

    return {
        "research": research_text,
        "summary": summary.content,
        "email": email.content,
        "critic": critic.content
    }