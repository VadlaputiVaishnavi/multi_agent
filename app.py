import os
from typing import TypedDict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    query: str
    research: str
    critique: str
    email: str
    logs: List[str]

class MultiAgentSystem:
    def __init__(self):
        # Key is pulled from .env file for security
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3
        )
        # Search Tools
        self.search = DuckDuckGoSearchRun()
        self.wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    def research_agent(self, state: AgentState):
        """Phase 1: Real-time Web Research"""
        query = state['query']
        
        # Perform actual web search
        search_results = self.search.run(query)
        wiki_results = self.wiki.run(query)
        
        combined_data = f"Web Search: {search_results}\nWikipedia: {wiki_results}"
        
        sys_msg = SystemMessage(content="You are a Senior Research Analyst. Synthesize the provided search data into a structured report.")
        res = self.llm.invoke([sys_msg, HumanMessage(content=f"Topic: {query}\nSource Data: {combined_data}")])
        
        state['research'] = res.content
        state['logs'].append("✅ Research Agent: Web search and analysis complete.")
        return state

    def critic_agent(self, state: AgentState):
        """Phase 2: Quality Control"""
        sys_msg = SystemMessage(content="You are a Lead Editor. Review the research for gaps, bias, or missing details.")
        res = self.llm.invoke([sys_msg, HumanMessage(content=f"Review this research: {state['research']}")])
        state['critique'] = res.content
        state['logs'].append("⚖️ Critic Agent: Accuracy audit finalized.")
        return state

    def email_agent(self, state: AgentState):
        """Phase 3: Final Output"""
        sys_msg = SystemMessage(content="You are a Communications Expert. Draft a professional email based on the research and critique.")
        prompt = f"Research: {state['research']}\nCritique: {state['critique']}"
        res = self.llm.invoke([sys_msg, HumanMessage(content=prompt)])
        state['email'] = res.content
        state['logs'].append("📧 Email Agent: Final draft prepared.")
        return state