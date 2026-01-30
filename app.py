import os
import time
from typing import TypedDict, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError


class AgentState(TypedDict):
    query: str
    research: str
    critique: str
    email: str
    logs: List[str]


class MultiAgentSystem:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY missing in Streamlit Secrets")

        # ✅ USE STABLE MODEL
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.3,
        )

        try:
            self.search = DuckDuckGoSearchRun()
        except Exception:
            self.search = None

        self.wiki = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=3)
        )

    # ---------- SAFE INVOKE ----------
    def safe_invoke(self, messages, state, agent_name):
        try:
            return self.llm.invoke(messages).content
        except ChatGoogleGenerativeAIError as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                state["logs"].append(
                    f"🚫 {agent_name}: Gemini quota exceeded. Try later."
                )
                return "⚠️ Gemini API quota exceeded. Please try again later."
            else:
                state["logs"].append(
                    f"❌ {agent_name}: Unexpected LLM error."
                )
                return "⚠️ An unexpected AI error occurred."

    # ---------- AGENTS ----------
    def research_agent(self, state: AgentState):
        query = state["query"]

        web_data = ""
        if self.search:
            try:
                web_data = self.search.run(query)
            except Exception:
                web_data = "Web search unavailable."

        wiki_data = self.wiki.run(query)

        prompt = f"""
        Topic: {query}

        Web:
        {web_data}

        Wikipedia:
        {wiki_data}
        """

        sys_msg = SystemMessage(
            content="You are a senior research analyst. Summarize clearly."
        )

        state["research"] = self.safe_invoke(
            [sys_msg, HumanMessage(content=prompt)],
            state,
            "Research Agent",
        )

        state["logs"].append("✅ Research completed")
        return state

    def critic_agent(self, state: AgentState):
        if "quota exceeded" in state["research"].lower():
            state["critique"] = state["research"]
            return state

        sys_msg = SystemMessage(
            content="You are a lead editor. Improve clarity and correctness."
        )

        state["critique"] = self.safe_invoke(
            [sys_msg, HumanMessage(content=state["research"])],
            state,
            "Critic Agent",
        )

        state["logs"].append("⚖️ Critique completed")
        return state

    def email_agent(self, state: AgentState):
        if "quota exceeded" in state["critique"].lower():
            state["email"] = state["critique"]
            return state

        sys_msg = SystemMessage(
            content="You are a professional email writer."
        )

        prompt = f"""
        Research:
        {state["research"]}

        Critique:
        {state["critique"]}

        Write a professional email.
        """

        state["email"] = self.safe_invoke(
            [sys_msg, HumanMessage(content=prompt)],
            state,
            "Email Agent",
        )

        state["logs"].append("📧 Email generated")
        return state
