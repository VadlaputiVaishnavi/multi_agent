import requests
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import tool, DuckDuckGoSearchRun
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_classic import hub

# Load environment variables
load_dotenv()

# ------------------ Tools ------------------

search_tool = DuckDuckGoSearchRun()

@tool
def get_place_temperature(city: str) -> str:
    """Fetch current weather for a city"""
    response = requests.get(
        "http://api.weatherstack.com/current",
        params={
            "access_key": os.environ["WEATHERSTACK_API_KEY"],
            "query": city
        },
        timeout=10
    ).json()

    if "current" not in response:
        return f"Weather data not found for {city}"

    current = response["current"]
    return f"{city} weather: {current['temperature']}°C, {current['weather_descriptions'][0]}"

@tool
def calculator(expression: str) -> str:
    """Simple local calculator (no API)"""
    try:
        result = eval(expression, {"__builtins__": {}})
        return f"Result: {result}"
    except Exception:
        return "Invalid mathematical expression"

# ------------------ LLM ------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)
# ------------------ MANUAL ReAct PROMPT (PUT HERE) ------------------

prompt = PromptTemplate.from_template("""
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Observation can repeat)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Question: {input}
{agent_scratchpad}
""")

# ------------------ Agent ------------------

agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_place_temperature, calculator],
    prompt=prompt
)


# ------------------ Agent Setup ------------------



agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_place_temperature, calculator],
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_place_temperature, calculator],
    verbose=True
)

# ------------------ Run ------------------

response = agent_executor.invoke(
    {"input": "Find the capital of India and calculate (3+7)"}
)

print(response["output"])
