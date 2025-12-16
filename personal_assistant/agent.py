from google.adk.agents.llm_agent import Agent
from google.adk.tools.google_search_tool import google_search
from tools.search_knowledge_base import rag_tool
from google.adk.tools.agent_tool import AgentTool



web_agent = Agent(
    model='gemini-2.5-pro',
    name='web_agent',
    description='A helpful assistant that can search the web and internal documents.',
    instruction="""
    You are a web researcher. Use Google Search to find public information.",

    CRITICAL RULES:
    1. To find information, you MUST use the 'google_search' tool.
    2. DO NOT write Python code (e.g. do not write "print(google_search...)").
    3. Simply call the tool with the user's query.
    """,
    tools = [google_search]
)

rag_agent = Agent(
    model='gemini-2.5-flash',
    name='rag_agent',
    instruction="""
    You are a document analyst. Search internal PDFs to answer questions, use the 'rag_tool' tool.

    If you don't know the answer, just say so.
    """,
    tools=[rag_tool]
    )

web_agent_tool = AgentTool(agent=web_agent)
rag_agent_tool = AgentTool(agent=rag_agent)


root_agent = Agent(
    model='gemini-2.5-flash',
    name='manager',
    instruction="""
    You are the Planner. Your goal is to route user queries to the correct specialist.

    - If question is about internal docs -> call 'rag_agent_tool'.
    - If question is about world/news -> call 'web_agent_tool'.
    - If you are not sure which tool to use, then firstly use rag_agent_tool. If 'rag_agent_tool' returns "NOT_FOUND_LOCALLY", then call 'web_agent_tool'.

    Do not answer questions yourself. Use the tools.
    If none of the tools found a result, say so.
    """,
    tools=[web_agent_tool, rag_agent_tool] 
)
