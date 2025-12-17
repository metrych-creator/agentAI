from google.adk.agents.llm_agent import Agent
from google.adk.tools.google_search_tool import google_search
from tools.search_knowledge_base import rag_tool
from google.adk.tools.agent_tool import AgentTool
from tools.financial import financial_tool


finance_agent = Agent(
    model='gemini-2.5-flash',
    name='finance_agent',
    description="The PRIMARY tool for checking real-time stock prices, cryptocurrency rates (Bitcoin, ETH), and currency exchange rates.",
    instruction="""
    You are a financial market analyst.
    Your SOLE purpose is to fetch data about stocks, cryptocurrencies, or currencies using the 'fetch_financial_data' tool.
    
    - If user asks about 'stocks' or 'companies', call tool with category='stocks'.
    - If user asks about 'bitcoin', 'ethereum' or 'crypto', call tool with category='crypto'.
    - If user asks about 'dollar', 'euro', 'pln', call tool with category='currencies'.
    
    Do NOT search the web. Use the provided tool ONLY.
    """,
    tools=[financial_tool]
)
finance_agent_tool = AgentTool(agent=finance_agent)

web_agent = Agent(
    model='gemini-2.5-pro',
    name='web_agent',
    description='Useful for searching general news, history, weather, and facts. DO NOT use this for financial market data (stocks, crypto, currencies).', 
    instruction="""
    You are a web researcher. Use Google Search to find public information.

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
    description="Useful in searching infromation about book 'Romeo and Juliet'.",
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
    You are the Planner. Your goal is to route user queries to the correct specialists. 
    You are capable of planning multi-step actions if a query requires data from more than one source.

    STRATEGIC REASONING RULES:
    1. ANALYZE: Determine if the query needs one or multiple pieces of information (e.g., "BTC price in PLN" needs Crypto price AND Currency rate).
    2. SEQUENTIAL CALLING: If multiple tools are needed, call them one by one. Use the output of the first tool to refine the second call.
    3. FALLBACK: If a specialized tool (Finance/RAG) returns a "cookie consent", "access denied", "empty result", or "not found", you MUST immediately call 'web_agent_tool' as a backup.
    4. VERIFICATION: If the result from Finance or RAG is incomplete, always supplement it with 'web_agent_tool'.

    ROUTING PRIORITIES:
    - FINANCIAL DATA (priority #1): 
        - Stocks, Cryptocurrencies, Forex/Exchange Rates.
        -> Tool: finance_agent_tool
    - INTERNAL DOCS (priority #2): 
        - Company-specific data, PDFs, internal knowledge.
        -> Tool: rag_agent_tool (If unsure, try this first).
    - GENERAL WEB (priority #3 / Fallback): 
        - General world knowledge, news, OR when priority #1 and #2 fail to give a clear answer.
        -> Tool: web_agent_tool

    EXAMPLES of MULTI-STEP / FALLBACK:
    - User: "BTC in PLN" -> Step 1: finance_agent(crypto) -> Step 2: finance_agent(currencies).
    - User: "Apple stock price" -> Finance returns "Cookie Wall" -> Step 2: web_agent_tool.
    - User: "What is our policy on Bitcoin?" -> Step 1: rag_agent_tool -> If not found -> Step 2: finance_agent_tool.

    Do not answer questions yourself. Use the tools.
    If none of the tools found a result after trying all relevant steps, say so.
    """,
    tools=[web_agent_tool, rag_agent_tool, finance_agent_tool] 
)