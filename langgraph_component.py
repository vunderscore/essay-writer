import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from tavily import TavilyClient
from prompts import PLAN_PROMPT, WRITER_PROMPT, REFLECTION_PROMPT, RESEARCH_PLAN_PROMPT, RESEARCH_CRITIQUE_PROMPT            


load_dotenv()

#calling the apis
model = ChatGoogleGenerativeAI(model='gemini-2.0-flash-001',google_api_key=os.environ["GOOGLE_API_KEY"])
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

#agent state class
class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int

#query format
class Queries(BaseModel):
    queries: List[str]

#creating the nodes
def plan_node(state: AgentState):
    messages = [
        SystemMessage(content=PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ]
    response = model.invoke(messages)
    return {"plan": response.content}

def research_plan_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ])
    content = state.get('content', [])

    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}

def generation_node(state: AgentState):
    content = "\n\n".join(state['content'] or [])
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}")
    messages = [
        SystemMessage(
            content=WRITER_PROMPT.format(content=content)
        ),
        user_message
        ]
    response = model.invoke(messages)
    return {
        "draft": response.content, 
        "revision_number": state.get("revision_number", 1) + 1
    }

def reflection_node(state: AgentState):
    messages = [
        SystemMessage(content=REFLECTION_PROMPT), 
        HumanMessage(content=state['draft'])
    ]
    response = model.invoke(messages)
    return {"critique": response.content}

def research_critique_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content=state['critique'])
    ])
    content = state['content'] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}


#condtional statement
def should_continue(state):
    if state["revision_number"] > state["max_revisions"]:
        return END
    else:
        return "reflect"

#building the graph
builder = StateGraph(AgentState)

builder.add_node("planner", plan_node)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.add_node("research_plan", research_plan_node)
builder.add_node("research_critique", research_critique_node)

builder.set_entry_point("planner")

builder.add_conditional_edges(
    "generate",
    should_continue,
    {END:END, "reflect":"reflect"}
)

builder.add_edge("planner", "research_plan")
builder.add_edge("research_plan", "generate")
builder.add_edge("reflect", "research_critique")
builder.add_edge("research_critique", "generate")

#compiling graph
graph = builder.compile()

# invoker = {"task": "what is the difference between chennai and mumbai",
#             "max_revisions":2,
#             "revision_number":1}

# result = graph.invoke(invoker)
# print(result['draft'])

def graph_output(query: str):
    invoker = {"task":query, "max_revisions":2,"revision_number":1}
    result = graph.invoke(invoker)
    return result['draft']

print(graph_output("diversity in india"))
