# prompt chaining: topic-->llm outline-->llm blog for this topic and outline-->blog
# states: topic, outline, blog
# why is this a prompt chaining? : Because there are two nodes here, and we are interacting with llm here.

from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from typing import TypedDict
import os

load_dotenv()

#define model
llm = ChatNVIDIA(
    model = "mistralai/mistral-large-3-675b-instruct-2512",
    api_key = os.getenv("NVIDIA_API_KEY"),
    temperature=0.2,
    max_completion_tokens=1024
)

#define state
class LLMState(TypedDict):
    topic: str
    outline: str
    blog: str

#define functions
def gen_outline(state: LLMState) -> LLMState:

    messages = [{"role": "system", "content": "Generate an outline for a blog"},
    {"role": "user", "content": state['topic']}
    ]

    response = llm.invoke(messages).content

    state["outline"] = response
    return state

def gen_blog(state: LLMState) -> LLMState:

    messages = [{"role": "system", "content": "Generate a blog for the topic"},
    {"role": "user", "content": state['outline']}
    ]
    
    response = llm.invoke(messages).content

    state["blog"] = response
    return state

#define graph
graph = StateGraph(LLMState)

#add node
graph.add_node('gen_outline', gen_outline)
graph.add_node('gen_blog', gen_blog)

#add edge
graph.add_edge(START,'gen_outline')
graph.add_edge('gen_outline','gen_blog')
graph.add_edge('gen_blog',END)

#compile graph
workflow = graph.compile()

initial_state = {"topic": "kerala"}
final_state = workflow.invoke(initial_state)
print(final_state)