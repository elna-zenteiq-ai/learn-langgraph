# Sequential Workflow: input ---> llm qa ---> output
# we have two states: question, answer

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
import os 
import requests

load_dotenv()

llm = ChatNVIDIA(
    model="mistralai/mistral-large-3-675b-instruct-2512",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0.2,
    max_completion_tokens=1024
)

class LLMState(TypedDict):
    question: str
    answer: str

def llm_qa(state: LLMState) -> LLMState:
    response = llm.invoke([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": state["question"]}
    ])

    return {
        "question": state["question"],
        "answer": response.content   
    }

graph = StateGraph(LLMState)

graph.add_node('llm_qa', llm_qa)

graph.add_edge(START, 'llm_qa')
graph.add_edge('llm_qa', END)

workflow = graph.compile()

initial_state = {"question": "what is nvidia"}
result = workflow.invoke(initial_state)
print(result)
