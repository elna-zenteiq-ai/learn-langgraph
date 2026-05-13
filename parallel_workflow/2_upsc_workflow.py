# parellel workflow: topic-->(clarity of thought, depth of analysis, language) --> evaluation
# llm will give text feedback, and score between 0-10
# evaluation will summarise the text feedback from the three llms, and also calculate an avg of the scores and give final score
# states: essay, cot_text, cot_score, doa_text, doa_score, lang_text, lang_score, summary, overall_score
# we have to ensure that these workflows will give the answer in the format we need
# for that we need to make use of structured output
# the scores from three llms can be stored as a list
# we also need to use reducer function as well as we need to merge these values into the list

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Annotated
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from pydantic import BaseModel, Field
import os
import json
import operator

load_dotenv()

llm = ChatNVIDIA(
    model = "nvidia/nemotron-mini-4b-instruct",
    api_key = os.getenv("NVIDIA_API_KEY"),
    temperature = 0.3,
    max_completion_tokens = 300
)

essay = """
Happiness is a deeply personal and often misunderstood aspect of human life. Many people associate it with success, wealth, or external achievements, but true happiness goes far beyond material possessions. It is a state of mind that arises from contentment, meaningful relationships, and a sense of purpose.

At its core, happiness is about appreciating the present moment. In a fast-paced world, individuals often chase future goals, believing that happiness lies just ahead. However, this constant pursuit can lead to dissatisfaction, as the mind becomes focused on what is lacking rather than what is already present. Learning to be grateful for small joys—such as a kind gesture, a peaceful morning, or time spent with loved ones—can greatly enhance one’s sense of well-being.

Moreover, happiness is closely linked to emotional balance. It does not mean the absence of difficulties or negative emotions, but rather the ability to navigate them with resilience. People who cultivate positive thinking and self-awareness are better equipped to handle challenges while maintaining inner peace.

Ultimately, happiness is not something to be found externally, but something to be nurtured within. By valuing simplicity, building meaningful connections, and practicing gratitude, individuals can create a lasting sense of fulfillment in their lives.
"""

class EvalResponse(BaseModel):
    feedback: str = Field(description = 'detailed feedback for the essay')
    score: int = Field(description = 'score out of 10', ge=0, le=10)

essay = """
Happiness is a deeply personal and often misunderstood aspect of human life. Many people associate it with success, wealth, or external achievements, but true happiness goes far beyond material possessions. It is a state of mind that arises from contentment, meaningful relationships, and a sense of purpose.

At its core, happiness is about appreciating the present moment. In a fast-paced world, individuals often chase future goals, believing that happiness lies just ahead. However, this constant pursuit can lead to dissatisfaction, as the mind becomes focused on what is lacking rather than what is already present. Learning to be grateful for small joys—such as a kind gesture, a peaceful morning, or time spent with loved ones—can greatly enhance one’s sense of well-being.

Moreover, happiness is closely linked to emotional balance. It does not mean the absence of difficulties or negative emotions, but rather the ability to navigate them with resilience. People who cultivate positive thinking and self-awareness are better equipped to handle challenges while maintaining inner peace.

Ultimately, happiness is not something to be found externally, but something to be nurtured within. By valuing simplicity, building meaningful connections, and practicing gratitude, individuals can create a lasting sense of fulfillment in their lives.
"""

prompt = f"""
Evaluate the following essay.

Return ONLY valid JSON in this exact format:
{{
  "feedback": "detailed feedback",
  "score": number between 0 and 10
}}

Do not include any extra text.

Essay:
{essay}
"""

response = llm.invoke(prompt)
data = json.loads(response.content)


class UPSCState(TypedDict):
    essay: str
    cot_text: str
    lang_text: str
    doa_text: str
    overall_text: str
    scores: Annotated[List[int], operator.add] # we have to add the ind. scores into list without being overwritten, so we need the add reducer function
    overall_score: int

def clarity_of_thought(state: UPSCState):
    prompt = f"""
    Evaluate the clarity of thought of the following essay.

    Return ONLY valid JSON in this exact format:
    {{
    "feedback": "detailed feedback",
    "score": number between 0 and 10
    }}

    Do not include any extra text.

    Essay:
    {state['essay']}
    """
    response = llm.invoke(prompt)
    data = json.loads(response.content)
    return {'cot_text': data['feedback'], 'scores': [data['score']]}

def language_analysis(state: UPSCState):
    prompt = f"""
    Evaluate the language quality of the following essay.

    Return ONLY valid JSON in this exact format:
    {{
    "feedback": "detailed feedback",
    "score": number between 0 and 10
    }}

    Do not include any extra text.

    Essay:
    {state['essay']}
    """
    response = llm.invoke(prompt)
    data = json.loads(response.content)
    return {'lang_text': data['feedback'], 'scores': [data['score']]}

def doa(state: UPSCState):
    prompt = f"""
    Evaluate the depth of analysis of the following essay.

    Return ONLY valid JSON in this exact format:
    {{
    "feedback": "detailed feedback",
    "score": number between 0 and 10
    }}

    Do not include any extra text.

    Essay:
    {state['essay']}
    """
    response = llm.invoke(prompt)
    data = json.loads(response.content)
    return {'doa_text': data['feedback'], 'scores': [data['score']]}


def evaluate(state:UPSCState):
    prompt = f"""
    Based on the following feedback give a summary feedback.

    Return ONLY valid JSON in this exact format:
    {{
        "feedback": "summary of all feedbacks"
    }}

    Do not include any extra text.

    Language Feedback: {state['lang_text']}
    Depth of Analysis Feedback: {state['doa_text']}
    Clarity of Thought Feedback: {state['cot_text']}
    """
    response = llm.invoke(prompt)
    data = json.loads(response.content)
    avg = sum(state['scores'])/len(state['scores'])
    return {'overall_text': data['feedback'], 'overall_score': avg}


graph = StateGraph(UPSCState)

graph.add_node('clarity_of_thought', clarity_of_thought)
graph.add_node('language_analysis', language_analysis)
graph.add_node('doa', doa)
graph.add_node('evaluate', evaluate)

graph.add_edge(START, 'clarity_of_thought')
graph.add_edge(START, 'language_analysis')
graph.add_edge(START, 'doa')

graph.add_edge('clarity_of_thought', 'evaluate')
graph.add_edge('language_analysis', 'evaluate')
graph.add_edge('doa', 'evaluate')

graph.add_edge('evaluate', END)

workflow = graph.compile()

initial_state = {
    "essay": essay
}

final_state = workflow.invoke(initial_state)
print(final_state)