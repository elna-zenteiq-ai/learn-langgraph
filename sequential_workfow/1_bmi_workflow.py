# Sequential Workflow: input ---> calculate bmi ---> output
# we have three states: height, weight, bmi

from langgraph.graph import StateGraph, START, END
from IPython.display import Image
from typing import TypedDict

#define state
class BMIState(TypedDict):
    height: float
    weight: float
    bmi: float
    category: str

#calculate bmi
def calculate_bmi(state: BMIState) -> BMIState:

    weight = state['weight']
    height = state['height']

    bmi = weight/(height**2)

    state['bmi'] =round(bmi,2)

    return state

def category_bmi(state:BMIState) -> BMIState:

    bmi = state['bmi']

    if bmi < 18.5:
        state["category"] = 'underweight'
    elif 18.5<= bmi < 25:
        state["category"] = 'normal'
    elif 25<= bmi < 30:
        state["category"] = 'overweight'
    else:
        state["category"] = 'obese'
    return state

#define graph
graph = StateGraph(BMIState)

#add nodes to graph
graph.add_node('calculate_bmi', calculate_bmi)
graph.add_node('category_bmi', category_bmi)

#add edges to graph
graph.add_edge(START, 'calculate_bmi')
graph.add_edge('calculate_bmi', 'category_bmi')
graph.add_edge('category_bmi', END)

#compile graph
workflow = graph.compile()

#execute graph
initial_state = {'weight': 50, 'height': 1.56 }
final_state = workflow.invoke(initial_state)
print(final_state)

#view graph
Image(workflow.get_graph().draw_mermaid_png())

# Sequential Workflow: input ---> calculate bmi ---> label bmi ---> output
# we have three states: height, weight, category, bmi

