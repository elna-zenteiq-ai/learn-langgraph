#parellel workflow: we will calculate boundray%, balls per boundary, strike rate parellely, and then get a summary on them
#state: runs, balls, 4s, 6s, sr, bp, bpb, summary

from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class CricketState(TypedDict):
    runs:   int
    balls:  int
    fours:  int
    sixes:  int
    sr:     float
    bpb:    float
    boundary_percent:   float
    summary: str
  

#define graph
graph = StateGraph(CricketState)

#define functions
def cal_sr(state: CricketState) -> CricketState:
    sr = (state['runs'] / state['balls']) * 100
    return {"sr": sr}

def cal_bpb(state: CricketState) -> CricketState:
    boundaries = state['fours'] + state['sixes']
    bpb = state['balls'] / boundaries if boundaries != 0 else 0
    return {"bpb": bpb}

def cal_bp(state: CricketState) -> CricketState:
    bp = ((state['fours']*4 + state['sixes']*6) / state['runs']) * 100 if state['runs'] != 0 else 0
    return {"boundary_percent": bp}

def summary(state: CricketState) -> CricketState:
    summary = f"""
    Strike Rate: {state['sr']:.2f}
    Balls per Boundary: {state['bpb']:.2f}
    Boundary %: {state['boundary_percent']:.2f}
    """
    return {"summary": summary}

#add node
graph.add_node('cal_sr', cal_sr)
graph.add_node('cal_bp', cal_bp)
graph.add_node('cal_bpb', cal_bpb)
graph.add_node('summary', summary)

#add edge
graph.add_edge(START, 'cal_sr')
graph.add_edge(START, 'cal_bp')
graph.add_edge(START, 'cal_bpb')
graph.add_edge('cal_sr', 'summary')
graph.add_edge('cal_bp', 'summary')
graph.add_edge('cal_bpb', 'summary')
graph.add_edge('summary', END)

workflow = graph.compile()

initial_state = {
    'runs': 100,
    'balls': 20,
    'fours': 10,
    'sixes': 4
}

result= workflow.invoke(initial_state)
print(result)

# Always return only the partial state (i.e., the fields updated by the node).
# Returning raw values (like floats or strings) will cause errors because
# LangGraph expects a dictionary of state updates.
# 
# In sequential workflows, returning full state may appear to work,
# but in parallel workflows it can lead to conflicts or InvalidUpdateError.
#
# Best practice: each node should return only the keys it updates,
# e.g., return {"sr": value} instead of returning state["sr"] or the full state.