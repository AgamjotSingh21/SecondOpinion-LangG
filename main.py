from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from agents import (
    search_db, for_agent, against_agent,
    questioner_agent, for_agent_round2, against_agent_round2, judge_agent
)


class SecondOpinionState(TypedDict):
    question: str
    pool: list
    r1_for: str
    r1_against: str
    blind_spots: str
    updated_pool: list
    r2_for: str
    r2_against: str
    verdict: str


def initial_retrieval(state):
    print("\n" + "="*60)
    print("SECONDOPINION — DEBATE SYSTEM")
    print("="*60)
    print(f"\nQuestion: {state['question']}\n")

    print("Searching knowledge base...")
    pool = search_db(state["question"], n=5)
    print(f"Retrieved {len(pool)} relevant documents\n")

    return {"pool": pool}


def round1_for(state):
    print("-"*40)
    print("ROUND 1 — FOR AGENT")
    print("-"*40)

    r1_for = for_agent(state["question"], state["pool"])
    print(r1_for)

    return {"r1_for": r1_for}


def round1_against(state):
    print("\n" + "-"*40)
    print("ROUND 1 — AGAINST AGENT")
    print("-"*40)

    r1_against = against_agent(state["question"], state["pool"])
    print(r1_against)

    return {"r1_against": r1_against}


def questioner(state):
    print("\n" + "-"*40)
    print("QUESTIONER AGENT — BLIND SPOTS")
    print("-"*40)

    blind_spots, new_docs = questioner_agent(
        state["question"],
        state["r1_for"],
        state["r1_against"]
    )

    print(blind_spots)

    updated_pool = state["pool"] + new_docs

    return {
        "blind_spots": blind_spots,
        "updated_pool": updated_pool
    }


def round2_for(state):
    print("\n" + "-"*40)
    print("ROUND 2 — FOR AGENT ON BLIND SPOTS")
    print("-"*40)

    r2_for = for_agent_round2(
        state["question"],
        state["blind_spots"],
        state["updated_pool"]
    )

    print(r2_for)

    return {"r2_for": r2_for}


def round2_against(state):
    print("\n" + "-"*40)
    print("ROUND 2 — AGAINST AGENT ON BLIND SPOTS")
    print("-"*40)

    r2_against = against_agent_round2(
        state["question"],
        state["blind_spots"],
        state["updated_pool"]
    )

    print(r2_against)

    return {"r2_against": r2_against}


def judge(state):
    print("\n" + "="*60)
    print("JUDGE AGENT — FINAL VERDICT")
    print("="*60)

    verdict = judge_agent(
        state["question"],
        state["r1_for"],
        state["r1_against"],
        state["blind_spots"],
        state["r2_for"],
        state["r2_against"],
        state["updated_pool"]
    )

    print(verdict)
    print("\n" + "="*60)

    return {"verdict": verdict}


graph = StateGraph(SecondOpinionState)

graph.add_node("initial_retrieval", initial_retrieval)
graph.add_node("round1_for", round1_for)
graph.add_node("round1_against", round1_against)
graph.add_node("questioner", questioner)
graph.add_node("round2_for", round2_for)
graph.add_node("round2_against", round2_against)
graph.add_node("judge", judge)

graph.add_edge(START, "initial_retrieval")
graph.add_edge("initial_retrieval", "round1_for")
graph.add_edge("round1_for", "round1_against")
graph.add_edge("round1_against", "questioner")
graph.add_edge("questioner", "round2_for")
graph.add_edge("round2_for", "round2_against")
graph.add_edge("round2_against", "judge")
graph.add_edge("judge", END)

app = graph.compile()


def run_second_opinion(question):
    result = app.invoke({
        "question": question
    })

    return result["verdict"]


if __name__ == "__main__":
    question = input("Enter your decision question: ")
    run_second_opinion(question)