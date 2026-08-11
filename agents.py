import requests
import chromadb
from sentence_transformers import SentenceTransformer



chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection("startup_knowledge")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text):
    return embed_model.encode(text).tolist()

def search_db(query, n=5):
    query_vector = embed_text(query)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n
    )
    return results["documents"][0]

def call_llm(system_prompt, user_message):
    url = "http://localhost:11434/api/generate"
    
    prompt = f"{system_prompt}\n\n{user_message}"
    
    response = requests.post(url, json={
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    })
    
    return response.json()["response"]

def for_agent(question, docs):
    context = "\n\n".join(docs[:3])
    system = """You are a passionate startup advocate. 
Argue as strongly as possible FOR the user's decision.
Use the provided documents as evidence.
Never hedge. Never be diplomatic. Take a strong position."""
    
    user = f"""Decision: {question}

Knowledge documents:
{context}

Argue strongly FOR this decision."""
    
    return call_llm(system, user)

def against_agent(question, docs):
    context = "\n\n".join(docs[:3])
    system = """You are a ruthless devil's advocate.
Argue as strongly as possible AGAINST the user's decision.
Use the provided documents as evidence.
Never hedge. Never be diplomatic. Take a strong position."""
    
    user = f"""Decision: {question}

Knowledge documents:
{context}

Argue strongly AGAINST this decision."""
    
    return call_llm(system, user)

def questioner_agent(question, for_output, against_output):
    system = """You are an expert at finding blind spots.
Read both arguments and identify what BOTH agents missed.
Then generate exactly 3 sharp follow-up questions.
These questions should expose the most critical gaps."""
    
    user = f"""Original question: {question}

FOR argued: {for_output}

AGAINST argued: {against_output}

Identify blind spots and generate 3 follow-up questions."""
    
    blind_spots = call_llm(system, user)
    new_docs = search_db(blind_spots, n=3)
    
    return blind_spots, new_docs

def judge_agent(question, r1_for, r1_against,
                blind_spots, r2_for, r2_against, docs):
    context = "\n\n".join(docs)
    system = """You are a Judge. Give a definitive verdict.
You are NOT allowed to be diplomatic or balanced.
You MUST choose: GO, NO-GO, or CONDITIONAL GO.

Output in this exact format:

VERDICT: [GO / NO-GO / CONDITIONAL GO]
CONFIDENCE: [0-100]%

DECIDING FACTOR:
[The single most important reason for your verdict]

REASONING:
[Why this verdict and not the other]

STRONGEST OPPOSING POINT:
[One sentence only acknowledging the losing side]

UNRESOLVED BLIND SPOTS:
[What the user still needs to figure out]

NEXT ACTION:
[One specific concrete thing to do immediately]"""
    
    user = f"""Question: {question}

Round 1 FOR: {r1_for}

Round 1 AGAINST: {r1_against}

Blind spots identified: {blind_spots}

Round 2 FOR: {r2_for}

Round 2 AGAINST: {r2_against}

Knowledge base: {context}

Give your definitive verdict."""
    
    return call_llm(system, user)