import requests
import os
import re
import chromadb
from sentence_transformers import SentenceTransformer


CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
chroma = chromadb.PersistentClient(path=CHROMA_DB_PATH)
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

def clean_and_ensure_verdict(text):
    if not text:
        return (
            "VERDICT: NO-GO\n"
            "CONFIDENCE: 85%\n\n"
            "DECIDING FACTOR:\n"
            "Insufficient information provided to validate the decision.\n\n"
            "REASONING:\n"
            "Without reliable business metrics and market validation, moving forward presents disproportionate risk.\n\n"
            "STRONGEST OPPOSING POINT:\n"
            "The opportunity could be explored if critical market data becomes available.\n\n"
            "UNRESOLVED BLIND SPOTS:\n"
            "1. Target customer willingness to pay.\n"
            "2. Sustainable unit economics.\n\n"
            "NEXT ACTION:\n"
            "Conduct preliminary customer interviews to validate demand."
        )

    # 1. Filter unwanted metadata lines
    filtered_lines = []
    for line in text.splitlines():
        trimmed = line.strip()
        if re.match(r"^(?:Decision\s*ID|Domain|Company(?:\s*Stage)?|Year|Stage|Growth|Decision\s*Category)\s*:\s*", trimmed, flags=re.IGNORECASE):
            continue
        filtered_lines.append(line)
    
    cleaned = "\n".join(filtered_lines).strip()

    # 2. Extract verdict (GO / NO-GO / CONDITIONAL GO)
    verdict_match = re.search(
        r"(?:\*{0,3}|#*\s*)VERDICT(?:\*{0,3})\s*:\s*\[?(?:\*{0,2})(CONDITIONAL\s+GO|NO\s*-?\s*GO|GO)(?:\*{0,2})\]?",
        cleaned,
        flags=re.IGNORECASE
    )
    if verdict_match:
        raw_v = re.sub(r"\s+", " ", verdict_match.group(1).upper()).strip()
        verdict_str = raw_v.replace("NO GO", "NO-GO")
    else:
        kw_match = re.search(r"\b(CONDITIONAL\s+GO|NO\s*-?\s*GO|GO)\b", cleaned, flags=re.IGNORECASE)
        if kw_match:
            raw_v = re.sub(r"\s+", " ", kw_match.group(1).upper()).strip()
            verdict_str = raw_v.replace("NO GO", "NO-GO")
        else:
            verdict_str = "NO-GO"

    # 3. Extract confidence percentage
    conf_match = re.search(
        r"(?:\*{0,3}|#*\s*)CONFIDENCE(?:\s*SCORE)?(?:\*{0,3})\s*:\s*\[?(?:\*{0,2})(\d{1,3})\s*%?(?:\*{0,2})\]?",
        cleaned,
        flags=re.IGNORECASE
    )
    if conf_match:
        conf_val = max(0, min(100, int(conf_match.group(1))))
    else:
        pct_match = re.search(r"(\d{1,3})\s*%", cleaned)
        if pct_match:
            conf_val = max(0, min(100, int(pct_match.group(1))))
        else:
            if verdict_str == "CONDITIONAL GO":
                conf_val = 75
            elif verdict_str == "NO-GO":
                conf_val = 90
            else:
                conf_val = 85

    # 4. Remove leading VERDICT / CONFIDENCE lines from body
    cleaned_body = re.sub(
        r"^(?:\s*(?:\*{0,3}|#*\s*)(?:VERDICT|CONFIDENCE(?:\s*SCORE)?)[^\n]*\n?)+",
        "",
        cleaned,
        flags=re.IGNORECASE
    ).strip()

    # 5. Clean markdown bolding and header formatting around standard sections
    sections = [
        ("DECIDING FACTOR", r"DECIDING\s*FACTOR"),
        ("REASONING", r"REASONING"),
        ("STRONGEST OPPOSING POINT", r"STRONGEST\s*OPPOSING\s*POINT"),
        ("UNRESOLVED BLIND SPOTS", r"UNRESOLVED\s*BLIND\s*SPOTS"),
        ("NEXT ACTION", r"NEXT\s*ACTION"),
    ]
    for standard_title, pattern in sections:
        cleaned_body = re.sub(
            rf"(?m)^[\s#*_]*{pattern}[\s#*_]*:\s*[\s*_]*",
            f"\n\n{standard_title}:\n",
            cleaned_body,
            flags=re.IGNORECASE
        )

    # 6. Ensure clean spacing between sections
    cleaned_body = re.sub(r"\n{3,}", "\n\n", cleaned_body).strip()

    final_output = f"VERDICT: {verdict_str}\nCONFIDENCE: {conf_val}%\n\n{cleaned_body}"
    return final_output

def for_agent(question, docs):
    context = "\n\n".join(docs[:3])
    system = """You are a passionate startup advocate. 
Argue as strongly as possible FOR the user's decision.
Use the provided documents as evidence.
Do NOT mention metadata fields like Year, Company, ID, Domain, or Stage.
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
Do NOT mention metadata fields like Year, Company, ID, Domain, or Stage.
Never hedge. Never be diplomatic. Take a strong position."""
    
    user = f"""Decision: {question}

Knowledge documents:
{context}

Argue strongly AGAINST this decision."""
    
    return call_llm(system, user)

def questioner_agent(question, for_output, against_output):
    system = """You are an expert at finding blind spots in startup decisions.
Read both arguments and identify what BOTH agents missed.
Then generate exactly 3 sharp follow-up questions.
Do NOT mention metadata fields like Year, Company, ID, Domain, or Stage.
These questions should expose the most critical gaps."""
    
    user = f"""Original question: {question}

FOR argued: {for_output}

AGAINST argued: {against_output}

Identify blind spots and generate 3 follow-up questions."""
    
    blind_spots = call_llm(system, user)
    search_query = f"{question} {blind_spots[:200]}"
    new_docs = search_db(search_query, n=3)
    
    return blind_spots, new_docs

def for_agent_round2(question, blind_spots, docs):
    context = "\n\n".join(docs[:4])
    system = """You are a passionate startup advocate in Round 2 of a debate.
Address the blind spots and follow-up questions raised by the Questioner.
Defend the user's decision vigorously against these criticisms.
Use the provided documents as evidence.
Do NOT mention metadata fields like Year, Company, ID, Domain, or Stage.
Never hedge. Never be diplomatic. Take a strong, confident position in favor of the decision."""

    user = f"""Original Decision Question: {question}

Blind spots and follow-up questions to address:
{blind_spots}

Knowledge documents:
{context}

Address the blind spots and argue strongly FOR the decision in context of the original question."""

    return call_llm(system, user)

def against_agent_round2(question, blind_spots, docs):
    context = "\n\n".join(docs[:4])
    system = """You are a ruthless devil's advocate in Round 2 of a debate.
Exploit the blind spots and follow-up questions raised by the Questioner.
Attack the user's decision by pressing hard on these critical vulnerabilities and risks.
Use the provided documents as evidence.
Do NOT mention metadata fields like Year, Company, ID, Domain, or Stage.
Never hedge. Never be diplomatic. Take a strong, uncompromising position against the decision."""

    user = f"""Original Decision Question: {question}

Blind spots and follow-up questions to exploit:
{blind_spots}

Knowledge documents:
{context}

Exploit the blind spots and argue strongly AGAINST the decision in context of the original question."""

    return call_llm(system, user)

def judge_agent(question, r1_for, r1_against,
                blind_spots, r2_for, r2_against, docs):
    context = "\n\n".join(docs)
    system = """You are a Judge delivering the final verdict on an entrepreneurial decision debate.
You are NOT allowed to be diplomatic or balanced.
You MUST evaluate all arguments and choose exactly one of the following 3 verdicts:

- GO: Choose this if the arguments in favor demonstrate clear product-market fit, sustainable economics, and manageable risks.
- NO-GO: Choose this if the risks, lack of local validation, poor economics, or market headwinds make failure likely.
- CONDITIONAL GO: Choose this if the decision is viable ONLY under specific mandatory conditions (e.g. validating demand first, securing partners, or limiting initial capital outlay).

MANDATORY OUTPUT FORMAT RULES:
1. The first line MUST be: VERDICT: [GO, NO-GO, or CONDITIONAL GO]
2. The second line MUST be: CONFIDENCE: [0-100]%
3. Ground your deciding factor, reasoning, and next actions strictly in context of the user's decision question.
4. Do NOT output metadata fields such as Year, Company, ID, Domain, Stage, Growth, or Decision Category.
5. Do NOT include markdown bold asterisks on section headers or brackets around values.

OUTPUT FORMAT TEMPLATE:
VERDICT: NO-GO
CONFIDENCE: 95%

DECIDING FACTOR:
The lack of specific market validation, scalable unit economics, and sustainable financial planning for the proposed venture.

REASONING:
While general market growth is positive, it does not outweigh the severe execution risks without verified local demand. The plan relies on first-mover advantage rather than sustainable unit economics and adequate cash runway.

STRONGEST OPPOSING POINT:
The market growth trends offer potential upside if local operational challenges can be resolved.

UNRESOLVED BLIND SPOTS:
1. Lack of verified demand data for the target customer segment.
2. Unclear unit economics and customer acquisition costs.
3. Insufficient cash runway to reach profitability.

NEXT ACTION:
Conduct direct customer interviews and build a 12-month unit economics model before committing capital."""
    
    user = f"""Question: {question}

Round 1 FOR: {r1_for}

Round 1 AGAINST: {r1_against}

Blind spots identified: {blind_spots}

Round 2 FOR: {r2_for}

Round 2 AGAINST: {r2_against}

Knowledge base evidence:
{context}

Give your definitive verdict following the exact format above."""
    
    raw_response = call_llm(system, user)
    return clean_and_ensure_verdict(raw_response)
