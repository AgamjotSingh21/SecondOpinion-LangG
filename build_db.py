import os
import json
import chromadb
from sentence_transformers import SentenceTransformer


CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", "./knowledge_base")
chroma = chromadb.PersistentClient(path=CHROMA_DB_PATH)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
COLLECTION_NAME = "startup_knowledge"
BATCH_SIZE = 64

def embed_text(text):
    return embed_model.encode(text).tolist()

def reset_collection():
    try:
        chroma.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    return chroma.get_or_create_collection(COLLECTION_NAME)

def format_list(title, values):
    if not values:
        return f"{title}: Not available"
    if isinstance(values, list):
        return f"{title}: " + " | ".join(str(value) for value in values)
    return f"{title}: {values}"

def decision_record_to_document(record):
    parts = []
    if record.get('Problem_Statement'):
        parts.append(f"Problem Statement: {record.get('Problem_Statement')}")
    if record.get('Business_Goal'):
        parts.append(f"Business Goal: {record.get('Business_Goal')}")
    if record.get('Market_Condition') or record.get('Economic_Environment'):
        parts.append(f"Market & Economic Context: {record.get('Market_Condition', '')} | {record.get('Economic_Environment', '')}")
    if record.get('Competitor_Situation'):
        parts.append(f"Competitor Situation: {record.get('Competitor_Situation')}")
    if record.get('Customer_Segment'):
        parts.append(f"Customer Segment: {record.get('Customer_Segment')}")
    if record.get('Budget_Available') or record.get('Cash_Runway'):
        parts.append(f"Financials: Budget: {record.get('Budget_Available', '')}, Burn: {record.get('Burn_Rate', '')}, Runway: {record.get('Cash_Runway', '')} months")
    if record.get('Decision_Taken'):
        parts.append(f"Decision Considered: {record.get('Decision_Taken')}")
    if record.get('Reason_for_Decision'):
        parts.append(f"Strategic Rationale: {record.get('Reason_for_Decision')}")
    if record.get('Arguments_in_Favor'):
        parts.append(format_list("Arguments in Favor", record.get("Arguments_in_Favor")))
    if record.get('Arguments_Against'):
        parts.append(format_list("Arguments Against", record.get("Arguments_Against")))
    if record.get('Outcome'):
        parts.append(f"Outcome: {record.get('Outcome')}")
    if record.get('Lessons_Learned'):
        parts.append(f"Lessons Learned: {record.get('Lessons_Learned')}")
    return "\n".join(parts)

def record_metadata(record, filename):
    return {
        "filename": filename,
        "decision_id": str(record.get("Decision_ID", "")),
        "domain": str(record.get("Domain", "")),
        "company": str(record.get("Company", "")),
        "year": str(record.get("Year", "")),
        "company_stage": str(record.get("Company_Stage", "")),
        "decision_category": str(record.get("Decision_Category", "")),
        "outcome": str(record.get("Outcome", "")),
        "evidence_level": str(record.get("Evidence_Level", "")),
    }

def add_batch(collection, documents, ids, metadatas):
    embeddings = embed_model.encode(documents).tolist()
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )

def add_json_file(collection, filepath, filename, start_index):
    with open(filepath, "r", encoding="utf-8") as f:
        records = json.load(f)

    if isinstance(records, dict):
        records = [records]

    documents = []
    ids = []
    metadatas = []
    added = 0

    for offset, record in enumerate(records):
        if not isinstance(record, dict):
            continue

        decision_id = record.get("Decision_ID") or f"{start_index + offset}"
        documents.append(decision_record_to_document(record))
        ids.append(f"decision_{decision_id}")
        metadatas.append(record_metadata(record, filename))

        if len(documents) >= BATCH_SIZE:
            add_batch(collection, documents, ids, metadatas)
            added += len(documents)
            print(f"Added {added} records from {filename}")
            documents, ids, metadatas = [], [], []

    if documents:
        add_batch(collection, documents, ids, metadatas)
        added += len(documents)

    print(f"Added: {filename} ({added} records)")
    return added

def add_text_file(collection, filepath, filename, doc_index):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    vector = embed_text(text)

    collection.add(
        documents=[text],
        embeddings=[vector],
        ids=[f"doc_{doc_index}"],
        metadatas=[{"filename": filename, "source_type": "text"}]
    )
    print(f"Added: {filename}")
    return 1

def build_db():
    folder = KNOWLEDGE_BASE_DIR
    files = sorted(os.listdir(folder))
    collection = reset_collection()
    
    added = 0
    for filename in files:
        filepath = os.path.join(folder, filename)
        if filename.endswith(".json"):
            added += add_json_file(collection, filepath, filename, added)
        elif filename.endswith(".txt"):
            added += add_text_file(collection, filepath, filename, added)
    
    print(f"Database built with {added} documents")

build_db()
