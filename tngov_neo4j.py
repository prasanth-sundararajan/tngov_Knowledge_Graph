import os
import ssl
from dotenv import load_dotenv
import certifi
from urllib.parse import urlsplit, urlunsplit

# Load variables from the .env file into environment variables
load_dotenv()
os.environ.setdefault(
    "USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
)

from langchain_community.document_loaders import WebBaseLoader
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Connect to Neo4j AuraDB using environment variables
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")

if not all([neo4j_uri, neo4j_username, neo4j_password]):
    raise ValueError(
        "Missing Neo4j config. Add these values to your .env file: "
        "NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD"
    )

uri_parts = urlsplit(neo4j_uri)
if uri_parts.scheme in {"neo4j+s", "bolt+s", "bolt"}:
    neo4j_uri = urlunsplit(
        ("neo4j", uri_parts.netloc, uri_parts.path, uri_parts.query, uri_parts.fragment)
    )

graph = Neo4jGraph(
    url=neo4j_uri,
    username=neo4j_username,
    password=neo4j_password,
    database=os.getenv("NEO4J_DATABASE", neo4j_username),
    driver_config={
        "encrypted": True,
        "ssl_context": ssl.create_default_context(cafile=certifi.where()),
    },
)

# 2. Load and split chunks from the TN government schemes page
url = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="
loader = WebBaseLoader(url)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

# 3. Extract Graph Documents via LLM
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
llm_transformer = LLMGraphTransformer(llm=llm)

print("Extracting entities and relationships from chunks for Neo4j upload...")
graph_documents = llm_transformer.convert_to_graph_documents(chunks[:3])

# 4. Store graph documents into Neo4j AuraDB
print("Storing graph documents into Neo4j AuraDB...")
graph.add_graph_documents(
    graph_documents,
    include_source=True,
    baseEntityLabel=True,
)
print("Successfully stored graph data into Neo4j AuraDB!")


# ==========================================
# STEP 5: KG-RAG Q&A CHATBOT CHAIN
# ==========================================
print("\nInitializing KG-RAG Q&A Chain...")
cypher_chain = GraphCypherQAChain.from_llm(
    llm=ChatOpenAI(temperature=0, model="gpt-4o-mini"),
    graph=graph,
    verbose=True,  # Shows the generated Cypher query in your terminal
    allow_dangerous_requests=True,  # Opt-in for running DB queries
)

# Test the chatbot with a question about your schemes
question = "What schemes or departments are mentioned in the database?"
print(f"\nAsking Question: {question}")

response = cypher_chain.invoke({"query": question})

print("\n--- KG-RAG Chatbot Response ---")
print(response["result"])