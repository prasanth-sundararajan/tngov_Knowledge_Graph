import os
from dotenv import load_dotenv
import networkx as nx

# Load environment variables (for OpenAI API key)
load_dotenv()
os.environ.setdefault(
    "USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
)

from langchain_community.document_loaders import WebBaseLoader
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Initialize a native NetworkX Directed Graph
nx_graph = nx.DiGraph()

# 2. Load and split chunks from the TN government schemes page
url = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="
print(f"Loading content from: {url}")
loader = WebBaseLoader(url)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

# 3. Extract Graph Documents via LLM
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
llm_transformer = LLMGraphTransformer(llm=llm)

print("Extracting entities and relationships for NetworkX in-memory graph...")
graph_documents = llm_transformer.convert_to_graph_documents(chunks[:3])

# 4. Populate the NetworkX graph with nodes and relationships
print("Storing graph documents into NetworkX...")
for graph_doc in graph_documents:
    for node in graph_doc.nodes:
        # Add nodes with their type attributes
        nx_graph.add_node(node.id, type=node.type)
        
    for rel in graph_doc.relationships:
        # Add directed edges containing relationship types
        nx_graph.add_edge(
            rel.source.id, 
            rel.target.id, 
            relation=rel.type
        )

# 5. Inspect the Local NetworkX Graph
print("\nSuccessfully built NetworkX In-Memory Graph!")
print(f"Total Nodes: {nx_graph.number_of_nodes()}")
print(f"Total Edges: {nx_graph.number_of_edges()}")

# Print a preview of the extracted triples/edges
print("\nSample Edges in NetworkX Graph:")
for u, v, data in list(nx_graph.edges(data=True))[:10]:
    print(f"  ({u}) -[:{data.get('relation')}]-> ({v})")

# ---------------------------------------------------------
# 6. Interactive Q&A Session over NetworkX Graph
# ---------------------------------------------------------


def ask_networkx_graph(user_query: str) -> str:
  """Retrieves relevant triples from NetworkX and answers the question using LLM."""
  # Convert graph edges into text triples for the LLM context
  triples = []
  for u, v, data in nx_graph.edges(data=True):
    rel = data.get("relation", "CONNECTED_TO")
    triples.append(f"({u}) -[{rel}]-> ({v})")

  context = "\n".join(triples)

  prompt = f"""
    You are a Knowledge Graph QA assistant. Answer the user's question based strictly on the following knowledge graph triples:
    
    KNOWLEDGE GRAPH TRIPLES:
    {context}
    
    QUESTION:
    {user_query}
    
    ANSWER:
    """

  response = llm.invoke(prompt)
  return response.content


# Start interactive loop in terminal
print("\n" + "=" * 50)
print("  NetworkX Interactive KG-RAG Chatbot Ready!")
print("  (Type 'exit' or 'quit' to end the session)")
print("=" * 50 + "\n")

while True:
  user_input = input("Ask a question about TN Govt Schemes: ").strip()
  if user_input.lower() in ["exit", "quit"]:
    print("Exiting chatbot session. Goodbye!")
    break
  if not user_input:
    continue

  answer = ask_networkx_graph(user_input)
  print(f"\nAnswer: {answer}\n" + "-" * 40 + "\n")