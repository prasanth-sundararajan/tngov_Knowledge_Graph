import logging
import os
import ssl
from urllib.parse import urljoin, urlsplit, urlunsplit

import certifi
import networkx as nx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault(
    "USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
)

from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from playwright.sync_api import sync_playwright

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SOURCE_URL = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="
VECTOR_DB_DIRECTORY = "./tn_chroma_db"
MAX_DETAIL_PAGES = 35
MAX_GRAPH_CHUNKS = 3

st.set_page_config(
    page_title="TN Government Schemes KG-RAG",
    page_icon="🏛️",
    layout="wide",
)
st.title("Tamil Nadu Government Schemes KG-RAG")
st.caption(
    "Searches official scheme pages with hybrid retrieval and combines "
    "knowledge graphs from Neo4j and NetworkX."
)


class SimpleHybridRetriever:
    def __init__(self, bm25_retriever, vector_retriever, k: int = 10):
        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever
        self.k = k

    def invoke(self, query: str):
        bm25_docs = self.bm25_retriever.invoke(query)
        vector_docs = self.vector_retriever.invoke(query)
        combined = []
        seen = set()
        for doc in bm25_docs + vector_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                combined.append(doc)
        return combined[: self.k]


def scrape_scheme_pages(base_url: str) -> list[Document]:
    documents = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(base_url, timeout=45000)
            page.wait_for_timeout(2000)
            soup = BeautifulSoup(page.content(), "html.parser")

            detail_links = set()
            for anchor in soup.find_all("a", href=True):
                link = urljoin(base_url, anchor["href"].strip())
                if "scheme_view.php" in link or "scheme" in link.lower():
                    detail_links.add(link)

            for element in soup(["script", "style"]):
                element.extract()
            directory_text = soup.get_text(separator="\n", strip=True)
            if directory_text:
                documents.append(
                    Document(
                        page_content=directory_text,
                        metadata={"source": base_url, "type": "directory"},
                    )
                )

            failures = []
            for link in list(detail_links)[:MAX_DETAIL_PAGES]:
                try:
                    page.goto(link, timeout=15000)
                    page.wait_for_timeout(800)
                    detail_soup = BeautifulSoup(page.content(), "html.parser")
                    for element in detail_soup(["script", "style"]):
                        element.extract()
                    detail_text = detail_soup.get_text(
                        separator="\n", strip=True
                    )
                    if len(detail_text) > 100:
                        documents.append(
                            Document(
                                page_content=detail_text,
                                metadata={
                                    "source": link,
                                    "type": "scheme_detail",
                                },
                            )
                        )
                except Exception as error:
                    logger.warning("Could not scrape scheme page %s: %s", link, error)
                    failures.append(link)

            if failures:
                st.warning(
                    f"Could not load {len(failures)} of the discovered scheme "
                    "detail pages. Continuing with the pages that loaded."
                )
        finally:
            browser.close()

    if not documents:
        raise RuntimeError("No text was extracted from the Tamil Nadu schemes page.")
    return documents


def create_neo4j_graph() -> Neo4jGraph:
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    if not all([neo4j_uri, neo4j_username, neo4j_password]):
        raise ValueError(
            "Neo4j is required. Add NEO4J_URI, NEO4J_USERNAME, and "
            "NEO4J_PASSWORD to the local .env file."
        )

    uri_parts = urlsplit(neo4j_uri)
    if uri_parts.scheme in {"neo4j+s", "bolt+s", "bolt"}:
        neo4j_uri = urlunsplit(
            (
                "neo4j",
                uri_parts.netloc,
                uri_parts.path,
                uri_parts.query,
                uri_parts.fragment,
            )
        )

    return Neo4jGraph(
        url=neo4j_uri,
        username=neo4j_username,
        password=neo4j_password,
        database=os.getenv("NEO4J_DATABASE", neo4j_username),
        driver_config={
            "encrypted": True,
            "ssl_context": ssl.create_default_context(cafile=certifi.where()),
        },
    )


def make_networkx_graph(graph_documents) -> nx.DiGraph:
    graph = nx.DiGraph()
    for graph_document in graph_documents:
        for node in graph_document.nodes:
            graph.add_node(node.id, type=node.type)
        for relationship in graph_document.relationships:
            graph.add_edge(
                relationship.source.id,
                relationship.target.id,
                relation=relationship.type,
            )
    return graph


@st.cache_resource(show_spinner=False)
def initialize_pipeline():
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Add OPENAI_API_KEY to the local .env file.")

    source_documents = scrape_scheme_pages(SOURCE_URL)
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150
    ).split_documents(source_documents)
    if not chunks:
        raise RuntimeError("The source pages did not produce any text chunks.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIRECTORY,
    )
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 10
    hybrid_retriever = SimpleHybridRetriever(bm25_retriever, vector_retriever)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    graph_transformer = LLMGraphTransformer(llm=llm)
    graph_documents = graph_transformer.convert_to_graph_documents(
        chunks[:MAX_GRAPH_CHUNKS]
    )

    neo4j_graph = create_neo4j_graph()
    neo4j_graph.add_graph_documents(
        graph_documents,
        include_source=True,
        baseEntityLabel=True,
    )
    networkx_graph = make_networkx_graph(graph_documents)

    return hybrid_retriever, neo4j_graph, networkx_graph, llm


def networkx_context(graph: nx.DiGraph, query: str) -> str:
    terms = [word.lower().strip(".,?!:;()[]{}") for word in query.split()]
    terms = [word for word in terms if len(word) > 2]
    matching_nodes = {
        node
        for node in graph.nodes
        if any(term in str(node).lower() for term in terms)
    }
    edges = [
        (source, target, data)
        for source, target, data in graph.edges(data=True)
        if source in matching_nodes or target in matching_nodes
    ]
    if not edges:
        edges = list(graph.edges(data=True))[:20]
    return "\n".join(
        f"({source}) -[{data.get('relation', 'RELATED_TO')}]-> ({target})"
        for source, target, data in edges[:30]
    )


def neo4j_context(graph: Neo4jGraph, query: str) -> str:
    terms = [
        word.lower().strip(".,?!:;()[]{}")
        for word in query.split()
        if len(word.strip(".,?!:;()[]{}")) > 2
    ][:8]
    if not terms:
        return "No matching Neo4j entities were found."

    cypher = """
    MATCH (n:__Entity__)
    WHERE any(term IN $terms WHERE toLower(toString(n.id)) CONTAINS term)
    OPTIONAL MATCH (n)-[r]-(m)
    RETURN n.id AS source, type(r) AS relation, m.id AS target
    LIMIT 20
    """
    records = graph.query(cypher, params={"terms": terms})
    if not records:
        return "No matching Neo4j entities were found."
    return "\n".join(str(record) for record in records)


def answer_question(query: str, pipeline) -> str:
    hybrid_retriever, neo4j_graph, networkx_graph, llm = pipeline
    retrieved_docs = hybrid_retriever.invoke(query)
    retrieved_context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
        for doc in retrieved_docs
    )

    graph_context = networkx_context(networkx_graph, query)
    try:
        neo4j_triples = neo4j_context(neo4j_graph, query)
    except Exception as error:
        logger.exception("Neo4j graph lookup failed.")
        st.warning(f"Neo4j graph lookup failed; using other retrieved context: {error}")
        neo4j_triples = "Neo4j graph lookup failed."

    prompt = f"""
You answer questions about Tamil Nadu Government schemes.
Use only the retrieved source text and graph data below. Do not guess or use
outside knowledge. Preserve eligibility conditions, amounts, and requirements
when they are present. If the answer is not available, say that you could not
find it in the retrieved official source content.

Retrieved source text:
{retrieved_context}

Neo4j graph facts:
{neo4j_triples}

NetworkX graph facts:
{graph_context}

User question: {query}
"""
    response = llm.invoke(prompt)
    return response.content


try:
    with st.spinner(
        "Loading scheme pages, building hybrid search, and preparing both graphs..."
    ):
        pipeline = initialize_pipeline()
    st.success("The Neo4j and NetworkX knowledge graphs are ready.")
except Exception as error:
    logger.exception("Could not initialize the KG-RAG pipeline.")
    st.error(f"Could not initialize the app: {error}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask about Tamil Nadu Government schemes..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching scheme documents and both knowledge graphs..."):
            try:
                answer = answer_question(question, pipeline)
            except Exception as error:
                logger.exception("Could not answer the question.")
                answer = f"Sorry, the question could not be answered: {error}"
        st.markdown(answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
