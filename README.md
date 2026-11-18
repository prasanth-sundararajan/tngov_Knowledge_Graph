# Tamil Nadu Government Schemes Knowledge Graph (KG-RAG)

> Turn government scheme pages into a knowledge graph and ask questions that use both text and relationships.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white) ![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logo=langchain&logoColor=white) ![Neo4j](https://img.shields.io/badge/Neo4j-4581C3?logo=neo4j&logoColor=white) ![Playwright](https://img.shields.io/badge/Playwright-2EAD33?logo=playwright&logoColor=white) ![OpenAI](https://img.shields.io/badge/OpenAI-412991?logo=openai&logoColor=white) ![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

## Overview

Scheme pages describe who is eligible, which department runs a scheme and what it provides, which is naturally a graph. This project scrapes the Tamil Nadu government [scheme list](https://www.tn.gov.in/scheme_list.php?dep_id=Mg==), uses an LLM to extract entities and relationships, stores them in **Neo4j** and **NetworkX**, and answers questions with **graph context plus hybrid text retrieval**.

## Features

- Scheme page scraping (Playwright + BeautifulSoup)
- LLM-based entity and relationship extraction (`LLMGraphTransformer`)
- Graph stored in **Neo4j** (Aura or local) and mirrored in **NetworkX**
- Hybrid retriever over scheme text
- Answers combine retrieved text with graph neighbourhoods
- Standalone scripts for Neo4j-only and NetworkX-only experiments

## Demo

![Graph visualisation](visualisation.png)

## How it works

```mermaid
flowchart LR
    W["tn.gov.in scheme pages"] --> S["Scraper"] --> D["Documents"]
    D --> T["LLMGraphTransformer"] --> N[("Neo4j")] & X[("NetworkX")]
    D --> H["Hybrid retriever"]
    Q["Question"] --> H & X
    H & X --> C["Text + graph context"] --> L["gpt-4o-mini"] --> A["Answer"]
```

## Tech stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Graph | Neo4j, NetworkX |
| Orchestration | LangChain, `langchain-experimental` |
| Models | OpenAI `gpt-4o-mini`, `text-embedding-3-small` |
| Scraping | Playwright, BeautifulSoup |

## Project structure

```text
.
├── KG_RAG.py            # full pipeline + Streamlit app
├── tngov_neo4j.py       # build the graph in Neo4j only
├── tngov_networkx.py    # build and query the graph with NetworkX only
├── visualisation.png
├── requirements.txt
└── .env.example
```

## Getting started

**Prerequisites:** Python 3.10+, a Neo4j database (local or [Neo4j Aura](https://neo4j.com/cloud/aura/))

```bash
git clone https://github.com/prasanth-sundararajan/tngov_Knowledge_Graph.git
cd tngov_Knowledge_Graph
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

Copy `.env.example` to `.env` and add your keys:

```bash
cp .env.example .env
```

## Usage

```bash
streamlit run KG_RAG.py
```
The first run scrapes the scheme pages and builds the graph; then ask questions such as *"Which schemes support women entrepreneurs?"*

## Configuration

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI API key used for chat and embeddings |
| `NEO4J_URI` | Yes | Neo4j connection URI (e.g. `neo4j+s://...`) |
| `NEO4J_USERNAME` | Yes | Neo4j user |
| `NEO4J_PASSWORD` | Yes | Neo4j password |
| `NEO4J_DATABASE` | No | Database name (defaults to the username) |

## License

Released under the [MIT License](LICENSE).

## Author

**Prasanth Sundararajan**, Senior SRE and AI Engineer  
[LinkedIn](https://www.linkedin.com/in/prasanth-s-a093a7109/) · [GitHub](https://github.com/prasanth-sundararajan)
