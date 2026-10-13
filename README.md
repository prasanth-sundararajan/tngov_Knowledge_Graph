# Tamil Nadu Government Schemes Knowledge Graph

This project is a small knowledge-graph and question-answering demo. It reads the Tamil Nadu Government schemes page, asks an OpenAI model to identify entities and relationships, stores the extracted graph in Neo4j, and runs a sample question against that graph.

## What the script does

The main program, `tngov_neo4j.py`:

1. Connects to Neo4j using settings from a `.env` file. It uses Certifi's CA bundle for TLS and Neo4j's routed connection.
2. Loads the Tamil Nadu Government schemes page.
3. Splits the page into text chunks (1,000 characters with 200 characters of overlap).
4. Processes the first three chunks with `gpt-4o-mini` to extract graph entities and relationships.
5. Stores the graph data in Neo4j.
6. Uses a LangChain Cypher question-answering chain to ask a sample question.

The source URL, sample question, and number of processed chunks are currently set in the Python script.

## Requirements

- Python 3.10 or newer
- An OpenAI API key
- A running Neo4j database and its URI, username, password, and database name
- The Python packages listed in `requirements.txt`

## Setup on Windows PowerShell

Open PowerShell in this folder:

```powershell
cd C:\Users\HP\GenAI_Program\knowledge_graph
```

Create and activate a virtual environment if you do not already have one:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Create a `.env` file in this folder. Add your own credentials locally; do not commit this file:

```env
OPENAI_API_KEY=your_openai_api_key
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=your_neo4j_username
NEO4J_PASSWORD=your_neo4j_password
NEO4J_DATABASE=your_neo4j_database_name
```

`NEO4J_DATABASE` is optional in the current script; if omitted, it defaults to the Neo4j username. Confirm that the selected database name exists in your Neo4j instance. For Neo4j Aura, use the exact URI and database details provided for your instance.

## Run

With the virtual environment active and `.env` configured:

```powershell
python tngov_neo4j.py
```

The program prints progress, the generated Cypher query, and the answer to its sample question.

## Security and usage notes

- Keep `.env` and all API credentials out of Git. If a key or password has been shared or exposed, revoke it and create a replacement.
- The script sends retrieved page text to OpenAI for graph extraction and question answering. Do not use it with confidential material unless that use is approved.
- `GraphCypherQAChain` is configured with `allow_dangerous_requests=True`. Run it only against a database you are authorized to use, and review generated Cypher queries before using this pattern with important data.
- This is a learning/demo project. Check generated answers against the source material before relying on them.

## Current limitations

- Only the first three text chunks are converted into graph data.
- The script asks one hard-coded sample question rather than accepting interactive questions.
- The imported LangChain community and experimental integrations may display deprecation warnings as those packages evolve.
