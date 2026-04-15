from src.api.v1.agents.agents import run_agent


def query_documents(query: str):
    return run_agent(query)