"""Servidor MCP (FastMCP, transporte streamable-http) que expõe o RAG acadêmico como tools."""
import os

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

import rag

mcp = FastMCP("academico-sis", host="0.0.0.0", port=int(os.getenv("MCP_PORT", "8001")))


class Chunk(BaseModel):
    text: str = Field(description="Trecho recuperado do documento")
    source: str = Field(description="Arquivo de origem")
    section: str = Field(description="Seção do documento (quando houver)")
    topic: str = Field(description="Topic key do chunk")
    score: float = Field(description="Similaridade de cosseno com a consulta (0-1)")


class SearchResult(BaseModel):
    count: int
    results: list[Chunk]


def _search(query: str, top_k: int, topic: str | None) -> SearchResult:
    # guardrail de execução: parâmetros válidos antes de consultar o banco
    if not query.strip():
        raise ValueError("query não pode ser vazia")
    if not 1 <= top_k <= 10:
        raise ValueError("top_k deve estar entre 1 e 10")
    results = rag.search(query, top_k, topic)
    return SearchResult(count=len(results), results=[Chunk(**r) for r in results])


@mcp.tool()
def search_all(query: str, top_k: int = 5) -> SearchResult:
    """Busca semântica em TODOS os documentos do corpus acadêmico do curso de Sistemas de Informação
    da AMF (matriz curricular, guia acadêmico 2026, grade horária 2027/1). Use para calendário,
    matrícula, normas, notas, frequência, horários, serviços e qualquer dúvida geral."""
    return _search(query, top_k, None)


@mcp.tool()
def search_by_dependencies(query: str, top_k: int = 5) -> SearchResult:
    """Busca apenas nos trechos da matriz curricular sobre disciplinas, períodos e PRÉ-REQUISITOS
    (qual disciplina libera qual, cadeias de pré-requisitos, eletivas). Use para perguntas do tipo
    'qual o pré-requisito de X' ou 'o que preciso cursar antes de X'."""
    return _search(query, top_k, "pre_requisitos")


@mcp.tool()
def horas_complementares(query: str, top_k: int = 5) -> SearchResult:
    """Busca apenas nos trechos sobre ATIVIDADES/HORAS COMPLEMENTARES: carga horária exigida,
    tabelas de equivalência, como solicitar o aproveitamento."""
    return _search(query, top_k, "horas_complementares")


if __name__ == "__main__":
    rag.embedder()  # carrega o modelo antes de aceitar conexões
    mcp.run(transport="streamable-http")
