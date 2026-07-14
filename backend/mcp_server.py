# ============================================================
# MCP Server - exposes KB tools via Model Context Protocol
# Usage: python -m backend.mcp_server (stdio, for IDE)
#        python -m backend.mcp_server --sse --port 8001
# ============================================================
import sys, os, json, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("EnterpriseKB")

@mcp.tool()
async def search_knowledge(query: str, top_k: int = 5) -> str:
    """Search knowledge base. Returns relevant document fragments."""
    from backend.services.hybrid_retriever import hybrid_retriever
    results = await hybrid_retriever.retrieve(query, top_k=min(top_k, 20))
    out = []
    for r in results:
        meta = r.get("metadata", {})
        out.append({
            "filename": meta.get("filename", ""),
            "content": r.get("content", "")[:500],
            "score": round(float(r.get("rrf_score", r.get("score", 0))), 4),
            "page_number": meta.get("page_number", ""),
        })
    return json.dumps(out, ensure_ascii=False, indent=2)

@mcp.tool()
async def ask_question(question: str, top_k: int = 5) -> str:
    """Ask a question. Returns answer with citations from knowledge base."""
    from backend.services.react_agent import react_agent
    answer, passed = await react_agent.process(query=question, session_id="mcp", top_k=min(top_k, 20))
    return json.dumps({
        "answer": answer or "(No relevant info found in knowledge base)",
        "self_check_passed": passed,
    }, ensure_ascii=False, indent=2)

@mcp.tool()
async def list_documents() -> str:
    """List all documents in the knowledge base."""
    docs = []
    if os.path.exists("./data/documents"):
        for fname in sorted(os.listdir("./data/documents")):
            fp = os.path.join("./data/documents", fname)
            if os.path.isfile(fp):
                docs.append({
                    "filename": fname,
                    "size_bytes": os.path.getsize(fp),
                    "file_type": os.path.splitext(fname)[1].lower(),
                })
    return json.dumps(docs, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sse", action="store_true", help="Use SSE transport")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    print(f"MCP Server starting ({'SSE:' + str(args.port) if args.sse else 'Stdio'})")
    if args.sse:
        mcp.run(transport="sse", port=args.port)
    else:
        mcp.run()
