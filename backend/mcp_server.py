# ============================================================
# MCP Server - exposes KB tools via Model Context Protocol
# Usage:
#   python -m backend.mcp_server                 # stdio (for IDE)
#   python -m backend.mcp_server --port 8001      # SSE (for network)
# ============================================================
import sys, os, json, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("EnterpriseKB")

@mcp.tool()
async def search_knowledge(query: str, top_k: int = 5) -> str:
    """Search knowledge base. Returns relevant document fragments with citations."""
    from backend.services.hybrid_retriever import hybrid_retriever
    results = await hybrid_retriever.retrieve(query, top_k=min(top_k, 20))
    out = []
    for r in results:
        meta = r.get("metadata", {})
        out.append({
            "filename": meta.get("filename", ""),
            "content": r.get("content", "")[:500],
            "score": round(float(r.get("rrf_score", r.get("score", 0))), 4),
        })
    return json.dumps(out, ensure_ascii=False, indent=2)

@mcp.tool()
async def ask_question(question: str, top_k: int = 5) -> str:
    """Ask a question. Returns answer with citations from knowledge base."""
    from backend.services.react_agent import react_agent
    answer, passed = await react_agent.process(query=question, session_id="mcp", top_k=min(top_k, 20))
    return json.dumps({
        "answer": answer or "(No relevant info found)",
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
    parser.add_argument("--port", type=int, default=0,
                        help="SSE port (default: stdio mode)")
    args = parser.parse_args()

    if args.port:
        # SSE mode: start HTTP server on specified port
        import uvicorn
        app = mcp.sse_app()
        print(f"MCP SSE Server starting on port {args.port}...")
        print(f"  Connect: http://127.0.0.1:{args.port}/sse")
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else:
        # Stdio mode (for IDE integration like Cursor/VS Code)
        print(f"MCP Stdio Server starting...")
        print(f"  Configure in Cursor: python -m backend.mcp_server")
        mcp.run()
