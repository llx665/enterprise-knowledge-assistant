from backend.router.auth import router as auth_router
from backend.router.upload import router as upload_router
from backend.router.chat import router as chat_router
from backend.router.knowledge import router as knowledge_router
from backend.router.index import router as index_router
from backend.router.logs import router as logs_router

__all__ = ["auth_router", "upload_router", "chat_router",
           "knowledge_router", "index_router", "logs_router"]
