# ============================================================
# 认证路由
# 登录 / 登出 / 用户信息
# ============================================================
from fastapi import APIRouter, Request
from loguru import logger

from backend.models.schemas import LoginRequest, LoginResponse, UserInfo, ApiResponse
from backend.services.auth_service import auth_service

router = APIRouter(prefix="/api/auth", tags=["认证管理"])


@router.post("/login", response_model=LoginResponse)
async def login(req: LoginRequest):
    """用户登录"""
    success, token, user_info = auth_service.authenticate(req.username, req.password)
    if not success:
        return LoginResponse(code=401, message="用户名或密码错误", data=None)

    return LoginResponse(
        code=200,
        message="登录成功",
        data={
            "token": token,
            "user": user_info.model_dump(),
        },
    )


@router.get("/me", response_model=ApiResponse)
async def get_current_user(request: Request):
    """获取当前登录用户信息"""
    user = auth_service.get_user_from_request(request)
    return ApiResponse(data=user.model_dump())


@router.post("/logout", response_model=ApiResponse)
async def logout():
    """登出（前端清除 Token 即可）"""
    return ApiResponse(message="已登出")
