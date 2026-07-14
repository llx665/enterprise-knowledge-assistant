# ============================================================
# 简易权限管理模块
# 区分访客(visitor)和管理员(admin)角色
# 基于 JWT Token 的身份校验
# ============================================================
import jwt
import time
from typing import Optional, Tuple
from loguru import logger

from backend.config import get_settings
from backend.models.schemas import UserInfo, UserRole
from backend.exceptions import UnauthorizedError

# 简易密钥（生产环境应替换为环境变量）
JWT_SECRET = "enterprise-kb-secret-key-2024"
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_SECONDS = 86400  # 24 小时

# 内置用户（简化版，生产环境应接入数据库）
_USERS = {
    "admin": {"password": "admin123", "role": UserRole.admin},
    "visitor": {"password": "visitor123", "role": UserRole.visitor},
}


class AuthService:
    """权限服务"""

    @staticmethod
    def authenticate(username: str, password: str) -> Tuple[bool, str, UserInfo]:
        """验证用户身份，成功返回 token"""
        user = _USERS.get(username)
        if not user or user["password"] != password:
            return False, "", UserInfo(username="visitor", role=UserRole.visitor)

        user_info = UserInfo(username=username, role=user["role"])
        payload = {
            "username": username,
            "role": user["role"].value,
            "exp": int(time.time()) + JWT_EXPIRE_SECONDS,
        }
        token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
        return True, token, user_info

    @staticmethod
    def verify_token(token: str) -> Optional[UserInfo]:
        """校验 Token，返回用户信息"""
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            return UserInfo(
                username=payload.get("username", "visitor"),
                role=UserRole(payload.get("role", "visitor")),
            )
        except jwt.ExpiredSignatureError:
            logger.warning("Token 已过期")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Token 无效: {e}")
            return None

    @staticmethod
    def require_admin(token: str) -> UserInfo:
        """要求管理员权限"""
        user = AuthService.verify_token(token)
        if not user:
            raise UnauthorizedError("请先登录")
        if user.role != UserRole.admin:
            raise UnauthorizedError("需要管理员权限")
        return user

    @staticmethod
    def get_user_from_request(request) -> UserInfo:
        """从请求头中提取用户信息"""
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            user = AuthService.verify_token(token)
            if user:
                return user
        return UserInfo(username="visitor", role=UserRole.visitor)


auth_service = AuthService()
