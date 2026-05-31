"""全局异常处理中间件。

将所有未捕获的异常转换为标准化的 ErrorResponse JSON 格式。
确保 API 永远不会向客户端暴露原始错误堆栈。
"""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """捕获未处理的异常并返回标准化的错误响应。

    作用：
        1. 防止敏感信息（数据库路径、代码结构）泄露到客户端
        2. 确保所有错误响应格式一致
        3. 异常发生时记录完整的堆栈信息供运维排查
    """

    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as exc:
            # 记录完整异常信息（包括堆栈），仅向客户端返回通用错误消息
            logger.exception(
                "未处理的异常",
                path=request.url.path,
                method=request.method,
                error=str(exc),
            )
            return JSONResponse(
                status_code=500,
                content={
                    "detail": "服务器内部错误",
                    "error_code": "INTERNAL_ERROR",
                },
            )
