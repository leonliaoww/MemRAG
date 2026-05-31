"""API router aggregation."""

from fastapi import APIRouter

from app.api.v1 import admin, documents, queries

api_router = APIRouter()
api_router.include_router(documents.router)
api_router.include_router(queries.router)
api_router.include_router(admin.router)

__all__ = ["api_router"]
