from fastapi import APIRouter

from api.endpoints import test, name


api_router = APIRouter()
api_router.include_router(test.router, prefix="/test", tags=["Test"])
api_router.include_router(name.router, prefix="/name", tags=["Name_pet"])
