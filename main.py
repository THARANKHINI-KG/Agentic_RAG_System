from fastapi import FastAPI
from src.api.v1.routes.query_route import router

app = FastAPI()

app.include_router(router,prefix="/api/v1")