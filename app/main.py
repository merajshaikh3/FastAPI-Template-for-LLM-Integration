from fastapi import FastAPI
from .routers import sentiment
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(lifespan = sentiment.lifespan)

origins=["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sentiment.router)

