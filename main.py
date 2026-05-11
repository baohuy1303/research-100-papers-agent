from fastapi import FastAPI
from api.routes.papers import router as papers_router

app = FastAPI(title="Research Comprehension API", version="0.1.0")

app.include_router(papers_router)


@app.get("/")
def health():
    return {"status": "ok"}
