"""
FastAPI web app for GitMatch.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="GitMatch", description="GitHub Issues Recommender System")


class RecommendRequest(BaseModel):
    username: str
    top_k_repos: int = 5
    top_k_issues: int = 10


class IssueResult(BaseModel):
    number: int
    title: str
    url: str
    labels: list[str]
    relevance_score: float


class RepoResult(BaseModel):
    repo: str
    explanation: str
    issues: list[IssueResult]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend", response_model=list[RepoResult])
def recommend(request: RecommendRequest):
    """
    Given a GitHub username, return ranked repositories and open issues.
    """
    # TODO: load models at startup and wire up stage1 + stage2 pipelines
    raise HTTPException(status_code=501, detail="Model loading not yet implemented.")
