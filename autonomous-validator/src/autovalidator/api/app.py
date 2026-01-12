from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from autovalidator.runner import run_validation
from autovalidator.storage.artifacts import load_json


app = FastAPI(title="Autonomous Validator", version="0.1.0")


class ValidateRequest(BaseModel):
    dataset_ref: str
    out_dir: str = Field(default="./out")
    sample_n: int = 500
    seed: int = 42
    policy_path: Optional[str] = None


class ValidateResponse(BaseModel):
    run_id: str
    decision: str
    artifacts_dir: str


@app.post("/validate", response_model=ValidateResponse)
def validate(req: ValidateRequest) -> ValidateResponse:
    try:
        res = run_validation(
            dataset_ref=req.dataset_ref,
            out_dir=req.out_dir,
            sample_n=req.sample_n,
            seed=req.seed,
            policy_path=req.policy_path,
        )
        return ValidateResponse(run_id=res.run_id, decision=res.decision, artifacts_dir=res.run_dir)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="dataset_ref not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/runs/{run_id}")
def get_run(run_id: str, out_dir: str = "./out"):
    run_dir = Path(out_dir) / "runs" / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="run_id not found")
    final_report = load_json(run_dir / "final_report.json")
    return {"run_id": run_id, "final_report": final_report, "run_dir": str(run_dir)}
