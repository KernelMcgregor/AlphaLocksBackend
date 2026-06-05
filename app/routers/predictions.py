from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.shared import ModelRun, OddsSnapshot, Prediction
from app.schemas.shared import (
    ModelRunCreate,
    ModelRunResponse,
    OddsSnapshotCreate,
    OddsSnapshotResponse,
    PredictionCreate,
    PredictionResponse,
)

router = APIRouter(tags=["predictions"])


@router.get("/predictions", response_model=list[PredictionResponse])
def list_predictions(sport: str | None = None, model_name: str | None = None, db: Session = Depends(get_db)):
    query = db.query(Prediction)
    if sport:
        query = query.filter(Prediction.sport == sport)
    if model_name:
        query = query.filter(Prediction.model_name == model_name)
    return query.order_by(Prediction.created_at.desc()).all()


@router.get("/predictions/{prediction_id}", response_model=PredictionResponse)
def get_prediction(prediction_id: int, db: Session = Depends(get_db)):
    prediction = db.get(Prediction, prediction_id)
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return prediction


@router.post("/predictions", response_model=PredictionResponse, status_code=status.HTTP_201_CREATED)
def create_prediction(data: PredictionCreate, db: Session = Depends(get_db)):
    prediction = Prediction(**data.model_dump())
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return prediction


@router.get("/model-runs", response_model=list[ModelRunResponse])
def list_model_runs(sport: str | None = None, db: Session = Depends(get_db)):
    query = db.query(ModelRun)
    if sport:
        query = query.filter(ModelRun.sport == sport)
    return query.order_by(ModelRun.run_date.desc()).all()


@router.post("/model-runs", response_model=ModelRunResponse, status_code=status.HTTP_201_CREATED)
def create_model_run(data: ModelRunCreate, db: Session = Depends(get_db)):
    run = ModelRun(**data.model_dump())
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


@router.get("/odds", response_model=list[OddsSnapshotResponse])
def list_odds(sport: str | None = None, event_id: int | None = None, db: Session = Depends(get_db)):
    query = db.query(OddsSnapshot)
    if sport:
        query = query.filter(OddsSnapshot.sport == sport)
    if event_id:
        query = query.filter(OddsSnapshot.event_id == event_id)
    return query.order_by(OddsSnapshot.timestamp.desc()).all()


@router.post("/odds", response_model=OddsSnapshotResponse, status_code=status.HTTP_201_CREATED)
def create_odds_snapshot(data: OddsSnapshotCreate, db: Session = Depends(get_db)):
    snapshot = OddsSnapshot(**data.model_dump())
    db.add(snapshot)
    db.commit()
    db.refresh(snapshot)
    return snapshot
