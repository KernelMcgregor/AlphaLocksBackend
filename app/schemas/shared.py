from datetime import datetime

from pydantic import BaseModel


class PredictionBase(BaseModel):
    sport: str
    event_id: int
    model_name: str
    predicted_outcome: str
    confidence: float
    actual_outcome: str | None = None


class PredictionCreate(PredictionBase):
    pass


class PredictionResponse(PredictionBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ModelRunBase(BaseModel):
    sport: str
    model_name: str
    run_date: datetime
    accuracy: float
    notes: str | None = None


class ModelRunCreate(ModelRunBase):
    pass


class ModelRunResponse(ModelRunBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class OddsSnapshotBase(BaseModel):
    sport: str
    event_id: int
    source: str
    home_odds: float | None = None
    away_odds: float | None = None
    draw_odds: float | None = None
    over_under: float | None = None


class OddsSnapshotCreate(OddsSnapshotBase):
    pass


class OddsSnapshotResponse(OddsSnapshotBase):
    id: int
    timestamp: datetime
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
