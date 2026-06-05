import datetime as dt
from typing import Optional

from pydantic import BaseModel, field_serializer


class BigIntStr(BaseModel):
    """Mixin: serialize all int IDs as strings so JS doesn't lose precision on CockroachDB big IDs."""

    @field_serializer("id", check_fields=False)
    @classmethod
    def serialize_id(cls, v):
        return str(v) if v is not None else None


# --- Fighters ---

class UFCFighterBase(BaseModel):
    ufcstats_id: str
    first_name: str
    last_name: str
    nickname: Optional[str] = None
    height: Optional[str] = None
    weight: Optional[str] = None
    reach: Optional[str] = None
    stance: Optional[str] = None
    dob: Optional[dt.date] = None
    wins: int = 0
    losses: int = 0
    draws: int = 0
    country_code: Optional[str] = None
    image_url: Optional[str] = None


class UFCFighterCreate(UFCFighterBase):
    pass


class UFCFighterResponse(BigIntStr, UFCFighterBase):
    id: int
    created_at: dt.datetime
    updated_at: dt.datetime

    model_config = {"from_attributes": True}


# --- Events ---

class UFCEventBase(BaseModel):
    ufcstats_id: str
    name: str
    date: dt.date
    location: Optional[str] = None


class UFCEventCreate(UFCEventBase):
    pass


class UFCEventResponse(BigIntStr, UFCEventBase):
    id: int
    created_at: dt.datetime

    model_config = {"from_attributes": True}


# --- Fights ---

class UFCFightBase(BaseModel):
    ufcstats_id: str
    date: Optional[dt.date] = None
    event_id: int
    red_fighter_id: int
    blue_fighter_id: int
    winner_id: Optional[int] = None
    red_result: Optional[str] = None
    blue_result: Optional[str] = None
    weight_class: Optional[str] = None
    method: Optional[str] = None
    details: Optional[str] = None
    referee: Optional[str] = None
    finish_round: Optional[int] = None
    finish_time: Optional[str] = None
    time_format: Optional[str] = None
    fight_time_seconds: Optional[int] = None
    max_fight_time_seconds: Optional[int] = None

    @field_serializer("event_id", "red_fighter_id", "blue_fighter_id", "winner_id", check_fields=False)
    @classmethod
    def serialize_fk_ids(cls, v):
        return str(v) if v is not None else None


class UFCFightCreate(UFCFightBase):
    pass


class UFCFightResponse(BigIntStr, UFCFightBase):
    id: int
    created_at: dt.datetime
    updated_at: dt.datetime

    model_config = {"from_attributes": True}


class UFCFightDetailResponse(UFCFightResponse):
    red_fighter: UFCFighterResponse
    blue_fighter: UFCFighterResponse
    winner: Optional[UFCFighterResponse] = None
    stats: list["UFCFightStatsResponse"] = []


class UFCEventDetailResponse(UFCEventResponse):
    fights: list[UFCFightDetailResponse] = []


# --- Fight Stats ---

class UFCFightStatsBase(BaseModel):
    fight_id: int
    fighter_id: int
    round_number: int = 0
    corner: str
    kd: int = 0
    sig_str_landed: int = 0
    sig_str_attempted: int = 0
    total_str_landed: int = 0
    total_str_attempted: int = 0
    td_landed: int = 0
    td_attempted: int = 0
    sub_att: int = 0
    rev: int = 0
    ctrl_seconds: int = 0
    head_landed: int = 0
    head_attempted: int = 0
    body_landed: int = 0
    body_attempted: int = 0
    leg_landed: int = 0
    leg_attempted: int = 0
    distance_landed: int = 0
    distance_attempted: int = 0
    clinch_landed: int = 0
    clinch_attempted: int = 0
    ground_landed: int = 0
    ground_attempted: int = 0

    @field_serializer("fight_id", "fighter_id", check_fields=False)
    @classmethod
    def serialize_stat_fk_ids(cls, v):
        return str(v) if v is not None else None


class UFCFightStatsCreate(UFCFightStatsBase):
    pass


class UFCFightStatsResponse(BigIntStr, UFCFightStatsBase):
    id: int
    created_at: dt.datetime
    updated_at: dt.datetime

    model_config = {"from_attributes": True}
