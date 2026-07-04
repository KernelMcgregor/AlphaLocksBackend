import datetime as dt
from typing import Optional

from pydantic import BaseModel, field_serializer


class BigIntStr(BaseModel):
    """Mixin: serialize int IDs as strings so JS doesn't lose precision on large IDs."""

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
    red_odds: Optional[int] = None
    blue_odds: Optional[int] = None


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

    # Derived columns (populated on totals rows only)
    fight_time_min: Optional[float] = None
    est_standing_min: Optional[float] = None
    est_ground_min: Optional[float] = None
    slpm: Optional[float] = None
    sapm: Optional[float] = None
    sl_diff: Optional[float] = None
    sig_acc: Optional[float] = None
    sig_def: Optional[float] = None
    tslpm: Optional[float] = None
    head_pct: Optional[float] = None
    head_pm: Optional[float] = None
    head_acc: Optional[float] = None
    head_abs_pct: Optional[float] = None
    head_abs_pm: Optional[float] = None
    head_def: Optional[float] = None
    body_pct: Optional[float] = None
    body_pm: Optional[float] = None
    body_acc: Optional[float] = None
    body_abs_pct: Optional[float] = None
    body_abs_pm: Optional[float] = None
    body_def: Optional[float] = None
    leg_pct: Optional[float] = None
    leg_pm: Optional[float] = None
    leg_acc: Optional[float] = None
    leg_abs_pct: Optional[float] = None
    leg_abs_pm: Optional[float] = None
    leg_def: Optional[float] = None
    dist_pct: Optional[float] = None
    dist_pm: Optional[float] = None
    dist_acc: Optional[float] = None
    dist_abs_pct: Optional[float] = None
    dist_abs_pm: Optional[float] = None
    dist_def: Optional[float] = None
    clinch_pct: Optional[float] = None
    clinch_pm: Optional[float] = None
    clinch_acc: Optional[float] = None
    clinch_abs_pct: Optional[float] = None
    clinch_abs_pm: Optional[float] = None
    clinch_def: Optional[float] = None
    ground_pct: Optional[float] = None
    ground_pm: Optional[float] = None
    ground_acc: Optional[float] = None
    ground_abs_pct: Optional[float] = None
    ground_abs_pm: Optional[float] = None
    ground_def: Optional[float] = None
    gnp15g: Optional[float] = None
    gnp_abs15g: Optional[float] = None
    kd15: Optional[float] = None
    kd15s: Optional[float] = None
    kd_abs15: Optional[float] = None
    kd_abs15s: Optional[float] = None
    td15: Optional[float] = None
    td15s: Optional[float] = None
    td_acc: Optional[float] = None
    td_abs15: Optional[float] = None
    td_abs15s: Optional[float] = None
    td_def: Optional[float] = None
    ctrl15: Optional[float] = None
    ctrl15g: Optional[float] = None
    ctrl_abs15: Optional[float] = None
    ctrl_abs15g: Optional[float] = None
    sub_att15: Optional[float] = None
    sub_att15g: Optional[float] = None
    sub_abs15: Optional[float] = None
    sub_abs15g: Optional[float] = None
    rev15: Optional[float] = None
    rev_abs15: Optional[float] = None

    @field_serializer("fight_id", "fighter_id", check_fields=False)
    @classmethod
    def serialize_stat_fk_ids(cls, v):
        return str(v) if v is not None else None


class UFCFightStatsCreate(UFCFightStatsBase):
    pass


# --- Fighter Career Stats ---

class UFCFighterCareerStatsResponse(BigIntStr, BaseModel):
    id: int
    fighter_id: int

    # Foundation
    fight_count: int = 0
    total_fight_min: Optional[float] = None
    est_standing_min: Optional[float] = None
    est_ground_min: Optional[float] = None

    # Striking: Overall
    slpm: Optional[float] = None
    sapm: Optional[float] = None
    sl_diff: Optional[float] = None
    sig_acc: Optional[float] = None
    sig_def: Optional[float] = None
    tslpm: Optional[float] = None

    # Striking: Head
    head_pct: Optional[float] = None
    head_pm: Optional[float] = None
    head_acc: Optional[float] = None
    head_abs_pct: Optional[float] = None
    head_abs_pm: Optional[float] = None
    head_def: Optional[float] = None

    # Striking: Body
    body_pct: Optional[float] = None
    body_pm: Optional[float] = None
    body_acc: Optional[float] = None
    body_abs_pct: Optional[float] = None
    body_abs_pm: Optional[float] = None
    body_def: Optional[float] = None

    # Striking: Legs
    leg_pct: Optional[float] = None
    leg_pm: Optional[float] = None
    leg_acc: Optional[float] = None
    leg_abs_pct: Optional[float] = None
    leg_abs_pm: Optional[float] = None
    leg_def: Optional[float] = None

    # Striking: Distance
    dist_pct: Optional[float] = None
    dist_pm: Optional[float] = None
    dist_acc: Optional[float] = None
    dist_abs_pct: Optional[float] = None
    dist_abs_pm: Optional[float] = None
    dist_def: Optional[float] = None

    # Striking: Clinch
    clinch_pct: Optional[float] = None
    clinch_pm: Optional[float] = None
    clinch_acc: Optional[float] = None
    clinch_abs_pct: Optional[float] = None
    clinch_abs_pm: Optional[float] = None
    clinch_def: Optional[float] = None

    # Striking: Ground + position-aware
    ground_pct: Optional[float] = None
    ground_pm: Optional[float] = None
    ground_acc: Optional[float] = None
    ground_abs_pct: Optional[float] = None
    ground_abs_pm: Optional[float] = None
    ground_def: Optional[float] = None
    gnp15g: Optional[float] = None
    gnp_abs15g: Optional[float] = None

    # Knockdowns
    kd15: Optional[float] = None
    kd15s: Optional[float] = None
    kd_abs15: Optional[float] = None
    kd_abs15s: Optional[float] = None

    # Takedowns
    td15: Optional[float] = None
    td15s: Optional[float] = None
    td_acc: Optional[float] = None
    td_abs15: Optional[float] = None
    td_abs15s: Optional[float] = None
    td_def: Optional[float] = None

    # Control time
    ctrl15: Optional[float] = None
    ctrl15g: Optional[float] = None
    ctrl_abs15: Optional[float] = None
    ctrl_abs15g: Optional[float] = None

    # Submissions
    sub_att15: Optional[float] = None
    sub_att15g: Optional[float] = None
    sub_abs15: Optional[float] = None
    sub_abs15g: Optional[float] = None

    # Reversals
    rev15: Optional[float] = None
    rev_abs15: Optional[float] = None

    # Outcomes
    ko_wins: int = 0
    sub_wins: int = 0
    dec_wins: int = 0
    finish_rate: Optional[float] = None
    win_pct: Optional[float] = None
    avg_fight_sec: Optional[float] = None

    # Metadata
    computed_at: Optional[dt.datetime] = None

    @field_serializer("fighter_id", check_fields=False)
    @classmethod
    def serialize_fighter_id(cls, v):
        return str(v) if v is not None else None

    model_config = {"from_attributes": True}


class UFCFightStatsResponse(BigIntStr, UFCFightStatsBase):
    id: int
    created_at: dt.datetime
    updated_at: dt.datetime

    model_config = {"from_attributes": True}


