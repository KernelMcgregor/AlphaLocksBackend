"""Tables for the markets feature track. Same Base / schema conventions as app.models.ufc."""
import datetime as dt

from sqlalchemy import BigInteger, Boolean, Date, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models.base import TimestampMixin
from app.models.ufc import UFC_SCHEMA, _fk
