"""Storage. SQLite by default, PostgreSQL via DATABASE_URL."""
import os
from datetime import datetime, timezone
from sqlalchemy import (create_engine, Column, Integer, Float, String,
                        DateTime, Boolean, ForeignKey, select, desc, func)
from sqlalchemy.orm import declarative_base, Session

Base = declarative_base()
# e.g. postgresql+psycopg2://user:pass@localhost/containers
engine = create_engine(os.getenv("DATABASE_URL", "sqlite:///containers.db"))


class Container(Base):
    __tablename__ = "containers"
    container_id = Column(String(11), primary_key=True)


class Sighting(Base):
    __tablename__ = "sightings"
    id = Column(Integer, primary_key=True)
    container_id = Column(String(11), ForeignKey("containers.container_id"), index=True)
    checkpoint = Column(String(64))        # gate / crane / port name
    x = Column(Float)                      # metres, yard frame
    y = Column(Float)
    z_tier = Column(Integer)               # stack level, 1 = ground
    heading_deg = Column(Float)            # clockwise from north
    heading_ambiguous = Column(Boolean, default=True)
    ocr_conf = Column(Float)
    image_path = Column(String(256))
    seen_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


def init_db():
    Base.metadata.create_all(engine)


def add_sighting(**kw) -> int:
    with Session(engine) as s:
        if not s.get(Container, kw["container_id"]):
            s.add(Container(container_id=kw["container_id"]))
        row = Sighting(**kw)
        s.add(row)
        s.commit()
        return row.id


def clear_all():
    """Forget every container and sighting (a new video / camera starts from an empty map)."""
    with Session(engine) as s:
        s.query(Sighting).delete()
        s.query(Container).delete()
        s.commit()


def latest(container_id: str):
    with Session(engine) as s:
        return s.scalars(select(Sighting).where(Sighting.container_id == container_id)
                         .order_by(desc(Sighting.seen_at), desc(Sighting.id)).limit(1)).first()


def history(container_id: str):
    with Session(engine) as s:
        return list(s.scalars(select(Sighting).where(Sighting.container_id == container_id)
                              .order_by(Sighting.seen_at, Sighting.id)))


def latest_all():
    """Most recent sighting of every container."""
    with Session(engine) as s:
        ids = select(func.max(Sighting.id)).group_by(Sighting.container_id)
        return list(s.scalars(select(Sighting).where(Sighting.id.in_(ids))
                              .order_by(desc(Sighting.id))))


def recent(limit=30):
    with Session(engine) as s:
        return list(s.scalars(select(Sighting).order_by(desc(Sighting.id)).limit(limit)))