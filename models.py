from sqlalchemy import Column, Integer, String, Numeric, DateTime, Text, LargeBinary
from datetime import datetime
from database import Base

class RequestLog(Base):
    __tablename__ = "request_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    model = Column(String)
    prompt_tokens = Column(Integer)
    completion_tokens = Column(Integer)
    total_tokens = Column(Integer)
    cost = Column(Numeric(10, 6))
    source = Column(String, default="groq")


class SemanticCache(Base):
    __tablename__ = "semantic_cache"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    prompt = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    embedding = Column(LargeBinary, nullable=False)
   