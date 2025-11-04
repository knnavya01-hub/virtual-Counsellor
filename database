"""
SQLAlchemy database models for PostgreSQL
User, Session, Feedback, Crisis, Scripture data
"""

from sqlalchemy import Column, Integer, String, Text, Float, Boolean, DateTime, JSON, ForeignKey, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import enum

Base = declarative_base()

class UserJourneyStage(str, enum.Enum):
    """User's spiritual journey stages"""
    TRIGGERED = "triggered"
    SEARCHING = "searching"
    SURRENDERING = "surrendering"
    AWAKENING = "awakening"
    DETACHED = "detached"
    EMPOWERED = "empowered"

class SubscriptionStatus(str, enum.Enum):
    """Subscription status"""
    FREE_TRIAL = "free_trial"
    ACTIVE = "active"
    EXPIRED = "expired"
    CANCELLED = "cancelled"

class User(Base):
    """User account and profile"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    
    # Profile information
    age = Column(Integer)
    life_context = Column(Text)  # User's current life situation
    personal_intro = Column(Text)  # For AI personalization
    
    # Journey tracking
    journey_stage = Column(Enum(UserJourneyStage), default=UserJourneyStage.SEARCHING)
    preferred_themes = Column(JSON, default=list)  # List of theme tags
    
    # Subscription
    subscription_status = Column(Enum(SubscriptionStatus), default=SubscriptionStatus.FREE_TRIAL)
    trial_start_date = Column(DateTime, default=datetime.utcnow)
    subscription_end_date = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime)
    
    # Relationships
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    feedbacks = relationship("Feedback", back_populates="user", cascade="all, delete-orphan")
    crisis_logs = relationship("CrisisLog", back_populates="user", cascade="all, delete-orphan")

class Conversation(Base):
    """User conversation sessions"""
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Conversation data
    user_query = Column(Text, nullable=False)
    user_emotion = Column(String(50))  # Detected emotion
    input_mode = Column(String(20))  # "text" or "voice"
    
    # AI Response
    ai_response = Column(Text)
    scriptures_cited = Column(JSON, default=list)  # List of scripture references
    response_time_seconds = Column(Float)
    
    # Context
    conversation_context = Column(JSON)  # Short-term memory context
    user_journey_snapshot = Column(JSON)  # User state at time of query
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="conversations")
    feedbacks = relationship("Feedback", back_populates="conversation", cascade="all, delete-orphan")

class Scripture(Base):
    """Sanskrit scripture metadata (for tracking what's in Weaviate)"""
    __tablename__ = "scriptures"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Source information
    source_name = Column(String(255), nullable=False, index=True)  # e.g., "Bhagavad Gita"
    source_type = Column(String(50), nullable=False)  # vedas, upanishads, gita, puranas
    chapter = Column(String(50))
    verse_number = Column(String(50))
    
    # Content
    sanskrit_text = Column(Text, nullable=False)
    english_translation = Column(Text)
    
    # Metadata tags
    emotions = Column(JSON, default=list)
    situations = Column(JSON, default=list)
    themes = Column(JSON, default=list)
    journey_stages = Column(JSON, default=list)
    
    # Translation pipeline
    gemini_translation = Column(Text)
    perplexity_verification = Column(Text)
    claude_interpretation = Column(Text)
    llama_curriculum_context = Column(Text)
    
    # Quality scores
    translation_quality_score = Column(Float)
    relevance_score = Column(Float)
    
    # Vector DB reference
    weaviate_id = Column(String(255), unique=True, index=True)
    
    # Copyright compliance
    is_public_domain = Column(Boolean, default=True)
    copyright_verified = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Feedback(Base):
    """User feedback on AI responses"""
    __tablename__ = "feedbacks"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    
    # Ratings (1-5 stars)
    overall_rating = Column(Integer)
    relevance_rating = Column(Integer)
    helpfulness_rating = Column(Integer)
    storytelling_rating = Column(Integer)
    
    # Qualitative feedback
    feedback_text = Column(Text)
    suggested_improvement = Column(Text)
    
    # Mood after response
    mood_before = Column(String(50))
    mood_after = Column(String(50))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="feedbacks")
    conversation = relationship("Conversation", back_populates="feedbacks")

class CrisisLog(Base):
    """Crisis detection and intervention logs"""
    __tablename__ = "crisis_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Crisis detection
    detected_keywords = Column(JSON, default=list)
    crisis_score = Column(Float)  # 0-1 confidence score
    user_message = Column(Text, nullable=False)
    
    # Intervention
    crisis_response_sent = Column(Text)
    helplines_displayed = Column(JSON, default=list)
    emergency_contact_attempted = Column(Boolean, default=False)
    
    # Follow-up
    user_returned = Column(Boolean, default=False)
    follow_up_message_sent = Column(Boolean, default=False)
    
    # Timestamps
    detected_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime)
    
    # Relationships
    user = relationship("User", back_populates="crisis_logs")

class VisualCreation(Base):
    """Generated visual content (6-scene stories)"""
    __tablename__ = "visual_creations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    
    # Content
    wisdom_theme = Column(String(255))
    scenes_data = Column(JSON)  # Array of 6 scenes with prompts and URLs
    
    # Generation details
    generation_service = Column(String(50))  # "leonardo", "stable_diffusion"
    total_generation_time = Column(Float)
    
    # Quality metrics
    image_quality_score = Column(Float)
    scene_coherence_score = Column(Float)
    
    # User engagement
    shared_to_social = Column(Boolean, default=False)
    user_rating = Column(Integer)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

class SystemMetrics(Base):
    """System performance and evaluation metrics"""
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Metric type and value
    metric_type = Column(String(100), nullable=False, index=True)
    metric_name = Column(String(255), nullable=False)
    metric_value = Column(Float, nullable=False)
    
    # Context
    context_data = Column(JSON)
    
    # Timestamp
    recorded_at = Column(DateTime, default=datetime.utcnow, index=True)
"""
Database connection management
PostgreSQL with SQLAlchemy async engine
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool
from app.core.config import settings
from app.models.models import Base
import logging

logger = logging.getLogger(__name__)

# Async engine
engine = None
AsyncSessionLocal = None

async def init_db():
    """Initialize database connection and create tables"""
    global engine, AsyncSessionLocal
    
    try:
        # Create async engine
        engine = create_async_engine(
            settings.DATABASE_URL,
            echo=settings.DEBUG,
            pool_size=settings.DB_POOL_SIZE,
            max_overflow=settings.DB_MAX_OVERFLOW,
            pool_pre_ping=True,
        )
        
        # Create session factory
        AsyncSessionLocal = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Create tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("✅ Database initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise

async def close_db():
    """Close database connections"""
    global engine
    if engine:
        await engine.dispose()
        logger.info("Database connections closed")

async def get_db() -> AsyncSession:
    """
    Dependency for getting database session
    Usage: db: AsyncSession = Depends(get_db)
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
