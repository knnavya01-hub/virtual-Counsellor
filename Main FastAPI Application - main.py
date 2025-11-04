"""
Ancient Wisdom Guidance App - Main FastAPI Application
Handles API routing, middleware, and core orchestration
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
from datetime import datetime

# Import routers (we'll create these)
from app.api.v1 import auth, wisdom, memory, crisis, feedback, visual
from app.core.config import settings
from app.core.database import init_db, close_db
from app.core.vector_store import init_weaviate, close_weaviate
from app.services.crisis_detector import CrisisDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources"""
    logger.info("🕉️ Ancient Wisdom App Starting...")
    
    # Startup
    await init_db()
    await init_weaviate()
    
    # Initialize crisis detector
    app.state.crisis_detector = CrisisDetector()
    
    logger.info("✅ All systems initialized")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down gracefully...")
    await close_db()
    await close_weaviate()
    logger.info("✅ Cleanup complete")

# Initialize FastAPI app
app = FastAPI(
    title="Ancient Wisdom Guidance API",
    description="Personalized spiritual guidance through Sanskrit scriptures",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "The universe encountered an unexpected disturbance",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Ancient Wisdom API",
        "version": "1.0.0"
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with cosmic welcome"""
    return {
        "message": "Welcome to the Ancient Wisdom API",
        "tagline": "Where cosmic consciousness meets digital transcendence",
        "docs": "/docs",
        "health": "/health"
    }

# Include API routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(wisdom.router, prefix="/api/v1/wisdom", tags=["Wisdom"])
app.include_router(memory.router, prefix="/api/v1/memory", tags=["Memory"])
app.include_router(crisis.router, prefix="/api/v1/crisis", tags=["Crisis Detection"])
app.include_router(feedback.router, prefix="/api/v1/feedback", tags=["Feedback"])
app.include_router(visual.router, prefix="/api/v1/visual", tags=["Visual Creation"])

# Middleware for request logging
@app.middleware("http")
async def log_requests(request, call_next):
    """Log all incoming requests"""
    start_time = datetime.utcnow()
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = (datetime.utcnow() - start_time).total_seconds()
    
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Duration: {duration:.3f}s"
    )
    
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
"""
Configuration management using Pydantic settings
Environment variables and application settings
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    # Application
    APP_NAME: str = "Ancient Wisdom API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    # Database - PostgreSQL
    DATABASE_URL: str
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    
    # Vector Database - Weaviate
    WEAVIATE_URL: str
    WEAVIATE_API_KEY: Optional[str] = None
    WEAVIATE_CLASS_NAME: str = "SanskritScripture"
    
    # Redis - Short-term memory
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_MAX_CONNECTIONS: int = 50
    
    # JWT Authentication
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    
    # LLM API Keys
    ANTHROPIC_API_KEY: str  # Claude
    GOOGLE_API_KEY: str  # Gemini
    PERPLEXITY_API_KEY: str
    OPENAI_API_KEY: str
    
    # Image Generation
    LEONARDO_API_KEY: Optional[str] = None
    CANVA_API_KEY: Optional[str] = None
    
    # Twilio for Crisis Management
    TWILIO_ACCOUNT_SID: Optional[str] = None
    TWILIO_AUTH_TOKEN: Optional[str] = None
    TWILIO_PHONE_NUMBER: Optional[str] = None
    
    # Crisis Helpline Numbers (by country)
    CRISIS_HELPLINES: dict = {
        "international": "988",
        "india": "91529 87821",
        "uk": "116 123",
        "australia": "13 11 14",
        "text": "HOME to 741741"
    }
    
    # RAG Configuration
    RAG_TOP_K: int = 5
    RAG_SIMILARITY_THRESHOLD: float = 0.7
    RERANKING_ENABLED: bool = True
    
    # LLM Pipeline Configuration
    MAX_TOKENS_PER_STAGE: int = 2000
    TEMPERATURE: float = 0.7
    PIPELINE_TIMEOUT: int = 120  # seconds
    
    # Memory Configuration
    SHORT_TERM_MEMORY_TTL: int = 3600  # 1 hour in seconds
    LONG_TERM_MEMORY_THRESHOLD: int = 5  # conversations before consolidation
    
    # Safety Configuration
    TOXICITY_THRESHOLD: float = 0.1
    HALLUCINATION_CHECK_ENABLED: bool = True
    CRISIS_DETECTION_ENABLED: bool = True
    
    # Evaluation Metrics
    LANGSMITH_API_KEY: Optional[str] = None
    LANGSMITH_PROJECT: str = "ancient-wisdom-app"
    METRICS_ENABLED: bool = True
    
    # File Storage
    UPLOAD_DIR: str = "data/uploads"
    PROCESSED_DIR: str = "data/processed"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000
    
    # Subscription
    FREE_TRIAL_DAYS: int = 30
    MONTHLY_PRICE_INR: float = 11.1
    YEARLY_PRICE_INR: float = 111.0
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

# Global settings instance
settings = get_settings()

# Sanskrit scriptures metadata for curriculum learning
SCRIPTURE_CHRONOLOGY = {
    "vedas": {
        "order": 1,
        "period": "1500-500 BCE",
        "texts": ["Rigveda", "Yajurveda", "Samaveda", "Atharvaveda"]
    },
    "upanishads": {
        "order": 2,
        "period": "800-200 BCE",
        "texts": ["Isha", "Kena", "Katha", "Mundaka", "Mandukya"]
    },
    "bhagavad_gita": {
        "order": 3,
        "period": "400-200 BCE",
        "texts": ["Bhagavad Gita"]
    },
    "puranas": {
        "order": 4,
        "period": "300-1500 CE",
        "texts": ["Vishnu Purana", "Shiva Purana", "Bhagavata Purana"]
    }
}

# Emotion and context tags for metadata
EMOTION_TAGS = [
    "fear", "guilt", "shame", "anger", "confusion", "grief",
    "love", "longing", "loneliness", "peace", "joy", "gratitude"
]

SITUATION_TAGS = [
    "breakup", "betrayal", "career_block", "illness", "loss",
    "destiny_vs_freewill", "decision_making", "transformation"
]

THEME_TAGS = [
    "dharma", "karma", "detachment", "surrender", "faith",
    "ego", "true_love", "illusion", "acceptance", "action"
]

JOURNEY_STAGES = [
    "triggered", "searching", "surrendering",
    "awakening", "detached", "empowered"
]
