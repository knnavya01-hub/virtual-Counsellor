"""
Weaviate vector database for RAG
Stores embedded Sanskrit scriptures with metadata
"""

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType
from app.core.config import settings
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Global Weaviate client
weaviate_client = None

async def init_weaviate():
    """Initialize Weaviate connection and schema"""
    global weaviate_client
    
    try:
        # Connect to Weaviate
        if settings.WEAVIATE_API_KEY:
            weaviate_client = weaviate.connect_to_weaviate_cloud(
                cluster_url=settings.WEAVIATE_URL,
                auth_credentials=Auth.api_key(settings.WEAVIATE_API_KEY)
            )
        else:
            weaviate_client = weaviate.connect_to_local(
                host=settings.WEAVIATE_URL.replace("http://", "").replace("https://", "")
            )
        
        # Create schema if doesn't exist
        await _create_schema()
        
        logger.info("✅ Weaviate initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Weaviate initialization failed: {e}")
        raise

async def _create_schema():
    """Create Weaviate schema for Sanskrit scriptures"""
    try:
        # Check if class exists
        if weaviate_client.collections.exists(settings.WEAVIATE_CLASS_NAME):
            logger.info(f"Schema '{settings.WEAVIATE_CLASS_NAME}' already exists")
            return
        
        # Create collection with hybrid search capability
        weaviate_client.collections.create(
            name=settings.WEAVIATE_CLASS_NAME,
            vectorizer_config=Configure.Vectorizer.text2vec_openai(),
            properties=[
                Property(
                    name="source_name",
                    data_type=DataType.TEXT,
                    description="Scripture source (e.g., Bhagavad Gita)"
                ),
                Property(
                    name="source_type",
                    data_type=DataType.TEXT,
                    description="Type: vedas, upanishads, gita, puranas"
                ),
                Property(
                    name="chapter",
                    data_type=DataType.TEXT,
                    description="Chapter or section"
                ),
                Property(
                    name="verse_number",
                    data_type=DataType.TEXT,
                    description="Verse or sloka number"
                ),
                Property(
                    name="sanskrit_text",
                    data_type=DataType.TEXT,
                    description="Original Sanskrit text"
                ),
                Property(
                    name="english_translation",
                    data_type=DataType.TEXT,
                    description="Multi-LLM translated English text"
                ),
                Property(
                    name="emotions",
                    data_type=DataType.TEXT_ARRAY,
                    description="Emotion tags"
                ),
                Property(
                    name="situations",
                    data_type=DataType.TEXT_ARRAY,
                    description="Life situation tags"
                ),
                Property(
                    name="themes",
                    data_type=DataType.TEXT_ARRAY,
                    description="Spiritual theme tags"
                ),
                Property(
                    name="journey_stages",
                    data_type=DataType.TEXT_ARRAY,
                    description="User journey stages"
                ),
                Property(
                    name="chronological_order",
                    data_type=DataType.INT,
                    description="Curriculum learning order (1=oldest)"
                ),
                Property(
                    name="interpretation_context",
                    data_type=DataType.TEXT,
                    description="Context from older scriptures for interpretation"
                ),
            ]
        )
        
        logger.info(f"✅ Created Weaviate schema: {settings.WEAVIATE_CLASS_NAME}")
        
    except Exception as e:
        logger.error(f"Schema creation error: {e}")
        raise

async def close_weaviate():
    """Close Weaviate connection"""
    global weaviate_client
    if weaviate_client:
        weaviate_client.close()
        logger.info("Weaviate connection closed")

def get_weaviate_client():
    """Get global Weaviate client"""
    if not weaviate_client:
        raise RuntimeError("Weaviate client not initialized")
    return weaviate_client

async def add_scripture(
    source_name: str,
    source_type: str,
    sanskrit_text: str,
    english_translation: str,
    chapter: Optional[str] = None,
    verse_number: Optional[str] = None,
    emotions: List[str] = None,
    situations: List[str] = None,
    themes: List[str] = None,
    journey_stages: List[str] = None,
    chronological_order: int = 0,
    interpretation_context: str = ""
) -> str:
    """Add scripture to Weaviate vector store"""
    
    client = get_weaviate_client()
    collection = client.collections.get(settings.WEAVIATE_CLASS_NAME)
    
    properties = {
        "source_name": source_name,
        "source_type": source_type,
        "chapter": chapter or "",
        "verse_number": verse_number or "",
        "sanskrit_text": sanskrit_text,
        "english_translation": english_translation,
        "emotions": emotions or [],
        "situations": situations or [],
        "themes": themes or [],
        "journey_stages": journey_stages or [],
        "chronological_order": chronological_order,
        "interpretation_context": interpretation_context
    }
    
    # Add object and return UUID
    result = collection.data.insert(properties)
    
    logger.info(f"Added scripture: {source_name} - {verse_number}")
    return str(result)

async def hybrid_search(
    query: str,
    filters: Optional[Dict] = None,
    limit: int = None,
    alpha: float = 0.5
) -> List[Dict]:
    """
    Hybrid search combining semantic and keyword search
    
    Args:
        query: User's question
        filters: Metadata filters (emotions, themes, etc.)
        limit: Number of results (default from settings)
        alpha: Balance between semantic (1.0) and keyword (0.0) search
    
    Returns:
        List of matching scripture passages with metadata
    """
    
    client = get_weaviate_client()
    collection = client.collections.get(settings.WEAVIATE_CLASS_NAME)
    
    limit = limit or settings.RAG_TOP_K
    
    # Build filter
    where_filter = None
    if filters:
        # Example: filter by emotions or themes
        # Implement based on your filtering needs
        pass
    
    # Hybrid search
    response = collection.query.hybrid(
        query=query,
"""
Weaviate vector database for RAG
Stores embedded Sanskrit scriptures with metadata
"""

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType
from app.core.config import settings
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Global Weaviate client
weaviate_client = None

async def init_weaviate():
    """Initialize Weaviate connection and schema"""
    global weaviate_client
    
    try:
        # Connect to Weaviate
        if settings.WEAVIATE_API_KEY:
            weaviate_client = weaviate.connect_to_weaviate_cloud(
                cluster_url=settings.WEAVIATE_URL,
                auth_credentials=Auth.api_key(settings.WEAVIATE_API_KEY)
            )
        else:
            weaviate_client = weaviate.connect_to_local(
                host=settings.WEAVIATE_URL.replace("http://", "").replace("https://", "")
            )
        
        # Create schema if doesn't exist
        await _create_schema()
        
        logger.info("✅ Weaviate initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Weaviate initialization failed: {e}")
        raise

async def _create_schema():
    """Create Weaviate schema for Sanskrit scriptures"""
    try:
        # Check if class exists
        if weaviate_client.collections.exists(settings.WEAVIATE_CLASS_NAME):
            logger.info(f"Schema '{settings.WEAVIATE_CLASS_NAME}' already exists")
            return
        
        # Create collection with hybrid search capability
        weaviate_client.collections.create(
            name=settings.WEAVIATE_CLASS_NAME,
            vectorizer_config=Configure.Vectorizer.text2vec_openai(),
            properties=[
                Property(
                    name="source_name",
                    data_type=DataType.TEXT,
                    description="Scripture source (e.g., Bhagavad Gita)"
                ),
                Property(
                    name="source_type",
                    data_type=DataType.TEXT,
                    description="Type: vedas, upanishads, gita, puranas"
                ),
                Property(
                    name="chapter",
                    data_type=DataType.TEXT,
                    description="Chapter or section"
                ),
                Property(
                    name="verse_number",
                    data_type=DataType.TEXT,
                    description="Verse or sloka number"
                ),
                Property(
                    name="sanskrit_text",
                    data_type=DataType.TEXT,
                    description="Original Sanskrit text"
                ),
                Property(
                    name="english_translation",
                    data_type=DataType.TEXT,
                    description="Multi-LLM translated English text"
                ),
                Property(
                    name="emotions",
                    data_type=DataType.TEXT_ARRAY,
                    description="Emotion tags"
                ),
                Property(
                    name="situations",
                    data_type=DataType.TEXT_ARRAY,
                    description="Life situation tags"
                ),
                Property(
                    name="themes",
                    data_type=DataType.TEXT_ARRAY,
                    description="Spiritual theme tags"
                ),
                Property(
                    name="journey_stages",
                    data_type=DataType.TEXT_ARRAY,
                    description="User journey stages"
                ),
                Property(
                    name="chronological_order",
                    data_type=DataType.INT,
                    description="Curriculum learning order (1=oldest)"
                ),
                Property(
                    name="interpretation_context",
                    data_type=DataType.TEXT,
                    description="Context from older scriptures for interpretation"
                ),
            ]
        )
        
        logger.info(f"✅ Created Weaviate schema: {settings.WEAVIATE_CLASS_NAME}")
        
    except Exception as e:
        logger.error(f"Schema creation error: {e}")
        raise

async def close_weaviate():
    """Close Weaviate connection"""
    global weaviate_client
    if weaviate_client:
        weaviate_client.close()
        logger.info("Weaviate connection closed")

def get_weaviate_client():
    """Get global Weaviate client"""
    if not weaviate_client:
        raise RuntimeError("Weaviate client not initialized")
    return weaviate_client

async def add_scripture(
    source_name: str,
    source_type: str,
    sanskrit_text: str,
    english_translation: str,
    chapter: Optional[str] = None,
    verse_number: Optional[str] = None,
    emotions: List[str] = None,
    situations: List[str] = None,
    themes: List[str] = None,
    journey_stages: List[str] = None,
    chronological_order: int = 0,
    interpretation_context: str = ""
) -> str:
    """Add scripture to Weaviate vector store"""
    
    client = get_weaviate_client()
    collection = client.collections.get(settings.WEAVIATE_CLASS_NAME)
    
    properties = {
        "source_name": source_name,
        "source_type": source_type,
        "chapter": chapter or "",
        "verse_number": verse_number or "",
        "sanskrit_text": sanskrit_text,
        "english_translation": english_translation,
        "emotions": emotions or [],
        "situations": situations or [],
        "themes": themes or [],
        "journey_stages": journey_stages or [],
        "chronological_order": chronological_order,
        "interpretation_context": interpretation_context
    }
    
    # Add object and return UUID
    result = collection.data.insert(properties)
    
    logger.info(f"Added scripture: {source_name} - {verse_number}")
    return str(result)

async def hybrid_search(
    query: str,
    filters: Optional[Dict] = None,
    limit: int = None,
    alpha: float = 0.5
) -> List[Dict]:
    """
    Hybrid search combining semantic and keyword search
    
    Args:
        query: User's question
        filters: Metadata filters (emotions, themes, etc.)
        limit: Number of results (default from settings)
        alpha: Balance between semantic (1.0) and keyword (0.0) search
    
    Returns:
        List of matching scripture passages with metadata
    """
    
    client = get_weaviate_client()
    collection = client.collections.get(settings.WEAVIATE_CLASS_NAME)
    
    limit = limit or settings.RAG_TOP_K
    
    # Build filter
    where_filter = None
    if filters:
        # Example: filter by emotions or themes
        # Implement based on your filtering needs
        pass
    
    # Hybrid search
    response = collection.query.hybrid(
        query=query,
        alpha=alpha,
        limit=limit,
        return_metadata=["score", "distance"]
    )
    
    # Format results
    results = []
    for item in response.objects:
        results.append({
            "uuid": str(item.uuid),
            "source_name": item.properties.get("source_name"),
            "source_type": item.properties.get("source_type"),
            "chapter": item.properties.get("chapter"),
            "verse_number": item.properties.get("verse_number"),
            "sanskrit_text": item.properties.get("sanskrit_text"),
            "english_translation": item.properties.get("english_translation"),
            "emotions": item.properties.get("emotions", []),
            "situations": item.properties.get("situations", []),
            "themes": item.properties.get("themes", []),
            "journey_stages": item.properties.get("journey_stages", []),
            "chronological_order": item.properties.get("chronological_order"),
            "interpretation_context": item.properties.get("interpretation_context"),
            "score": item.metadata.score if hasattr(item.metadata, 'score') else 0
        })
    
    return results
