"""
Wisdom Service - Main RAG pipeline for answering user queries
Retrieves relevant scriptures and generates personalized responses
"""

import asyncio
from typing import Dict, List, Optional
import logging
from anthropic import AsyncAnthropic
from app.core.config import settings
from app.core.vector_store import hybrid_search
from app.services.crisis_detector import CrisisDetector
from app.services.memory_service import MemoryService

logger = logging.getLogger(__name__)

class WisdomService:
    """Main service for delivering spiritual wisdom to users"""
    
    def __init__(self):
        self.claude = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.crisis_detector = CrisisDetector()
        self.memory_service = MemoryService()
    
    async def get_wisdom(
        self,
        user_query: str,
        user_id: int,
        user_profile: Dict,
        conversation_history: List[Dict] = None
    ) -> Dict:
        """
        Main method: Get wisdom response for user query
        
        Args:
            user_query: User's question or situation
            user_id: User ID for personalization
            user_profile: User profile with journey stage, preferences
            conversation_history: Recent conversation context
        
        Returns:
            Complete wisdom response with scriptures, insights, visual prompts
        """
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # CRITICAL: Crisis Detection First
            is_crisis, confidence, keywords = self.crisis_detector.detect_crisis(user_query)
            
            if is_crisis:
                logger.warning(f"🚨 Crisis detected for user {user_id}")
                crisis_response = self.crisis_detector.get_crisis_response(
                    country=user_profile.get("country", "international")
                )
                
                # Log crisis event (handled by API endpoint)
                return {
                    "type": "crisis",
                    "crisis_response": crisis_response,
                    "confidence": confidence,
                    "detected_keywords": keywords
                }
            
            # Get short-term memory context
            memory_context = await self.memory_service.get_short_term_context(
                user_id, conversation_history
            )
            
            # Step 1: Retrieve relevant scriptures (RAG)
            scriptures = await self._retrieve_scriptures(
                user_query,
                user_profile.get("journey_stage"),
                user_profile.get("preferred_themes")
            )
            
            # Step 2: Rerank scriptures for relevance
            ranked_scriptures = await self._rerank_scriptures(
                user_query, scriptures, user_profile
            )
            
            # Step 3: Generate wisdom response using Claude
            wisdom_response = await self._generate_wisdom_response(
                user_query,
                ranked_scriptures,
                user_profile,
                memory_context
            )
            
            # Step 4: Hallucination check & relevance scoring
            quality_scores = await self._evaluate_response(
                user_query, wisdom_response, ranked_scriptures
            )
            
            # Step 5: Generate visual scene prompts
            visual_prompts = await self._generate_visual_prompts(
                wisdom_response, user_profile.get("journey_stage")
            )
            
            # Calculate response time
            response_time = asyncio.get_event_loop().time() - start_time
            
            result = {
                "type": "wisdom",
                "wisdom_text": wisdom_response,
                "scriptures_cited": ranked_scriptures[:3],  # Top 3
                "quality_scores": quality_scores,
                "visual_prompts": visual_prompts,
                "response_time_seconds": response_time,
                "personalization": {
                    "journey_stage": user_profile.get("journey_stage"),
                    "matched_themes": self._extract_matched_themes(ranked_scriptures)
                }
            }
            
            # Update short-term memory
            await self.memory_service.add_to_short_term(
                user_id, user_query, wisdom_response
            )
            
            logger.info(f"✅ Wisdom delivered to user {user_id} in {response_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Wisdom service error: {e}")
            raise
    
    async def _retrieve_scriptures(
        self,
        query: str,
        journey_stage: Optional[str] = None,
        preferred_themes: Optional[List[str]] = None
    ) -> List[Dict]:
        """Retrieve relevant scriptures from Weaviate"""
        
        # Build filters based on user profile
        filters = {}
        if journey_stage:
            filters["journey_stages"] = [journey_stage]
        if preferred_themes:
            filters["themes"] = preferred_themes
        
        # Hybrid search (semantic + keyword)
        scriptures = await hybrid_search(
            query=query,
            filters=filters,
            limit=settings.RAG_TOP_K,
            alpha=0.7  # Favor semantic search
        )
        
        logger.info(f"Retrieved {len(scriptures)} scriptures for query")
        return scriptures
    
    async def _rerank_scriptures(
        self,
        query: str,
        scriptures: List[Dict],
        user_profile: Dict
    ) -> List[Dict]:
        """Rerank scriptures for better relevance"""
        
        if not settings.RERANKING_ENABLED or len(scriptures) <= 1:
            return scriptures
        
        # Simple reranking based on:
        # 1. Semantic similarity score
        # 2. Journey stage match
        # 3. Theme match
        
        journey_stage = user_profile.get("journey_stage")
        preferred_themes = user_profile.get("preferred_themes", [])
        
        for scripture in scriptures:
            score = scripture.get("score", 0)
            
            # Boost for journey stage match
            if journey_stage in scripture.get("journey_stages", []):
                score += 0.15
            
            # Boost for theme match
            matched_themes = set(scripture.get("themes", [])) & set(preferred_themes)
            score += len(matched_themes) * 0.05
            
            scripture["reranked_score"] = score
        
        # Sort by reranked score
        ranked = sorted(scriptures, key=lambda x: x["reranked_score"], reverse=True)
        
        return ranked
    
    async def _generate_wisdom_response(
        self,
        user_query: str,
        scriptures: List[Dict],
        user_profile: Dict,
        memory_context: Dict
    ) -> str:
        """Generate personalized wisdom response using Claude"""
        
        # Build context from scriptures
        scripture_context = "\n\n".join([
            f"**{s['source_name']} {s['chapter']}:{s['verse_number']}**\n"
            f"Sanskrit: {s['sanskrit_text']}\n"
            f"Translation: {s['english_translation']}\n"
            f"Context: {s.get('interpretation_context', '')}"
            for s in scriptures[:3]  # Use top 3
        ])
        
        # Build user context
        user_context = f"""
User Profile:
- Journey Stage: {user_profile.get('journey_stage', 'searching')}
- Age: {user_profile.get('age', 'not specified')}
- Life Context: {user_profile.get('life_context', 'not specified')}
"""
        
        # Build conversation context
        recent_topics = memory_context.get("recent_topics", [])
        context_summary = memory_context.get("summary", "")
        
        prompt = f"""You are a wise spiritual guide channeling ancient Sanskrit wisdom. A seeker has come to you with a question.

{user_context}

Recent conversation context: {context_summary}
Recent topics discussed: {', '.join(recent_topics) if recent_topics else 'First interaction'}

Seeker's Question:
{user_query}

Relevant Sacred Scriptures:
{scripture_context}

Your Task:
1. Understand the seeker's deeper emotional and spiritual need
2. Draw wisdom from the scriptures provided
3. Speak in a warm, compassionate voice - as if the universe itself is responding
4. Make it personal and relevant to their life stage and journey
5. Keep it concise (3-5 paragraphs)
6. End with an actionable insight or reflection question

Respond with profound wisdom that feels both ancient and immediately applicable to modern life."""

        try:
            message = await self.claude.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                temperature=0.8,  # Slightly higher for warmth
                messages=[{"role": "user", "content": prompt}]
            )
            
            return message.content[0].text
            
        except Exception as e:
            logger.error(f"Claude wisdom generation error: {e}")
            raise
    
    async def _evaluate_response(
        self,
        query: str,
        response: str,
        scriptures: List[Dict]
    ) -> Dict:
        """Evaluate response quality: faithfulness, relevance, hallucination check"""
        
        # Simple evaluation scores (for production, use more sophisticated metrics)
        scores = {
            "faithfulness": 0.0,  # How grounded in scriptures
            "relevance": 0.0,  # How relevant to query
            "coherence": 0.0,  # Logical flow
            "hallucination_score": 0.0  # Lower is better
        }
        
        # Check if scripture references appear in response
        scripture
