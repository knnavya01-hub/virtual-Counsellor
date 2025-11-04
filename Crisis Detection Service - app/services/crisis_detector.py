"""
Crisis Detection System
Multi-layer detection for suicidal ideation and mental health emergencies
"""

import re
from typing import Dict, List, Tuple
import logging
from transformers import pipeline
from app.core.config import settings

logger = logging.getLogger(__name__)

class CrisisDetector:
    """Detects crisis situations in user messages"""
    
    # Crisis keywords and phrases
    CRISIS_KEYWORDS = [
        "suicide", "kill myself", "end it all", "no reason to live",
        "want to die", "better off dead", "can't go on", "ending my life",
        "say goodbye", "nobody cares", "world without me", "hurt myself",
        "harm myself", "take my life", "not worth living"
    ]
    
    CRISIS_PHRASES = [
        "give up on life", "tired of living", "end the pain",
        "nothing to live for", "everyone would be better", "final goodbye",
        "can't take it anymore", "too much to bear"
    ]
    
    EMOTIONAL_MARKERS = [
        "hopeless", "worthless", "unbearable pain", "give up",
        "no way out", "desperate", "can't escape", "trapped"
    ]
    
    def __init__(self):
        """Initialize crisis detector with optional ML model"""
        self.enabled = settings.CRISIS_DETECTION_ENABLED
        
        # Try to load a fine-tuned model for crisis detection
        # For production, you'd fine-tune BERT/RoBERTa on crisis text
        try:
            self.classifier = pipeline(
                "text-classification",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            logger.info("Crisis detection ML model loaded")
        except Exception as e:
            logger.warning(f"ML model not available, using rule-based detection: {e}")
            self.classifier = None
    
    def detect_crisis(self, text: str) -> Tuple[bool, float, List[str]]:
        """
        Detect crisis indicators in text
        
        Returns:
            (is_crisis, confidence_score, detected_keywords)
        """
        if not self.enabled:
            return False, 0.0, []
        
        text_lower = text.lower()
        detected_keywords = []
        
        # Rule-based detection
        keyword_score = self._check_keywords(text_lower, detected_keywords)
        phrase_score = self._check_phrases(text_lower, detected_keywords)
        emotional_score = self._check_emotional_markers(text_lower, detected_keywords)
        
        # ML-based detection (if available)
        ml_score = 0.0
        if self.classifier:
            try:
                result = self.classifier(text[:512])[0]  # Truncate for model
                if result['label'] == 'NEGATIVE':
                    ml_score = result['score'] * 0.5  # Weight adjustment
            except Exception as e:
                logger.error(f"ML classification error: {e}")
        
        # Combined confidence score
        confidence = min(
            (keyword_score * 0.4 + phrase_score * 0.3 + 
             emotional_score * 0.2 + ml_score * 0.1),
            1.0
        )
        
        # Crisis threshold
        is_crisis = confidence >= 0.6
        
        if is_crisis:
            logger.warning(
                f"🚨 CRISIS DETECTED - Confidence: {confidence:.2f}, "
                f"Keywords: {detected_keywords}"
            )
        
        return is_crisis, confidence, detected_keywords
    
    def _check_keywords(self, text: str, detected: List[str]) -> float:
        """Check for direct crisis keywords"""
        found_count = 0
        for keyword in self.CRISIS_KEYWORDS:
            if keyword in text:
                detected.append(keyword)
                found_count += 1
        
        # Return normalized score (0-1)
        return min(found_count / 3, 1.0)  # 3+ keywords = max score
    
    def _check_phrases(self, text: str, detected: List[str]) -> float:
        """Check for crisis phrases"""
        found_count = 0
        for phrase in self.CRISIS_PHRASES:
            if phrase in text:
                detected.append(phrase)
                found_count += 1
        
        return min(found_count / 2, 1.0)  # 2+ phrases = max score
    
    def _check_emotional_markers(self, text: str, detected: List[str]) -> float:
        """Check for emotional distress markers"""
        found_count = 0
        for marker in self.EMOTIONAL_MARKERS:
            if marker in text:
                detected.append(marker)
                found_count += 1
        
        return min(found_count / 4, 1.0)  # 4+ markers = max score
    
    def get_crisis_response(self, country: str = "international") -> Dict:
        """
        Get crisis intervention message with appropriate helplines
        
        Args:
            country: User's country for localized helplines
        
        Returns:
            Crisis response message with helplines
        """
        helplines = settings.CRISIS_HELPLINES
        
        # Determine country-specific helpline
        country_lower = country.lower()
        primary_helpline = helplines.get(country_lower, helplines["international"])
        
        response = {
            "is_crisis": True,
            "message": {
                "opening": "I sense you're in deep pain right now. You matter deeply, and your life has profound value.",
                "urgency": "Please reach out to someone who can help immediately:",
                "primary_helpline": {
                    "country": country,
                    "number": primary_helpline
                },
                "additional_helplines": [
                    {
                        "name": "International Crisis Line",
                        "number": helplines["international"]
                    },
                    {
                        "name": "Crisis Text Line",
                        "number": helplines["text"]
                    }
                ],
                "reassurance": "You are not alone. There are people who want to help you through this darkness.",
                "wisdom_note": "The ancient scriptures teach that even the darkest nights end, and the sun rises again. Please stay. The universe needs you.",
                "immediate_action": "If you're in immediate danger, please call emergency services (911, 112, or your local emergency number) or go to the nearest emergency room."
            },
            "background_color": "#2C1810",  # Warm, comforting brown
            "accent_color": "#FFD700",  # Gold - hope and light
            "show_normal_response": False
        }
        
        return response
    
    def get_post_crisis_message(self) -> Dict:
        """Message for users returning after crisis detection"""
        return {
            "message": "I'm glad you're still here. Your strength in facing another day is remarkable.",
            "tone": "gentle and supportive",
            "offer": "I'm here to listen and offer wisdom whenever you need it. How are you feeling right now?",
            "resources_available": True
        }


def log_crisis_event(
    user_id: int,
    message: str,
    confidence: float,
    keywords: List[str]
) -> Dict:
    """
    Log crisis event for tracking and analysis
    
    Returns:
        Log entry data
    """
    log_entry = {
        "user_id": user_id,
        "detected_at": "timestamp",  # Will be set by DB
        "message_excerpt": message[:200],  # First 200 chars
        "confidence_score": confidence,
        "detected_keywords": keywords,
        "intervention_sent": True
    }
    
    logger.critical(
        f"🚨 CRISIS EVENT - User {user_id} - Confidence: {confidence:.2f}"
    )
    
    return log_entry
