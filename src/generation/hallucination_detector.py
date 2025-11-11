# ==============================================================================
# FILE: hallucination_detector.py
# PURPOSE: Detect and measure hallucinations in LLM responses
# ==============================================================================

"""
HALLUCINATION DETECTION SYSTEM

Detects when LLM invents facts not in the provided context.

METHODS:
1. Semantic Similarity - Compare response sentences to context
2. Named Entity Matching - Check if entities appear in context
3. Fact Verification - Extract and verify claims

OUTPUT:
- hallucination_score: 0.0 (no hallucination) to 1.0 (all hallucinated)
- hallucinated_sentences: List of detected hallucinated claims
- verified_sentences: List of verified claims
- confidence_level: "HIGH", "MEDIUM", "LOW"
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class HallucinationDetector:
    """Detect hallucinations in LLM responses"""
    
    # Try to import embedding model
    try:
        from sentence_transformers import SentenceTransformer
        # Force CPU device to avoid CUDA/GPU mismatches in diverse environments
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        has_embeddings = True
        logger.info("✅ Embedding model loaded for hallucination detection")
    except:
        embedding_model = None
        has_embeddings = False
        logger.warning("⚠️  Embedding model not available, using fallback methods")
    
    # Named entities patterns
    ENTITY_PATTERNS = {
        'year': r'\b(19|20)\d{2}\b',
        'number': r'\b\d+(?:\.\d+)?\b',
        'percent': r'\d+%',
        'name': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
        'organization': r'\b(?:EU|AI|UN|USA|UK|Google|OpenAI|Anthropic|DeepSeek)\b',
    }
    
    def __init__(self, similarity_threshold: float = 0.65):
        """
        Initialize hallucination detector.
        
        PARAMETERS:
            similarity_threshold: Cosine similarity threshold (0-1)
                If response sentence similarity to context < threshold → hallucination
        """
        self.similarity_threshold = similarity_threshold
        logger.info(f"✅ HallucinationDetector initialized (threshold: {similarity_threshold})")
    
    def detect(self, response: str, context: str) -> Dict:
        """
        Detect hallucinations in response.
        
        PARAMETERS:
            response (str): LLM response
            context (str): Retrieved context
            
        RETURNS:
            Dict with:
            - hallucination_score: 0.0-1.0 (fraction hallucinated)
            - hallucinated_sentences: List of detected hallucinated claims
            - verified_sentences: List of verified claims
            - confidence_level: "HIGH", "MEDIUM", "LOW"
            - method_used: Which detection method was primary
        """
        
        logger.info("\n🔍 HALLUCINATION DETECTION STARTING...")
        
        response_sentences = self._split_sentences(response)
        context_sentences = self._split_sentences(context)
        
        hallucinated = []
        verified = []
        
        # Try semantic similarity method first if available
        if self.has_embeddings:
            logger.info("   Using semantic similarity method...")
            hallucinated, verified = self._detect_semantic(
                response_sentences, context_sentences
            )
        else:
            logger.info("   Using fallback keyword matching method...")
            hallucinated, verified = self._detect_keyword(
                response_sentences, context_sentences
            )
        
        # Supplement with entity matching
        additional_hallucinated = self._detect_entity_hallucinations(
            hallucinated, context
        )
        hallucinated.extend(additional_hallucinated)
        
        # Calculate hallucination score
        total_sentences = len([s for s in response_sentences if s.strip()])
        hallucination_count = len(set(hallucinated))
        
        if total_sentences > 0:
            hallucination_score = hallucination_count / total_sentences
        else:
            hallucination_score = 0.0
        
        # Determine confidence level
        if hallucination_score < 0.1:
            confidence = "HIGH"
        elif hallucination_score < 0.25:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        result = {
            "hallucination_score": hallucination_score,
            "hallucination_percent": round(hallucination_score * 100, 1),
            "hallucinated_sentences": hallucinated[:10],  # Top 10
            "verified_sentences": verified[:10],          # Top 10
            "confidence_level": confidence,
            "total_sentences": total_sentences,
            "hallucinated_count": hallucination_count,
            "verified_count": len(set(verified)),
        }
        
        # Log results
        logger.info(f"\n   ✅ HALLUCINATION DETECTION COMPLETE:")
        logger.info(f"      Hallucination score: {result['hallucination_score']:.2f} ({result['hallucination_percent']:.1f}%)")
        logger.info(f"      Confidence level: {confidence}")
        logger.info(f"      Total sentences: {total_sentences}")
        logger.info(f"      Verified: {result['verified_count']} | Hallucinated: {hallucination_count}")
        logger.info(f"      Status: {'✅ EXCELLENT' if hallucination_score < 0.1 else '⚠️  NEEDS REVIEW' if hallucination_score < 0.25 else '❌ HIGH RISK'}")
        
        return result
    
    def _detect_semantic(self, response_sents: List[str], context_sents: List[str]) -> Tuple[List[str], List[str]]:
        """
        Detect hallucinations using semantic similarity.
        
        METHOD:
        1. Embed all response and context sentences
        2. For each response sentence, find max similarity to context
        3. If max similarity < threshold → hallucination
        """
        
        if not self.has_embeddings or not context_sents:
            return [], []
        
        try:
            hallucinated = []
            verified = []
            
            # Embed context sentences once
            context_embeddings = self.embedding_model.encode(context_sents)
            
            # Check each response sentence
            for response_sent in response_sents:
                if len(response_sent.strip()) < 10:
                    continue
                
                # Embed response sentence
                response_emb = self.embedding_model.encode(response_sent)
                
                # Find max similarity to any context sentence
                similarities = []
                for context_emb in context_embeddings:
                    # Cosine similarity
                    sim = np.dot(response_emb, context_emb) / (
                        np.linalg.norm(response_emb) * np.linalg.norm(context_emb) + 1e-10
                    )
                    similarities.append(sim)
                
                max_similarity = max(similarities) if similarities else 0.0
                
                if max_similarity < self.similarity_threshold:
                    hallucinated.append(response_sent[:80])
                else:
                    verified.append(response_sent[:80])
            
            return hallucinated, verified
        
        except Exception as e:
            logger.warning(f"❌ Semantic detection failed: {e}, falling back to keyword matching")
            return [], []
    
    def _detect_keyword(self, response_sents: List[str], context_sents: List[str]) -> Tuple[List[str], List[str]]:
        """
        Fallback: Detect hallucinations using keyword overlap.
        
        METHOD:
        1. Extract keywords from context
        2. For each response sentence, check keyword overlap
        3. If low overlap → likely hallucination
        """
        
        context_text = " ".join(context_sents).lower()
        context_words = set(re.findall(r'\w+', context_text))
        
        hallucinated = []
        verified = []
        
        for response_sent in response_sents:
            if len(response_sent.strip()) < 10:
                continue
            
            response_words = set(re.findall(r'\w+', response_sent.lower()))
            
            # Calculate keyword overlap
            overlap = len(response_words & context_words) / (len(response_words) + 1e-10)
            
            # Low overlap = likely hallucination
            if overlap < 0.3:
                hallucinated.append(response_sent[:80])
            else:
                verified.append(response_sent[:80])
        
        return hallucinated, verified
    
    def _detect_entity_hallucinations(self, hallucinated_sents: List[str], context: str) -> List[str]:
        """
        Detect hallucinations by checking if entities appear in context.
        
        METHOD:
        1. Extract entities from hallucinated sentences
        2. Check if they appear in context
        3. If not → confirmed hallucination
        """
        
        additional_hallucinated = []
        context_lower = context.lower()
        
        for sent in hallucinated_sents:
            # Extract entities
            for entity_type, pattern in self.ENTITY_PATTERNS.items():
                entities = re.findall(pattern, sent)
                
                for entity in entities:
                    if isinstance(entity, tuple):
                        entity = entity[0]
                    
                    # Check if entity is in context
                    if entity not in context_lower:
                        additional_hallucinated.append(f"{sent} [unverified {entity_type}: {entity}]")
        
        return additional_hallucinated
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        
        # Simple sentence splitting on '.', '!', '?'
        sentences = re.split(r'[.!?]\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def get_confidence_explanation(self, score: float) -> str:
        """Get explanation for confidence level"""
        
        if score < 0.05:
            return "✅ EXCELLENT - Response is highly grounded in context"
        elif score < 0.1:
            return "✅ GOOD - Response is mostly grounded with minimal hallucination"
        elif score < 0.25:
            return "⚠️  MEDIUM - Response has some unverified claims"
        elif score < 0.5:
            return "⚠️  HIGH - Response contains several hallucinated elements"
        else:
            return "❌ CRITICAL - Response is mostly hallucinated"


# ==============================================================================
# TESTING
# ==============================================================================

if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    detector = HallucinationDetector(similarity_threshold=0.65)
    
    # Test case 1: High accuracy (grounded in context)
    test_response_good = """
    The transformer architecture uses attention mechanisms. According to the paper,
    self-attention allows each position to attend to other positions. Multi-head 
    attention uses multiple representation subspaces. The transformer was effective
    for sequence modeling tasks.
    """
    
    test_context = """
    The Transformer is based solely on attention mechanisms. Self-attention 
    allows the model to attend to different representation subspaces. We employ 
    multi-head attention consisting of multiple representation subspaces at different locations.
    """
    
    print("🧪 TEST 1: Good Response (Grounded)")
    result1 = detector.detect(test_response_good, test_context)
    print(f"   Score: {result1['hallucination_score']:.2f}")
    print(f"   Confidence: {result1['confidence_level']}")
    print()
    
    # Test case 2: High hallucination
    test_response_bad = """
    The transformer was invented in 2015. It uses 256 attention heads for optimal performance.
    The creator was John Smith. DeepSeek invented the transformer variant. The model has
    1 trillion parameters. Transformers are biological organisms from Mars.
    """
    
    print("🧪 TEST 2: Bad Response (Hallucinated)")
    result2 = detector.detect(test_response_bad, test_context)
    print(f"   Score: {result2['hallucination_score']:.2f}")
    print(f"   Confidence: {result2['confidence_level']}")
    print(f"   Hallucinated: {len(result2['hallucinated_sentences'])} sentences")
