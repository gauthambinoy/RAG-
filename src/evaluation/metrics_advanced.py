# ==============================================================================
# FILE: hallucination_metrics.py
# PURPOSE: Calculate hallucination rate metrics for evaluation
# ==============================================================================

"""
HALLUCINATION METRICS

Metrics:
1. Hallucination Rate - % of hallucinated content
2. Hallucination Status - GREEN/YELLOW/RED
3. Tracking over time - Monitor improvements
"""

import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class HallucinationMetrics:
    """Hallucination metrics container"""
    hallucination_rate: float  # 0.0-1.0
    hallucination_percent: float  # 0-100
    total_queries: int
    hallucinated_queries: int
    verified_queries: int
    status: str  # GREEN, YELLOW, RED
    status_message: str


class HallucinationRateCalculator:
    """Calculate hallucination rate across queries"""
    
    def __init__(self):
        """Initialize calculator"""
        logger.info("✅ HallucinationRateCalculator initialized")
    
    def calculate_rate(self, hallucination_scores: List[float]) -> HallucinationMetrics:
        """
        Calculate hallucination rate across multiple queries.
        
        PARAMETERS:
            hallucination_scores: List of hallucination scores (0.0-1.0) per query
            
        RETURNS:
            HallucinationMetrics object
        """
        
        if not hallucination_scores:
            logger.warning("⚠️  No hallucination scores provided")
            return HallucinationMetrics(
                hallucination_rate=0.0,
                hallucination_percent=0.0,
                total_queries=0,
                hallucinated_queries=0,
                verified_queries=0,
                status="UNKNOWN",
                status_message="No data"
            )
        
        total = len(hallucination_scores)
        
        # Count hallucinated queries (score > 0.2)
        hallucinated_count = sum(1 for score in hallucination_scores if score > 0.2)
        verified_count = total - hallucinated_count
        
        # Calculate average hallucination rate
        avg_hallucination = sum(hallucination_scores) / total
        
        # Determine status
        if avg_hallucination < 0.1:
            status = "GREEN"
            message = f"✅ EXCELLENT - Only {avg_hallucination*100:.1f}% hallucination"
        elif avg_hallucination < 0.25:
            status = "YELLOW"
            message = f"⚠️  GOOD - {avg_hallucination*100:.1f}% hallucination (acceptable)"
        else:
            status = "RED"
            message = f"❌ HIGH - {avg_hallucination*100:.1f}% hallucination (needs work)"
        
        metrics = HallucinationMetrics(
            hallucination_rate=avg_hallucination,
            hallucination_percent=round(avg_hallucination * 100, 1),
            total_queries=total,
            hallucinated_queries=hallucinated_count,
            verified_queries=verified_count,
            status=status,
            status_message=message
        )
        
        logger.info(f"\n📊 HALLUCINATION RATE ANALYSIS:")
        logger.info(f"   Total queries: {total}")
        logger.info(f"   Hallucination rate: {metrics.hallucination_percent:.1f}%")
        logger.info(f"   Status: {status}")
        logger.info(f"   Verified queries: {verified_count}/{total}")
        logger.info(f"   Hallucinated queries: {hallucinated_count}/{total}")
        logger.info(f"   Message: {message}")
        
        return metrics
    
    def generate_report(self, metrics: HallucinationMetrics) -> str:
        """Generate readable report"""
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║          HALLUCINATION RATE EVALUATION REPORT                ║
╚══════════════════════════════════════════════════════════════╝

📊 METRICS:
   Total Queries: {metrics.total_queries}
   Hallucination Rate: {metrics.hallucination_percent:.1f}%
   Status: [{metrics.status}] {metrics.status_message}
   
📈 BREAKDOWN:
   ✅ Verified queries: {metrics.verified_queries}/{metrics.total_queries}
   ❌ Hallucinated queries: {metrics.hallucinated_queries}/{metrics.total_queries}

🎯 INTERPRETATION:
   0-10%: Excellent (GREEN) ✅
   10-25%: Good (YELLOW) ⚠️
   25%+: High (RED) ❌
   
   Current level: {metrics.status_message}
"""
        return report


# ==============================================================================

class AccuracyMetricsCalculator:
    """Calculate BLEU, ROUGE, and accuracy metrics"""
    
    def __init__(self):
        """Initialize calculator"""
        try:
            from rouge_score import rouge_scorer
            self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'])
            self.has_rouge = True
            logger.info("✅ ROUGE scorer loaded")
        except:
            self.rouge = None
            self.has_rouge = False
            logger.warning("⚠️  ROUGE not available, using BLEU only")
        
        logger.info("✅ AccuracyMetricsCalculator initialized")
    
    def calculate_bleu(self, generated: str, reference: str) -> float:
        """
        Calculate BLEU score (n-gram overlap).
        
        RETURNS:
            Score 0.0-1.0 (1.0 = perfect match)
        """
        
        try:
            from nltk.translate.bleu_score import sentence_bleu
            from nltk.tokenize import word_tokenize
            
            reference_tokens = word_tokenize(reference.lower())
            generated_tokens = word_tokenize(generated.lower())
            
            # Calculate BLEU-4
            score = sentence_bleu([reference_tokens], generated_tokens, weights=(0.25, 0.25, 0.25, 0.25))
            return score
        except:
            # Fallback: simple word overlap
            ref_words = set(reference.lower().split())
            gen_words = set(generated.lower().split())
            overlap = len(ref_words & gen_words)
            total = max(len(ref_words), len(gen_words))
            return overlap / total if total > 0 else 0.0
    
    def calculate_rouge(self, generated: str, reference: str) -> Dict[str, float]:
        """
        Calculate ROUGE score (longest common subsequence).
        
        RETURNS:
            Dict with rouge1 and rougeL scores
        """
        
        if self.has_rouge:
            try:
                scores = self.rouge.score(reference, generated)
                return {
                    'rouge1': scores['rouge1'].fmeasure,
                    'rougeL': scores['rougeL'].fmeasure,
                }
            except:
                pass
        
        # Fallback: simple word overlap
        ref_words = set(reference.lower().split())
        gen_words = set(generated.lower().split())
        overlap = len(ref_words & gen_words)
        return {
            'rouge1': overlap / len(ref_words) if ref_words else 0.0,
            'rougeL': overlap / len(gen_words) if gen_words else 0.0,
        }
    
    def calculate_exact_match(self, generated: str, reference: str, key_phrases: List[str] = None) -> float:
        """
        Calculate exact match score (key points matched).
        
        PARAMETERS:
            generated: Generated answer
            reference: Reference answer
            key_phrases: Important phrases to match
            
        RETURNS:
            Score 0.0-1.0
        """
        
        if key_phrases:
            matched = sum(1 for phrase in key_phrases if phrase.lower() in generated.lower())
            score = matched / len(key_phrases)
        else:
            # Fallback: substring matching
            if reference.lower() in generated.lower():
                score = 1.0
            else:
                ref_words = set(reference.lower().split())
                gen_words = set(generated.lower().split())
                overlap = len(ref_words & gen_words)
                score = overlap / len(ref_words) if ref_words else 0.0
        
        return score
    
    def calculate_combined_accuracy(
        self,
        generated: str,
        reference: str,
        key_phrases: List[str] = None
    ) -> Dict:
        """
        Calculate all accuracy metrics and return combined score.
        
        RETURNS:
            Dict with all metrics and combined score
        """
        
        bleu = self.calculate_bleu(generated, reference)
        rouge = self.calculate_rouge(generated, reference)
        exact = self.calculate_exact_match(generated, reference, key_phrases)
        
        # Combined score (weighted average)
        combined = (bleu + rouge['rougeL'] + exact) / 3
        
        logger.info(f"\n📊 ACCURACY METRICS:")
        logger.info(f"   BLEU: {bleu:.3f}")
        logger.info(f"   ROUGE-L: {rouge['rougeL']:.3f}")
        logger.info(f"   Exact Match: {exact:.3f}")
        logger.info(f"   Combined: {combined:.3f}")
        
        return {
            'bleu': bleu,
            'rouge1': rouge['rouge1'],
            'rougeL': rouge['rougeL'],
            'exact_match': exact,
            'combined_accuracy': combined,
        }


# ==============================================================================

class QualityMetricsCalculator:
    """Calculate coherence, fluency, and relevance metrics"""
    
    def __init__(self):
        """Initialize calculator"""
        logger.info("✅ QualityMetricsCalculator initialized")
    
    def calculate_coherence(self, text: str) -> float:
        """
        Calculate semantic coherence between sentences.
        
        RETURNS:
            Score 0.0-1.0 (1.0 = perfect coherence)
        """
        
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            
            model = SentenceTransformer('all-MiniLM-L6-v2')
            sentences = text.split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) < 2:
                return 1.0
            
            embeddings = model.encode(sentences)
            
            # Calculate average similarity between consecutive sentences
            similarities = []
            for i in range(len(embeddings) - 1):
                sim = np.dot(embeddings[i], embeddings[i+1]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1]) + 1e-10
                )
                similarities.append(sim)
            
            return np.mean(similarities) if similarities else 1.0
        except:
            return 0.5  # Unknown
    
    def calculate_fluency(self, text: str) -> float:
        """
        Calculate fluency (grammatical correctness).
        
        RETURNS:
            Score 0.0-1.0
        """
        
        # Simple heuristics
        issues = 0
        
        # Check for common patterns
        if text.count('  ') > 0:  # Double spaces
            issues += 1
        if text.count(',,,') > 0:  # Triple commas
            issues += 1
        
        # Count sentence structure variety
        sentences = text.split('.')
        avg_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        # Ideal sentence length 10-20 words
        if 10 <= avg_length <= 20:
            fluency = 0.9
        elif 5 <= avg_length <= 30:
            fluency = 0.7
        else:
            fluency = 0.5
        
        # Penalize for issues
        fluency -= issues * 0.1
        return max(0.0, min(1.0, fluency))
    
    def calculate_relevance(self, text: str, question: str) -> float:
        """
        Calculate relevance to question.
        
        RETURNS:
            Score 0.0-1.0
        """
        
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            text_emb = model.encode(text)
            question_emb = model.encode(question)
            
            similarity = np.dot(text_emb, question_emb) / (
                np.linalg.norm(text_emb) * np.linalg.norm(question_emb) + 1e-10
            )
            
            return max(0.0, min(1.0, similarity))
        except:
            return 0.5
    
    def calculate_combined_quality(self, text: str, question: str) -> Dict:
        """Calculate all quality metrics"""
        
        coherence = self.calculate_coherence(text)
        fluency = self.calculate_fluency(text)
        relevance = self.calculate_relevance(text, question)
        
        combined = (coherence + fluency + relevance) / 3
        
        logger.info(f"\n📊 QUALITY METRICS:")
        logger.info(f"   Coherence: {coherence:.3f}")
        logger.info(f"   Fluency: {fluency:.3f}")
        logger.info(f"   Relevance: {relevance:.3f}")
        logger.info(f"   Combined: {combined:.3f}")
        
        return {
            'coherence': coherence,
            'fluency': fluency,
            'relevance': relevance,
            'combined_quality': combined,
        }


# ==============================================================================
# TESTING
# ==============================================================================

if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # Test hallucination rate
    print("\n🧪 TEST 1: Hallucination Rate Calculator")
    halluc_calc = HallucinationRateCalculator()
    scores = [0.05, 0.08, 0.12, 0.03, 0.15, 0.09, 0.07, 0.11, 0.04, 0.06]
    metrics = halluc_calc.calculate_rate(scores)
    print(halluc_calc.generate_report(metrics))
    
    # Test accuracy metrics
    print("\n🧪 TEST 2: Accuracy Metrics")
    acc_calc = AccuracyMetricsCalculator()
    ref = "The transformer architecture uses attention mechanisms for processing sequences."
    gen = "Transformers use attention for sequence processing."
    accuracy = acc_calc.calculate_combined_accuracy(gen, ref)
    print(f"   Accuracy metrics: {accuracy}")
    
    # Test quality metrics
    print("\n🧪 TEST 3: Quality Metrics")
    qual_calc = QualityMetricsCalculator()
    text = "The transformer is powerful. It uses attention. Attention works well. Results are good."
    question = "How does transformer attention work?"
    quality = qual_calc.calculate_combined_quality(text, question)
    print(f"   Quality metrics: {quality}")
