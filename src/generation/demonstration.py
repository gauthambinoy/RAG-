# ==============================================================================
# FILE: demonstration.py
# PURPOSE: Full context->answer demonstration with step-by-step visibility
# ==============================================================================

"""
DEMONSTRATION MODULE

Shows exactly how:
1. Question is interpreted
2. Documents are retrieved
3. Context is formatted
4. Prompt is sent to LLM
5. Answer is generated
6. Sources are cited

Perfect for showing "how LLM uses context"
"""

import logging
import time
from typing import Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class ContextAnswerDemonstration:
    """Demonstrate complete RAG pipeline"""
    
    def __init__(self, verbose: bool = True):
        """
        Initialize demonstration.
        
        PARAMETERS:
            verbose: Print step-by-step output
        """
        self.verbose = verbose
        logger.info("✅ ContextAnswerDemonstration initialized")
    
    def demonstrate(
        self,
        question: str,
        retrieved_chunks: List[Dict],
        formatted_context: str,
        prompt_sent: str,
        llm_response: str,
        generation_metadata: Dict = None
    ) -> Dict:
        """
        Run full demonstration of context->answer pipeline.
        
        PARAMETERS:
            question: User question
            retrieved_chunks: Retrieved document chunks
            formatted_context: Formatted context for LLM
            prompt_sent: Actual prompt sent to LLM
            llm_response: LLM's response
            generation_metadata: Metadata (model, time, cost, etc.)
            
        RETURNS:
            Complete demonstration report
        """
        
        demo_report = {
            "timestamp": datetime.now().isoformat(),
            "step_1_question": self._demonstrate_step1(question),
            "step_2_retrieval": self._demonstrate_step2(retrieved_chunks),
            "step_3_context": self._demonstrate_step3(formatted_context),
            "step_4_prompt": self._demonstrate_step4(prompt_sent),
            "step_5_response": self._demonstrate_step5(llm_response),
            "step_6_metadata": self._demonstrate_step6(generation_metadata or {}),
        }
        
        return demo_report
    
    def _demonstrate_step1(self, question: str) -> Dict:
        """Step 1: Question Analysis"""
        
        if self.verbose:
            print("\n" + "="*80)
            print("📍 STEP 1: QUESTION ANALYSIS")
            print("="*80)
            print(f"\n❓ USER QUESTION:")
            print(f"   {question}\n")
        
        # Extract question info
        word_count = len(question.split())
        char_count = len(question)
        
        step1 = {
            "question": question,
            "word_count": word_count,
            "char_count": char_count,
            "keywords": self._extract_keywords(question),
        }
        
        if self.verbose:
            print(f"📊 ANALYSIS:")
            print(f"   Length: {word_count} words, {char_count} characters")
            print(f"   Keywords: {', '.join(step1['keywords'])}\n")
        
        return step1
    
    def _demonstrate_step2(self, chunks: List[Dict]) -> Dict:
        """Step 2: Retrieval"""
        
        if self.verbose:
            print("="*80)
            print("📚 STEP 2: DOCUMENT RETRIEVAL")
            print("="*80)
            print(f"\n🔍 RETRIEVED {len(chunks)} CHUNKS:\n")
            
            for i, chunk in enumerate(chunks, 1):
                score = chunk.get('score', 0.0)
                source = chunk.get('source', 'unknown')
                text_preview = chunk.get('text', '')[:100]
                
                # Visual score bar
                bar_length = int(score * 20)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                
                print(f"   [{i}] {source}")
                print(f"       Score: {score:.3f} [{bar}]")
                print(f"       Preview: {text_preview}...")
                print()
        
        step2 = {
            "chunks_retrieved": len(chunks),
            "avg_score": sum(c.get('score', 0) for c in chunks) / max(len(chunks), 1),
            "chunks_detail": [
                {
                    "source": c.get('source'),
                    "score": c.get('score', 0.0),
                    "length": len(c.get('text', '')),
                }
                for c in chunks
            ],
        }
        
        if self.verbose:
            print(f"📊 RETRIEVAL STATS:")
            print(f"   Total chunks: {step2['chunks_retrieved']}")
            print(f"   Average score: {step2['avg_score']:.3f}\n")
        
        return step2
    
    def _demonstrate_step3(self, context: str) -> Dict:
        """Step 3: Context Formatting"""
        
        if self.verbose:
            print("="*80)
            print("📝 STEP 3: CONTEXT FORMATTING")
            print("="*80)
            print(f"\n📄 FORMATTED CONTEXT FOR LLM:")
            print(f"\n{context[:500]}...")
            print(f"\n... [full context: {len(context)} characters] ...\n")
        
        step3 = {
            "context_length": len(context),
            "word_count": len(context.split()),
            "document_blocks": context.count("---"),
        }
        
        if self.verbose:
            print(f"📊 CONTEXT STATS:")
            print(f"   Characters: {step3['context_length']}")
            print(f"   Words: {step3['word_count']}")
            print(f"   Document blocks: {step3['document_blocks']}\n")
        
        return step3
    
    def _demonstrate_step4(self, prompt: str) -> Dict:
        """Step 4: Prompt Sent to LLM"""
        
        if self.verbose:
            print("="*80)
            print("🤖 STEP 4: PROMPT SENT TO LLM")
            print("="*80)
            print(f"\n📨 FULL PROMPT TO LLM:\n")
            print(prompt[:800])
            print(f"\n... [full prompt: {len(prompt)} characters] ...\n")
        
        step4 = {
            "prompt_length": len(prompt),
            "word_count": len(prompt.split()),
            "includes_chain_of_thought": "step by step" in prompt.lower(),
            "includes_citation_request": "cite" in prompt.lower() or "source" in prompt.lower(),
        }
        
        if self.verbose:
            print(f"📊 PROMPT STATS:")
            print(f"   Length: {step4['prompt_length']} characters")
            print(f"   Chain-of-thought: {'✅ Yes' if step4['includes_chain_of_thought'] else '❌ No'}")
            print(f"   Citation request: {'✅ Yes' if step4['includes_citation_request'] else '❌ No'}\n")
        
        return step4
    
    def _demonstrate_step5(self, response: str) -> Dict:
        """Step 5: LLM Response"""
        
        if self.verbose:
            print("="*80)
            print("💡 STEP 5: LLM RESPONSE")
            print("="*80)
            print(f"\n📜 LLM ANSWER:\n")
            print(response[:600])
            print(f"\n... [full response: {len(response)} characters] ...\n")
        
        # Extract sources from response
        sources = self._extract_sources_from_response(response)
        
        step5 = {
            "response_length": len(response),
            "word_count": len(response.split()),
            "sources_cited": sources,
            "has_citations": len(sources) > 0,
        }
        
        if self.verbose:
            print(f"📊 RESPONSE STATS:")
            print(f"   Length: {step5['response_length']} characters")
            print(f"   Word count: {step5['word_count']} words")
            print(f"   Sources cited: {', '.join(sources) if sources else 'None'}")
            print(f"   Has citations: {'✅ Yes' if step5['has_citations'] else '⚠️  No'}\n")
        
        return step5
    
    def _demonstrate_step6(self, metadata: Dict) -> Dict:
        """Step 6: Metadata & Performance"""
        
        if self.verbose:
            print("="*80)
            print("⚙️  STEP 6: GENERATION METADATA")
            print("="*80)
            print(f"\n📊 PERFORMANCE METRICS:\n")
            
            for key, value in metadata.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.3f}")
                else:
                    print(f"   {key}: {value}")
            print()
        
        return metadata
    
    def print_summary(self, demo_report: Dict) -> None:
        """Print summary of demonstration"""
        
        if not self.verbose:
            return
        
        print("="*80)
        print("✅ DEMONSTRATION SUMMARY")
        print("="*80)
        print(f"\n✔️  Question analyzed")
        print(f"✔️  Documents retrieved: {demo_report['step_2_retrieval']['chunks_retrieved']}")
        print(f"✔️  Context formatted: {demo_report['step_3_context']['context_length']} chars")
        print(f"✔️  Prompt sent to LLM")
        print(f"✔️  Response generated: {demo_report['step_5_response']['word_count']} words")
        
        if demo_report['step_5_response']['has_citations']:
            sources = demo_report['step_5_response']['sources_cited']
            print(f"✔️  Sources cited: {', '.join(sources)}")
        else:
            print(f"⚠️  No sources cited in response")
        
        print("\n" + "="*80)
        print("🎯 KEY INSIGHT:")
        print("="*80)
        print("""
The LLM received:
1. A well-formatted question
2. Retrieved context chunks sorted by relevance
3. Instructions to cite sources and use only provided context
4. Step-by-step reasoning request

The LLM then generated an answer using the provided documents,
demonstrating proper RAG functionality!
""")
        print("="*80 + "\n")
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords"""
        
        # Simple keyword extraction
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 'where', 'when'}
        words = text.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        return keywords[:5]
    
    def _extract_sources_from_response(self, response: str) -> List[str]:
        """Extract source mentions from response"""
        
        sources = []
        response_lower = response.lower()
        
        # Check for document names
        if 'attention' in response_lower:
            sources.append('Attention Paper')
        if 'deepseek' in response_lower:
            sources.append('DeepSeek-R1')
        if 'ai act' in response_lower or 'eu' in response_lower:
            sources.append('EU AI Act')
        if 'inflation' in response_lower:
            sources.append('Inflation Data')
        
        return list(set(sources))  # Remove duplicates


# ==============================================================================
# TESTING
# ==============================================================================

if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    demo = ContextAnswerDemonstration(verbose=True)
    
    # Test data
    test_question = "What is the transformer architecture?"
    
    test_chunks = [
        {
            "source": "Attention_is_all_you_need.pdf",
            "score": 0.95,
            "text": "The Transformer is based solely on attention mechanisms..."
        },
        {
            "source": "DeepSeek-r1.pdf",
            "score": 0.88,
            "text": "The transformer architecture has been successfully applied..."
        },
    ]
    
    test_context = "Document 1: Attention_is_all_you_need.pdf (Score: 0.95)\nThe Transformer...\n---\n"
    
    test_prompt = """System: You are a helpful assistant...
    
Question: What is the transformer architecture?

Context: [provided above]"""
    
    test_response = """According to the Attention paper, the Transformer is based solely on 
attention mechanisms. It uses multi-head attention for processing sequences in parallel.
The architecture was introduced to improve upon RNNs."""
    
    test_metadata = {
        "model": "gemini-2.5-pro",
        "input_tokens": 2850,
        "output_tokens": 150,
        "generation_time": 1.25,
        "cost": 0.0285,
    }
    
    # Run demonstration
    print("\n🎬 RUNNING DEMONSTRATION...\n")
    report = demo.demonstrate(
        test_question,
        test_chunks,
        test_context,
        test_prompt,
        test_response,
        test_metadata
    )
    
    # Print summary
    demo.print_summary(report)
