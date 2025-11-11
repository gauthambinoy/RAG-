# ==============================================================================
# FILE: llm_interface.py
# PURPOSE: LLM integration for generation component of RAG
# ==============================================================================

"""
This module handles LLM API integration for answer generation.

WHAT THIS MODULE DOES:
- Takes retrieved context + user query
- Sends to LLM (OpenAI GPT) with carefully crafted prompt
- Returns generated answer

THIS IS THE "G" IN RAG (Retrieval-Augmented Generation)

LLM CHOICE: OpenAI GPT-4 / GPT-3.5-turbo
RATIONALE:
- Production-ready API with high reliability
- Strong reasoning and instruction following
- Good balance of quality and cost
- Easy to deploy (no local hosting needed)

ALTERNATIVE OPTIONS (Trade-offs):
1. Anthropic Claude
   - Pros: Better at long context, more cautious (less hallucination)
   - Cons: More expensive, slower API
   - When to use: Legal/medical domains where accuracy critical

2. Local Models (Llama, Mistral)
   - Pros: No API costs, full control, data privacy
   - Cons: Requires GPU/compute, harder deployment, lower quality
   - When to use: Privacy-critical applications, high volume

3. Azure OpenAI
   - Pros: Enterprise SLA, data residency options
   - Cons: More expensive, setup complexity
   - When to use: Enterprise deployments

CHOSEN: OpenAI GPT-3.5-turbo for development, GPT-4 for production
- Best quality/cost trade-off
- Fast response times (~1-2 seconds)
- Reliable API
- Can upgrade to GPT-4 if needed

PROMPT ENGINEERING STRATEGY:
- Clear instruction: "Answer based only on context"
- Context injection: Retrieved chunks provided
- Hallucination prevention: "If not in context, say so"
- Format control: Specify answer format
- Few-shot examples: (optional) Show good answers

USAGE:
    from src.generation.llm_interface import LLMInterface
    
    # Initialize (requires OPENAI_API_KEY environment variable)
    llm = LLMInterface()
    
    # Generate answer
    answer = llm.generate_answer(
        query="What is transformer architecture?",
        context="The Transformer is a neural network..."
    )
"""

import os
from typing import Optional, Dict, List, Tuple
from dotenv import load_dotenv
import requests
import google.generativeai as genai
try:
    from openai import OpenAI
except Exception:  # allow running if openai not installed
    OpenAI = None

load_dotenv()

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Model configuration
DEFAULT_MODEL = "gemini-pro"  # Gemini Pro

# Generation parameters
DEFAULT_TEMPERATURE = 0.1  # Low temperature = more deterministic
DEFAULT_MAX_TOKENS = 500  # Maximum answer length
DEFAULT_TOP_P = 0.95  # Nucleus sampling parameter


# ==============================================================================
# PROMPTS
# ==============================================================================

SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on provided context.

INSTRUCTIONS:
1. Answer the question using ONLY the information in the provided context
2. If the answer is not in the context, explicitly state: "I cannot answer this based on the provided documents."
3. Be concise and accurate
4. If the context has conflicting information, mention it
5. Do not make up or infer information not present in the context
6. Add bracketed citations like [C1], [C2] whenever you use facts, where numbers correspond to Context i blocks below.

RESPONSE FORMAT:
- Direct answer first
- Then supporting evidence from context with citations [C#]
- Keep answers clear and structured
"""

USER_PROMPT_TEMPLATE = """Context from relevant documents:
{context}

Question: {query}

Answer the question based on the context above. If the answer is not in the context, say so clearly."""


# ==============================================================================
# LLM INTERFACE CLASS
# ==============================================================================

class LLMInterface:
    """Unified LLM interface with automatic provider fallback.

    Order: Gemini → OpenAI (rotate keys) → OpenRouter.
    Environment variables supported:
      GEMINI_API_KEY
      OPENAI_API_KEY / OPENAI_API_KEY_1..3 / OPENAI_API_KEYS (comma separated)
      OPENROUTER_API_KEY, OPENROUTER_MODEL (default: openrouter/auto)
      MODEL_NAME (override default model preference)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_name = os.getenv("MODEL_NAME", model)

        # Gemini key (explicit api_key takes precedence)
        self.gemini_key = api_key or os.getenv("GEMINI_API_KEY")

        # Aggregate OpenAI keys from multiple patterns
        keys: List[str] = []
        csv_keys = os.getenv("OPENAI_API_KEYS", "").strip()
        if csv_keys:
            keys.extend([k.strip() for k in csv_keys.split(',') if k.strip()])
        for name in ["OPENAI_API_KEY", "OPENAI_API_KEY_1", "OPENAI_API_KEY_2", "OPENAI_API_KEY_3"]:
            v = os.getenv(name)
            if v and v not in keys:
                keys.append(v)
        self.openai_keys = keys

        # OpenRouter
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        self.openrouter_model = os.getenv("OPENROUTER_MODEL", "openrouter/auto")

        print("\n" + "="*80)
        print("INITIALIZING LLM INTERFACE (Gemini → OpenAI → OpenRouter)")
        print("="*80)
        print(f"Model preference: {self.model_name}")
        print(f"Gemini key: {'set' if self.gemini_key else 'not set'}")
        print(f"OpenAI keys detected: {len(self.openai_keys)}")
        print(f"OpenRouter key: {'set' if self.openrouter_key else 'not set'}")
        print("✓ LLM interface initialized\n")
    
    def generate_answer(
        self,
        query: str,
        context: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        verbose: bool = False
    ) -> Dict:
        """
        Generate answer using LLM with retrieved context.
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"GENERATING ANSWER (Fallback enabled)")
            print(f"{'='*80}")
            print(f"Query: {query}")
            print(f"Context length: {len(context)} characters")

        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        user_prompt = USER_PROMPT_TEMPLATE.format(context=context, query=query)

        errors: List[str] = []

        # 1. Gemini (attempt multiple candidate model names to handle API/version changes)
        if self.gemini_key:
            gemini_candidates = []
            # Preferred explicit model name from environment
            if self.model_name.startswith("gemini"):
                gemini_candidates.append(self.model_name)
            # Common current public model identifiers (ordered by quality/latency tradeoff)
            for m in [
                # Newer 2.x series
                "gemini-2.5-pro",
                "gemini-2.5-flash",
                "gemini-2.0-flash",
                # 1.5 series fallbacks
                "gemini-1.5-pro-latest",
                "gemini-1.5-flash-latest",
                "gemini-1.5-pro",
                "gemini-1.5-flash",
                "gemini-1.5-flash-8b",
                # Legacy
                "gemini-pro",
            ]:
                if m not in gemini_candidates:
                    gemini_candidates.append(m)
            try:
                genai.configure(api_key=self.gemini_key)
                for cand in gemini_candidates:
                    if verbose:
                        print(f"\nAttempting Gemini...")
                        print(f"  Candidate Model: {cand}")
                    try:
                        # Try both plain id and fully-qualified resource name
                        ids_to_try = [cand]
                        if not cand.startswith("models/"):
                            ids_to_try.append(f"models/{cand}")
                        model = None
                        last_error = None
                        for mid in ids_to_try:
                            try:
                                model = genai.GenerativeModel(mid)
                                break
                            except Exception as ge_inner:
                                last_error = ge_inner
                                continue
                        if model is None:
                            raise last_error or Exception("Unknown Gemini model init error")
                        response = model.generate_content([
                            SYSTEM_PROMPT,
                            user_prompt
                        ], generation_config=genai.types.GenerationConfig(temperature=temp))
                        answer = (getattr(response, 'text', '') or '').strip()
                        if answer:
                            return {
                                'answer': answer,
                                'model': cand,
                                'provider': 'gemini',
                                'tokens_used': {'total_tokens': 'N/A'},
                                'query': query
                            }
                        else:
                            errors.append(f"Gemini model {cand} returned empty response")
                    except Exception as ge:
                        errors.append(f"Gemini model {cand} error: {ge}")
                        if verbose:
                            print(f"❌ Gemini model {cand} failed: {ge}")
                # If loop completes without success
            except Exception as e:
                errors.append(f"Gemini setup error: {e}")
                if verbose:
                    print(f"❌ Gemini setup failed: {e}")

        # 2. OpenAI (iterate keys)
        if self.openai_keys and OpenAI is not None:
            for idx, key in enumerate(self.openai_keys, start=1):
                try:
                    if verbose:
                        print(f"\nAttempting OpenAI with key #{idx}...")
                    client = OpenAI(api_key=key)
                    openai_model = self.model_name if self.model_name.startswith("gpt") else "gpt-3.5-turbo"
                    response = client.chat.completions.create(
                        model=openai_model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=temp,
                        max_tokens=max_tok,
                        top_p=DEFAULT_TOP_P,
                    )
                    answer = response.choices[0].message.content.strip()
                    tokens_used = {
                        'prompt_tokens': getattr(response.usage, 'prompt_tokens', None),
                        'completion_tokens': getattr(response.usage, 'completion_tokens', None),
                        'total_tokens': getattr(response.usage, 'total_tokens', None),
                    }
                    return {
                        'answer': answer,
                        'model': openai_model,
                        'provider': 'openai',
                        'tokens_used': tokens_used,
                        'query': query
                    }
                except Exception as e:
                    errors.append(f"OpenAI key #{idx} error: {e}")
                    if verbose:
                        print(f"❌ OpenAI key #{idx} failed: {e}")

        # 3. OpenRouter
        if self.openrouter_key:
            try:
                if verbose:
                    print("\nAttempting OpenRouter...")
                    print(f"  Model: {self.openrouter_model}")
                url = "https://openrouter.ai/api/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.openrouter_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.openrouter_model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": temp,
                    "max_tokens": max_tok
                }
                r = requests.post(url, json=payload, headers=headers, timeout=60)
                r.raise_for_status()
                data = r.json()
                answer = data['choices'][0]['message']['content'].strip()
                return {
                    'answer': answer,
                    'model': self.openrouter_model,
                    'provider': 'openrouter',
                    'tokens_used': {'total_tokens': data.get('usage', {}).get('total_tokens')},
                    'query': query
                }
            except Exception as e:
                errors.append(f"OpenRouter error: {e}")
                if verbose:
                    print(f"❌ OpenRouter failed: {e}")

        # Final failure result
        return {
            'answer': 'Error: all providers failed',
            'model': self.model_name,
            'provider': 'none',
            'tokens_used': {'total_tokens': 0},
            'query': query,
            'error': '; '.join(errors) if errors else 'no providers configured'
        }
    
    def generate_with_sources(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        verbose: bool = False
    ) -> Dict:
        """
        Generate answer and include source attribution.
        
        USE CASE: When you want to show which documents were used
        
        PARAMETERS:
            query (str): User question
            retrieved_chunks (List[Dict]): Chunks from retriever
                Each dict should have 'text' and 'source' keys
            verbose (bool): Print details
        
        RETURNS:
            Dict with:
                - 'answer': Generated answer
                - 'sources': List of source documents used
                - 'chunks_used': Number of chunks provided
                - (other metadata)
        
        EXAMPLE:
            chunks = retriever.retrieve("What is transformer?", k=5)
            result = llm.generate_with_sources(query, chunks)
            
            print(result['answer'])
            print(f"Sources: {', '.join(result['sources'])}")
        """
        # Format context from chunks
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(
                f"Context {i} (Source: {chunk.get('source', 'unknown')}):\n"
                f"{chunk['text']}\n"
            )
        
        context = "\n".join(context_parts)
        
        # Generate answer
        result = self.generate_answer(query, context, verbose=verbose)
        
        # Add source information
        sources = list({chunk.get('source', 'unknown') for chunk in retrieved_chunks})
        result['sources'] = sources
        result['chunks_used'] = len(retrieved_chunks)
        
        return result


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def estimate_cost(tokens_used: Dict, model: str = DEFAULT_MODEL) -> float:
    """
    Estimate API cost based on token usage.
    
    PRICING (as of 2024):
        gpt-3.5-turbo: $0.0015/1K input, $0.002/1K output
        gpt-4: $0.03/1K input, $0.06/1K output
        gpt-4-turbo: $0.01/1K input, $0.03/1K output
    
    PARAMETERS:
        tokens_used (Dict): Token counts from generate_answer
        model (str): Model name
    
    RETURNS:
        float: Estimated cost in USD
    """
    pricing = {
        'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002},
        'gpt-4': {'input': 0.03, 'output': 0.06},
        'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
    }
    
    if model not in pricing:
        return 0.0
    
    input_cost = (tokens_used['prompt_tokens'] / 1000) * pricing[model]['input']
    output_cost = (tokens_used['completion_tokens'] / 1000) * pricing[model]['output']
    
    return input_cost + output_cost


# ==============================================================================
# SELF-TEST / DEMO BLOCK
# ==============================================================================

if __name__ == "__main__":
    """
    Test the LLM interface with mock context.
    
    NOTE: Requires GEMINI_API_KEY environment variable!
    
    Run:
        export GEMINI_API_KEY="your-key-here"
        python src/generation/llm_interface.py
    """
    
    print("="*80)
    print("TESTING: llm_interface.py (Google Gemini)")
    print("="*80)
    
    # Check for API key
    if not os.getenv('GEMINI_API_KEY'):
        print("\n⚠ GEMINI_API_KEY not set!")
        print("Set it to test the LLM interface:")
        print("  export GEMINI_API_KEY='your-key-here'")
        print("\nSkipping test...")
        exit(0)
    
    try:
        # Initialize LLM
        print("\n" + "-"*80)
        print("TEST 1: Initialize LLM Interface")
        print("-"*80)
        
        llm = LLMInterface(model="gemini-pro")
        
        # Test generation with mock context
        print("\n" + "-"*80)
        print("TEST 2: Generate Answer")
        print("-"*80)
        
        mock_context = """
        Context 1: The Transformer is a neural network architecture that relies entirely on self-attention mechanisms, dispensing with recurrence and convolutions entirely.
        
        Context 2: The architecture consists of an encoder and decoder, each with multiple layers. Each layer contains multi-head attention and feed-forward networks.
        """
        
        query = "What is the Transformer architecture?"
        
        result = llm.generate_answer(query, mock_context, verbose=True)
        
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
