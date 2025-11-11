# ==============================================================================
# FILE: prompt_templates_advanced.py
# PURPOSE: Advanced prompt templates with few-shot examples and chain-of-thought
# ==============================================================================

"""
ADVANCED PROMPT TEMPLATES

Features:
1. Few-shot examples per query type
2. Chain-of-thought reasoning
3. Source citation requests
4. Domain-specific instructions

Query Types:
- Technical Q&A (papers, concepts)
- Legal Q&A (regulations, compliance)
- Data Q&A (numbers, statistics)
- Comparative Analysis (vs, differences)
- Summarization
"""

import logging
from typing import Dict, List, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Query types"""
    TECHNICAL = "technical"
    LEGAL = "legal"
    DATA = "data"
    COMPARATIVE = "comparative"
    SUMMARIZATION = "summarization"
    GENERAL = "general"


class PromptTemplates:
    """Advanced prompt templates"""
    
    # ===========================================================================
    # SYSTEM PROMPTS
    # ===========================================================================
    
    SYSTEM_PROMPT_BASE = """You are a highly knowledgeable assistant that answers questions 
using ONLY the provided context documents. 

KEY INSTRUCTIONS:
1. ALWAYS base your answer on the provided context
2. If information is not in the context, say "Not found in documents"
3. Include document sources in your answer (e.g., "According to [Document Name]...")
4. Think through the question step by step before answering
5. Be specific and cite evidence
6. If uncertain, explicitly state your confidence level

CITATION FORMAT:
- First mention: "According to [Document Name] (relevance: X%), ..."
- Quote format: "The document states: '[exact quote]' - [Document Name]"
- Multiple sources: "Document A mentions X, while Document B mentions Y"

RESPONSE STRUCTURE:
1. Answer the question directly
2. Provide reasoning for your answer
3. Cite which documents support the answer
4. Note any limitations or uncertainties
"""
    
    # ===========================================================================
    # TECHNICAL Q&A TEMPLATE
    # ===========================================================================
    
    TECHNICAL_QA_PROMPT = """System Instructions:
{system_prompt}

You are answering a technical question. Technical questions often require deep understanding 
of complex concepts. Use these strategies:

1. DECOMPOSE: Break the question into sub-components
2. LOCATE: Find relevant passages in the context for each component
3. SYNTHESIZE: Connect the pieces into a coherent answer
4. VERIFY: Double-check your answer uses only context information

Question: {question}

Context from Documents:
{context}

EXAMPLE ANSWER STRUCTURE:
Q: How does attention work in transformers?
A: Let me break this down into components:
   1. Query-Key-Value mechanism: According to the Attention paper, each token has...
   2. Attention weight calculation: The similarity is computed as...
   3. Multi-head attention: The paper uses multiple representation subspaces...
   
   Therefore, attention works by... [complete explanation with sources]

Your Answer:
Let's think step by step.

Step 1: Identify the core question
- Main topic: [what is the main topic]

Step 2: Find relevant passages in context
- Key passage 1: [relevant excerpt]
- Key passage 2: [relevant excerpt]

Step 3: Synthesize your answer
[Your detailed answer with citations]

Confidence: [HIGH/MEDIUM/LOW]
Sources cited: [Document names]
"""
    
    # ===========================================================================
    # LEGAL Q&A TEMPLATE
    # ===========================================================================
    
    LEGAL_QA_PROMPT = """System Instructions:
{system_prompt}

You are answering a legal/regulatory question. Legal questions require precision and 
exact compliance with stated rules.

1. IDENTIFY: Which regulation/article applies?
2. LOCATE: Find the exact text in the context
3. INTERPRET: Explain the precise meaning
4. APPLY: Show how it applies to the question

Question: {question}

Context from Documents:
{context}

EXAMPLE ANSWER STRUCTURE:
Q: What are penalties for non-compliance under EU AI Act?
A: According to EU AI Act Article 99:
   - Administrative fines up to 6% of annual turnover for prohibited practices
   - Administrative fines up to 4% for high-risk violations
   
   This applies because [explanation] as stated in Article X.

Your Answer:
Let's approach this systematically.

Step 1: Identify Applicable Regulations
- Relevant regulation/article: [which regulation applies]
- Section: [specific section]

Step 2: Extract Exact Language
- Direct text: "[exact quote from document]"
- Source: [Document Name, Article/Section X]

Step 3: Interpretation
- What this means: [clear explanation]
- Scope: [who/what does this apply to]
- Exceptions: [any exceptions or carve-outs]

Step 4: Application to Your Question
- How it applies: [connect to the original question]
- Key points: [main takeaways]

Confidence: [HIGH/MEDIUM/LOW]
Sources: [Document names and article numbers]
"""
    
    # ===========================================================================
    # DATA Q&A TEMPLATE
    # ===========================================================================
    
    DATA_QA_PROMPT = """System Instructions:
{system_prompt}

You are answering a data/statistics question. Data questions require precision with numbers 
and clear sourcing of the data.

1. IDENTIFY: What data is needed?
2. LOCATE: Find the exact data point in context
3. EXTRACT: Get the specific number/statistic
4. CONTEXTUALIZE: Explain what the number means

Question: {question}

Context from Documents:
{context}

EXAMPLE ANSWER STRUCTURE:
Q: What was inflation in 2023?
A: According to the Inflation Calculator data:
   - Year 2023: 4.1% annual inflation rate
   - This represents a decrease from 8.0% in 2022
   
   The data shows [explanation of significance].

Your Answer:
Let's find and interpret the data.

Step 1: Identify Data Needed
- Looking for: [what specific data point]
- Metric: [how is it measured]
- Time period: [what timeframe]

Step 2: Locate Data in Context
- Found in: [which document/table]
- Exact value: [the number(s)]
- Units: [percent, dollars, etc.]

Step 3: Extract & Verify
- Data point: [X value Y unit]
- Source row/column: [table location]
- Reliability: [is this verified data]

Step 4: Contextualization
- What this means: [interpretation]
- Comparison: [compare to other periods if relevant]
- Significance: [why this matters]

Confidence: [HIGH/MEDIUM/LOW]
Data source: [Document name and location]
"""
    
    # ===========================================================================
    # COMPARATIVE ANALYSIS TEMPLATE
    # ===========================================================================
    
    COMPARATIVE_PROMPT = """System Instructions:
{system_prompt}

You are performing a comparative analysis. Compare items/concepts on multiple dimensions.

1. IDENTIFY: What aspects to compare?
2. LOCATE: Find information for each item
3. STRUCTURE: Create comparison matrix
4. ANALYZE: Identify similarities and differences

Question: {question}

Context from Documents:
{context}

EXAMPLE ANSWER STRUCTURE:
Q: How do transformers compare to RNNs?
A: Let me compare these architectures:
   
   Aspect 1 - Processing:
   - Transformers: Parallel processing of all tokens
   - RNNs: Sequential processing
   
   Aspect 2 - Memory:
   - Transformers: Attention mechanism for long-range dependencies
   - RNNs: Hidden state, limited context window
   
   Therefore, transformers offer [summary of comparison].

Your Answer:
Let's build a structured comparison.

Step 1: Identify Comparison Dimensions
- Dimension 1: [first aspect to compare]
- Dimension 2: [second aspect]
- Dimension 3: [third aspect]

Step 2: For Each Item, Find Supporting Context
Item A context:
- Characteristic 1: [from document]
- Characteristic 2: [from document]

Item B context:
- Characteristic 1: [from document]
- Characteristic 2: [from document]

Step 3: Structured Comparison
| Aspect | Item A | Item B |
|--------|--------|--------|
| [Dim 1] | [value] | [value] |
| [Dim 2] | [value] | [value] |

Step 4: Analysis
- Key similarities: [what they have in common]
- Key differences: [how they differ]
- When to use each: [guidance on choice]

Confidence: [HIGH/MEDIUM/LOW]
Sources: [Document names]
"""
    
    # ===========================================================================
    # SUMMARIZATION TEMPLATE
    # ===========================================================================
    
    SUMMARIZATION_PROMPT = """System Instructions:
{system_prompt}

You are creating a summary. Extract key points without losing important details.

Question: {question}

Context from Documents:
{context}

EXAMPLE ANSWER STRUCTURE:
Q: Summarize the attention mechanism
A: The attention mechanism is a key component of transformers that:
   1. Allows parallel processing of sequences
   2. Uses query-key-value computations to weight input relevance
   3. Employs multiple heads for diverse representation

Your Answer:
Let's create a structured summary.

Step 1: Identify Key Topics
- Main topic 1: [primary topic]
- Main topic 2: [secondary topic]
- Main topic 3: [tertiary topic]

Step 2: Extract Main Points for Each Topic
Topic 1 key points:
- Point 1: [key insight from context]
- Point 2: [key insight from context]

Topic 2 key points:
- Point 1: [key insight from context]
- Point 2: [key insight from context]

Step 3: Synthesize Summary
[Concise summary of 3-5 sentences covering key topics]

Key Takeaways:
- Takeaway 1: [main learning]
- Takeaway 2: [main learning]
- Takeaway 3: [main learning]

Confidence: [HIGH/MEDIUM/LOW]
Sources: [Document names]
"""
    
    # ===========================================================================
    # GENERAL TEMPLATE (Fallback)
    # ===========================================================================
    
    GENERAL_QA_PROMPT = """System Instructions:
{system_prompt}

Question: {question}

Context from Documents:
{context}

Let me think through this step by step:

Step 1: Understanding the Question
- Main topic: [what is being asked]
- Key terms: [important keywords]

Step 2: Relevant Context
- Applicable passages: [which parts of context apply]
- Key information: [relevant facts]

Step 3: Answer Construction
[Your detailed answer based on context]

Step 4: Verification
- Used context? [Yes/No - which parts?]
- Confidence: [HIGH/MEDIUM/LOW]
- Sources: [which documents provided information?]
"""
    
    # ===========================================================================
    # HELPER METHODS
    # ===========================================================================
    
    @staticmethod
    def get_template(query_type: str) -> str:
        """Get prompt template by query type"""
        
        templates = {
            QueryType.TECHNICAL.value: PromptTemplates.TECHNICAL_QA_PROMPT,
            QueryType.LEGAL.value: PromptTemplates.LEGAL_QA_PROMPT,
            QueryType.DATA.value: PromptTemplates.DATA_QA_PROMPT,
            QueryType.COMPARATIVE.value: PromptTemplates.COMPARATIVE_PROMPT,
            QueryType.SUMMARIZATION.value: PromptTemplates.SUMMARIZATION_PROMPT,
            QueryType.GENERAL.value: PromptTemplates.GENERAL_QA_PROMPT,
        }
        
        return templates.get(query_type, PromptTemplates.GENERAL_QA_PROMPT)
    
    @staticmethod
    def format_prompt(
        template: str,
        question: str,
        context: str,
        system_prompt: str = None
    ) -> Tuple[str, str]:
        """
        Format prompt with question and context.
        
        RETURNS:
            (system_prompt, user_prompt)
        """
        
        if system_prompt is None:
            system_prompt = PromptTemplates.SYSTEM_PROMPT_BASE
        
        # Format template
        user_prompt = template.format(
            system_prompt=system_prompt,
            question=question,
            context=context
        )
        
        return system_prompt, user_prompt
    
    @staticmethod
    def create_prompt_pair(
        question: str,
        context: str,
        query_type: str = QueryType.GENERAL.value
    ) -> Tuple[str, str]:
        """
        Create a complete prompt pair for LLM.
        
        PARAMETERS:
            question: User question
            context: Retrieved context
            query_type: Type of query (technical/legal/data/etc)
            
        RETURNS:
            (system_prompt, user_prompt)
        """
        
        logger.info(f"📝 Creating prompt for query type: {query_type}")
        
        template = PromptTemplates.get_template(query_type)
        system, user = PromptTemplates.format_prompt(template, question, context)
        
        logger.info(f"   ✅ Prompt created")
        logger.info(f"   System prompt length: {len(system)} chars")
        logger.info(f"   User prompt length: {len(user)} chars")
        
        return system, user


# ===========================================================================
# TESTING
# ===========================================================================

if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    test_question = "What is the transformer architecture?"
    test_context = """
    The Transformer is based solely on attention mechanisms. The attention mechanism 
    allows the model to attend to different representation subspaces. We employ multi-head 
    attention consisting of multiple representation subspaces at different locations.
    """
    
    print("\n🧪 TEST 1: Technical Query")
    sys_prompt, user_prompt = PromptTemplates.create_prompt_pair(
        test_question,
        test_context,
        QueryType.TECHNICAL.value
    )
    print(f"\n👤 SYSTEM PROMPT (first 200 chars):\n{sys_prompt[:200]}...\n")
    print(f"🤖 USER PROMPT (first 300 chars):\n{user_prompt[:300]}...\n")
    
    print("\n" + "="*80)
    print("🧪 TEST 2: Data Query")
    test_question_data = "What was the inflation rate in 2023?"
    sys_prompt, user_prompt = PromptTemplates.create_prompt_pair(
        test_question_data,
        test_context,
        QueryType.DATA.value
    )
    print(f"✅ Data query prompt created (length: {len(user_prompt)} chars)")
