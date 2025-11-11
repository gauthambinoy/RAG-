# ==============================================================================
# FILE: chunk_statistics.py
# PURPOSE: Analyze and visualize chunking strategy results
# ==============================================================================

"""
CHUNK STATISTICS ANALYZER

Provides comprehensive analysis of chunking results:
- Distribution by document type
- Size statistics per chunk type
- Quality metrics
- Visual summaries

Used for:
1. Validating chunking quality
2. Identifying optimization opportunities
3. Documenting improvements
4. Creating reports
"""

import logging
import numpy as np
from typing import List, Dict
from collections import defaultdict

logger = logging.getLogger(__name__)


class ChunkStatistics:
    """Analyze chunk statistics"""
    
    def __init__(self, chunks: List[Dict]):
        """
        Initialize with chunks.
        
        PARAMETERS:
            chunks (List[Dict]): List of chunks from SmartChunker
        """
        self.chunks = chunks
        self.analyze()
    
    def analyze(self):
        """Perform comprehensive analysis"""
        
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 CHUNK STATISTICS ANALYSIS")
        logger.info(f"{'='*70}\n")
        
        # Basic stats
        total_chars = sum(len(c['text']) for c in self.chunks)
        
        logger.info(f"Total Chunks: {len(self.chunks)}")
        logger.info(f"Total Characters: {total_chars:,}")
        logger.info(f"Total Words: {total_chars // 5:,}")
        
        # By document type
        logger.info(f"\n📋 BY DOCUMENT TYPE:")
        doc_types = defaultdict(list)
        for chunk in self.chunks:
            doc_type = chunk.get('document_type', 'unknown')
            doc_types[doc_type].append(chunk)
        
        for doc_type, chunks_of_type in doc_types.items():
            count = len(chunks_of_type)
            chars = sum(len(c['text']) for c in chunks_of_type)
            avg_size = np.mean([len(c['text']) for c in chunks_of_type])
            
            logger.info(f"\n   {doc_type.upper()}:")
            logger.info(f"   - Count: {count} chunks")
            logger.info(f"   - Total: {chars:,} chars")
            logger.info(f"   - Average size: {avg_size:.0f} chars")
            logger.info(f"   - Min size: {min([len(c['text']) for c in chunks_of_type])} chars")
            logger.info(f"   - Max size: {max([len(c['text']) for c in chunks_of_type])} chars")
        
        # By chunk type (method used)
        logger.info(f"\n🔪 BY CHUNKING METHOD:")
        chunk_types = defaultdict(list)
        for chunk in self.chunks:
            chunk_type = chunk.get('chunk_type', 'unknown')
            chunk_types[chunk_type].append(chunk)
        
        for chunk_type, chunks_of_method in chunk_types.items():
            count = len(chunks_of_method)
            avg_size = np.mean([len(c['text']) for c in chunks_of_method])
            
            logger.info(f"\n   {chunk_type.upper()}:")
            logger.info(f"   - Count: {count} chunks")
            logger.info(f"   - Average size: {avg_size:.0f} chars")
        
        # Size distribution
        sizes = [len(c['text']) for c in self.chunks]
        
        logger.info(f"\n📏 SIZE DISTRIBUTION:")
        logger.info(f"   Mean: {np.mean(sizes):.0f} chars")
        logger.info(f"   Median: {np.median(sizes):.0f} chars")
        logger.info(f"   Std Dev: {np.std(sizes):.0f} chars")
        logger.info(f"   Min: {np.min(sizes)} chars")
        logger.info(f"   Max: {np.max(sizes)} chars")
        
        # Size buckets
        logger.info(f"\n📊 SIZE DISTRIBUTION BUCKETS:")
        buckets = {
            'Tiny (<200)': 0,
            'Small (200-500)': 0,
            'Medium (500-1000)': 0,
            'Large (1000-1500)': 0,
            'XLarge (>1500)': 0
        }
        
        for size in sizes:
            if size < 200:
                buckets['Tiny (<200)'] += 1
            elif size < 500:
                buckets['Small (200-500)'] += 1
            elif size < 1000:
                buckets['Medium (500-1000)'] += 1
            elif size < 1500:
                buckets['Large (1000-1500)'] += 1
            else:
                buckets['XLarge (>1500)'] += 1
        
        for bucket_name, count in buckets.items():
            pct = 100 * count / len(self.chunks)
            bar = "█" * int(pct / 2)
            logger.info(f"   {bucket_name:20} {count:3} ({pct:5.1f}%) {bar}")
        
        logger.info(f"\n{'='*70}\n")
    
    def get_summary(self) -> Dict:
        """Get summary as dictionary"""
        
        sizes = [len(c['text']) for c in self.chunks]
        
        return {
            'total_chunks': len(self.chunks),
            'total_characters': sum(sizes),
            'average_chunk_size': np.mean(sizes),
            'median_chunk_size': np.median(sizes),
            'min_chunk_size': np.min(sizes),
            'max_chunk_size': np.max(sizes),
            'std_dev': np.std(sizes)
        }


def print_chunk_comparison(before_chunks: List[Dict], after_chunks: List[Dict]):
    """
    Compare before and after chunking strategies.
    
    PARAMETERS:
        before_chunks: Chunks from old strategy
        after_chunks: Chunks from smart chunking
    """
    
    logger.info(f"\n{'='*70}")
    logger.info(f"📊 CHUNKING STRATEGY COMPARISON")
    logger.info(f"{'='*70}\n")
    
    before_stats = ChunkStatistics(before_chunks).get_summary()
    after_stats = ChunkStatistics(after_chunks).get_summary()
    
    logger.info(f"{'Metric':<30} {'Before':<15} {'After':<15} {'Change':<15}")
    logger.info(f"{'-'*70}")
    
    # Total chunks
    change = after_stats['total_chunks'] - before_stats['total_chunks']
    pct = 100 * change / before_stats['total_chunks']
    logger.info(f"{'Total Chunks':<30} {before_stats['total_chunks']:<15} {after_stats['total_chunks']:<15} {change:+.0f} ({pct:+.1f}%)")
    
    # Average size
    change = after_stats['average_chunk_size'] - before_stats['average_chunk_size']
    pct = 100 * change / before_stats['average_chunk_size']
    logger.info(f"{'Average Size (chars)':<30} {before_stats['average_chunk_size']:<15.0f} {after_stats['average_chunk_size']:<15.0f} {change:+.0f} ({pct:+.1f}%)")
    
    # Median size
    change = after_stats['median_chunk_size'] - before_stats['median_chunk_size']
    logger.info(f"{'Median Size (chars)':<30} {before_stats['median_chunk_size']:<15.0f} {after_stats['median_chunk_size']:<15.0f} {change:+.0f}")
    
    # Std dev (lower is better)
    change = after_stats['std_dev'] - before_stats['std_dev']
    logger.info(f"{'Std Deviation (lower=better)':<30} {before_stats['std_dev']:<15.0f} {after_stats['std_dev']:<15.0f} {change:+.0f}")
    
    logger.info(f"\n{'='*70}\n")


# ==============================================================================
# TESTING
# ==============================================================================

if __name__ == "__main__":
    """Test chunk statistics"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    # Sample chunks
    sample_chunks = [
        {"text": "A" * 500, "chunk_type": "semantic", "document_type": "technical_paper"},
        {"text": "B" * 800, "chunk_type": "semantic", "document_type": "technical_paper"},
        {"text": "C" * 600, "chunk_type": "hierarchical", "document_type": "policy_legal"},
        {"text": "D" * 200, "chunk_type": "row_aware", "document_type": "tabular_data"},
    ]
    
    analyzer = ChunkStatistics(sample_chunks)
    summary = analyzer.get_summary()
    
    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
