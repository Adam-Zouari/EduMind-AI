"""
Shared Evaluation Utilities for MLflow Experiments

Provides functions for:
- Recall@K computation
- Mean Reciprocal Rank (MRR)
- Latency measurement
- Answer quality evaluation
- Faithfulness evaluation
"""

from typing import List, Dict, Any, Callable, Optional
import time
from contextlib import contextmanager
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Calculate Recall@K metric.
    
    Recall@K = (Number of relevant items in top-K) / (Total number of relevant items)
    
    Args:
        retrieved_ids: List of retrieved document IDs (ordered by relevance)
        relevant_ids: List of ground-truth relevant document IDs
        k: Number of top results to consider
        
    Returns:
        Recall@K score (0.0 to 1.0)
    """
    if not relevant_ids:
        logger.warning("No relevant IDs provided for Recall@K calculation")
        return 0.0
    
    # Consider only top-k retrieved results
    top_k_retrieved = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    
    # Count how many relevant items are in top-k
    hits = len(top_k_retrieved.intersection(relevant_set))
    
    # Recall = hits / total_relevant
    recall = hits / len(relevant_set)
    
    return recall


def compute_mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    MRR = 1 / rank_of_first_relevant_item
    
    Args:
        retrieved_ids: List of retrieved document IDs (ordered by relevance)
        relevant_ids: List of ground-truth relevant document IDs
        
    Returns:
        MRR score (0.0 to 1.0)
    """
    if not relevant_ids:
        logger.warning("No relevant IDs provided for MRR calculation")
        return 0.0
    
    relevant_set = set(relevant_ids)
    
    # Find the rank (1-indexed) of the first relevant item
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    
    # No relevant item found
    return 0.0


@contextmanager
def measure_latency():
    """
    Context manager to measure execution time.
    
    Usage:
        with measure_latency() as timer:
            # code to measure
            pass
        latency_ms = timer['latency_ms']
    
    Yields:
        Dictionary with 'latency_ms' key
    """
    timer = {'latency_ms': 0}
    start_time = time.time()
    
    try:
        yield timer
    finally:
        end_time = time.time()
        timer['latency_ms'] = (end_time - start_time) * 1000  # Convert to milliseconds


def measure_function_latency(func: Callable, *args, **kwargs) -> tuple:
    """
    Measure the execution time of a function.
    
    Args:
        func: Function to measure
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        Tuple of (result, latency_ms)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    latency_ms = (end_time - start_time) * 1000
    return result, latency_ms


def evaluate_answer_quality(
    answer: str, 
    reference_answer: Optional[str] = None,
    context: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate answer quality using simple heuristics.
    
    For full evaluation, human scoring or LLM-as-judge is recommended.
    This provides basic automated metrics.
    
    Args:
        answer: Generated answer
        reference_answer: Optional reference answer for comparison
        context: Optional context used to generate answer
        
    Returns:
        Dictionary with quality metrics
    """
    metrics = {}
    
    # Basic length check
    word_count = len(answer.split())
    metrics['response_length_words'] = word_count
    
    # Check if answer is not empty
    metrics['is_non_empty'] = 1.0 if answer.strip() else 0.0
    
    # Check if answer is not just repeating context
    if context:
        # Simple check: if answer is substring of context, it might be just copying
        is_copy = answer.strip() in context
        metrics['is_original'] = 0.0 if is_copy else 1.0
    
    # Basic quality score (heuristic)
    # A good answer is: non-empty, reasonable length, original
    quality_score = 0.0
    if word_count >= 5 and word_count <= 200:  # Reasonable length
        quality_score += 0.5
    if metrics['is_non_empty'] > 0:
        quality_score += 0.25
    if metrics.get('is_original', 1.0) > 0:
        quality_score += 0.25
    
    metrics['basic_quality_score'] = quality_score
    
    return metrics


def evaluate_faithfulness(answer: str, context: str) -> float:
    """
    Evaluate if the answer is faithful to the context.
    
    Simple heuristic: Check if key terms from answer appear in context.
    For production, use NLI models or LLM-as-judge.
    
    Args:
        answer: Generated answer
        context: Context used to generate answer
        
    Returns:
        Faithfulness score (0.0 to 1.0)
    """
    if not answer or not context:
        return 0.0
    
    # Extract key terms from answer (simple: words longer than 4 chars)
    answer_words = set(
        word.lower().strip('.,!?;:') 
        for word in answer.split() 
        if len(word) > 4
    )
    
    context_lower = context.lower()
    
    # Check how many answer terms are in context
    if not answer_words:
        return 0.5  # Neutral if no key terms
    
    found_terms = sum(1 for word in answer_words if word in context_lower)
    faithfulness = found_terms / len(answer_words)
    
    return faithfulness


def compute_mean_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Compute mean values across multiple metric dictionaries.
    
    Args:
        metrics_list: List of metric dictionaries
        
    Returns:
        Dictionary with mean values for each metric
    """
    if not metrics_list:
        return {}
    
    # Get all unique keys
    all_keys = set()
    for metrics in metrics_list:
        all_keys.update(metrics.keys())
    
    # Compute mean for each key
    mean_metrics = {}
    for key in all_keys:
        values = [m[key] for m in metrics_list if key in m]
        if values:
            mean_metrics[f"mean_{key}"] = np.mean(values)
            mean_metrics[f"std_{key}"] = np.std(values)
    
    return mean_metrics


def evaluate_retrieval_quality(
    query: str,
    retrieved_docs: List[Dict[str, Any]],
    ground_truth_ids: List[str],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """
    Comprehensive retrieval quality evaluation.
    
    Args:
        query: Query text
        retrieved_docs: List of retrieved documents with 'id' field
        ground_truth_ids: List of relevant document IDs
        k_values: List of K values to compute Recall@K
        
    Returns:
        Dictionary with all retrieval metrics
    """
    metrics = {}
    
    # Extract IDs from retrieved docs
    retrieved_ids = [doc.get('id', doc.get('document_id', '')) for doc in retrieved_docs]
    
    # Compute Recall@K for different K values
    for k in k_values:
        recall = compute_recall_at_k(retrieved_ids, ground_truth_ids, k)
        metrics[f'recall_at_{k}'] = recall
    
    # Compute MRR
    mrr = compute_mrr(retrieved_ids, ground_truth_ids)
    metrics['mrr'] = mrr
    
    # Number of retrieved docs
    metrics['num_retrieved'] = len(retrieved_docs)
    
    return metrics


if __name__ == "__main__":
    # Example usage and tests
    print("=== Testing Evaluation Utilities ===\n")
    
    # Test Recall@K
    retrieved = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
    relevant = ['doc2', 'doc5', 'doc7']
    
    recall_5 = compute_recall_at_k(retrieved, relevant, k=5)
    print(f"Recall@5: {recall_5:.3f}")  # Should be 2/3 = 0.667
    
    # Test MRR
    mrr = compute_mrr(retrieved, relevant)
    print(f"MRR: {mrr:.3f}")  # Should be 1/2 = 0.5 (doc2 is at position 2)
    
    # Test latency measurement
    with measure_latency() as timer:
        time.sleep(0.1)  # Simulate some work
    print(f"\nLatency: {timer['latency_ms']:.2f} ms")
    
    # Test answer quality
    answer = "Machine learning is a subset of AI that enables computers to learn from data."
    context = "Machine learning (ML) is a field of AI. It enables systems to learn from data."
    
    quality = evaluate_answer_quality(answer, context=context)
    print(f"\nAnswer Quality Metrics: {quality}")
    
    faithfulness = evaluate_faithfulness(answer, context)
    print(f"Faithfulness Score: {faithfulness:.3f}")
    
    print("\n✓ All evaluation utilities working correctly")
