"""Shared utilities for MLflow experiments."""

from .evaluation import (
    compute_recall_at_k,
    compute_mrr,
    measure_latency,
    measure_function_latency,
    evaluate_answer_quality,
    evaluate_faithfulness,
    compute_mean_metrics,
    evaluate_retrieval_quality
)

from .gpu_utils import (
    is_cuda_available,
    get_gpu_memory_usage,
    get_gpu_utilization,
    get_gpu_info,
    monitor_gpu_during_execution,
    measure_throughput,
    reset_peak_memory_stats,
    get_peak_memory_stats
)

from .metrics_logger import (
    set_experiment,
    start_run,
    end_run,
    log_params,
    log_metrics,
    log_dict_as_json,
    log_text_as_artifact,
    log_numpy_array,
    log_figure,
    log_experiment_results,
    create_comparison_plot,
    MLflowExperiment
)

__all__ = [
    # Evaluation
    'compute_recall_at_k',
    'compute_mrr',
    'measure_latency',
    'measure_function_latency',
    'evaluate_answer_quality',
    'evaluate_faithfulness',
    'compute_mean_metrics',
    'evaluate_retrieval_quality',
    # GPU Utils
    'is_cuda_available',
    'get_gpu_memory_usage',
    'get_gpu_utilization',
    'get_gpu_info',
    'monitor_gpu_during_execution',
    'measure_throughput',
    'reset_peak_memory_stats',
    'get_peak_memory_stats',
    # Metrics Logger
    'set_experiment',
    'start_run',
    'end_run',
    'log_params',
    'log_metrics',
    'log_dict_as_json',
    'log_text_as_artifact',
    'log_numpy_array',
    'log_figure',
    'log_experiment_results',
    'create_comparison_plot',
    'MLflowExperiment'
]
