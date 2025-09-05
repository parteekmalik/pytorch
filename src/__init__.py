"""
Utils package for crypto prediction notebooks.
"""

from .memory_utils import (
    get_memory_usage,
    check_memory_limit,
    force_garbage_collection,
    get_memory_stats,
    print_memory_stats
)

# Data utilities are now part of binance_data_organizer

from .model_utils import (
    create_lstm_model,
    evaluate_model,
    predict_next_candle
)

from .binance_data_organizer import (
    BinanceDataOrganizer,
    DataConfig,
    GroupedScaler,
    scale_data,
    predict_with_grouped_scaler,
    # Data utility functions
    create_minimal_features,
    create_sliding_windows,
    download_crypto_data,
    # Legacy compatibility
    scale_time_series_data_grouped
)

from .plotting_utils import (
    plot_training_history,
    plot_predictions_vs_actual,
    plot_price_data,
    plot_prediction_errors,
    plot_feature_importance,
    plot_feature_analysis
)

__all__ = [
    # Memory utilities
    'get_memory_usage',
    'check_memory_limit',
    'force_garbage_collection',
    'get_memory_stats',
    'print_memory_stats',
    
    # Data utilities (now part of binance_data_organizer)
    'create_minimal_features',
    'create_sliding_windows',
    'download_crypto_data',
    'scale_data',
    
    # Model utilities
    'create_lstm_model',
    'evaluate_model',
    'predict_next_candle',
    
    # Binance Data Organizer (includes normalization utilities)
    'BinanceDataOrganizer',
    'DataConfig',
    'GroupedScaler',
    'scale_time_series_data_grouped',
    'predict_with_grouped_scaler',
    
    # Plotting utilities
    'plot_training_history',
    'plot_predictions_vs_actual',
    'plot_price_data',
    'plot_prediction_errors',
    'plot_feature_importance',
    'plot_feature_analysis'
]
