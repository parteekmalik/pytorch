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
    scale_data
)

from .plotting_utils import (
    plot_training_history,
    plot_predictions_vs_actual,
    plot_price_data,
    plot_prediction_errors,
    plot_feature_importance,
    plot_feature_analysis,
    plot_candlestick_chart,
    plot_prediction_comparison,
    plot_prediction_accuracy_distribution,
    plot_model_performance_summary
)

from .config import (
    BaseConfig,
    ProductionConfig,
    TestConfig,
    production_config,
    test_config
)

__all__ = [
    # Memory utilities
    'get_memory_usage',
    'check_memory_limit',
    'force_garbage_collection',
    'get_memory_stats',
    'print_memory_stats',
    
    # Data utilities (now part of binance_data_organizer)
    'scale_data',
    
    # Model utilities
    'create_lstm_model',
    'evaluate_model',
    'predict_next_candle',
    
    # Binance Data Organizer (includes normalization utilities)
    'BinanceDataOrganizer',
    'DataConfig',
    'GroupedScaler',
    
    # Plotting utilities
    'plot_training_history',
    'plot_predictions_vs_actual',
    'plot_price_data',
    'plot_prediction_errors',
    'plot_feature_importance',
    'plot_feature_analysis',
    'plot_candlestick_chart',
    'plot_prediction_comparison',
    'plot_prediction_accuracy_distribution',
    'plot_model_performance_summary',
    
    # Configuration
    'BaseConfig',
    'ProductionConfig',
    'TestConfig',
    'production_config',
    'test_config'
]
