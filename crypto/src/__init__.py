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
    predict_next_candle,
    add_open_to_predictions,
    calculate_prediction_metrics
)

from .loss_functions import (
    get_available_loss_functions,
    describe_loss_function,
    get_loss_function
)

from .binance_data_organizer import (
    BinanceDataOrganizer
)

from .plotting_utils import (
    draw_candlestick_chart,
    plot_combined_input_output_charts,
    plot_sample_data_comparison,
    plot_training_history,
    plot_predictions_vs_actual,
    plot_prediction_accuracy,
    create_interactive_candlestick_chart,
    plot_feature_importance,
    plot_residuals
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
    
    # Model utilities
    'create_lstm_model',
    'evaluate_model',
    'predict_next_candle',
    'add_open_to_predictions',
    'calculate_prediction_metrics',
    
    # Loss function utilities
    'get_available_loss_functions',
    'describe_loss_function',
    'get_loss_function',
    
    # Binance Data Organizer (includes normalization utilities)
    'BinanceDataOrganizer',
    
    # Plotting utilities
    'draw_candlestick_chart',
    'plot_combined_input_output_charts',
    'plot_sample_data_comparison',
    'plot_training_history',
    'plot_predictions_vs_actual',
    'plot_prediction_accuracy',
    'create_interactive_candlestick_chart',
    'plot_feature_importance',
    'plot_residuals',
    
    # Configuration
    'BaseConfig',
    'ProductionConfig',
    'TestConfig',
    'production_config',
    'test_config'
]
