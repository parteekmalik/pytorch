"""
Configuration classes for production and testing environments.
"""
from dataclasses import dataclass

@dataclass
class BaseConfig:
    """Base configuration class with common parameters."""
    
    # Data configuration
    symbol: str = "BTCUSDT"
    timeframe: str = "5m"
    start_date: str = "2021-01"
    end_date: str = "2021-03"
    
    # Model configuration
    sequence_length: int = 50
    prediction_length: int = 1
    train_split: float = 0.8
    
    # Training configuration
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    lstm_units: int = 100
    dropout_rate: float = 0.2
    
    # Early stopping configuration
    patience: int = 10
    min_delta: float = 1e-6
    
    # Learning rate reduction
    lr_patience: int = 5
    lr_factor: float = 0.5
    min_lr: float = 1e-7


@dataclass
class ProductionConfig(BaseConfig):
    """Production configuration with full parameters for real-world usage."""
    
    # Production-specific overrides
    epochs: int = 50
    batch_size: int = 1024 * 2
    lstm_units: int = 128

    sequence_length: int = 100
    prediction_length: int = 30
    
    # Extended data range for production
    start_date: str = "2021-01"
    end_date: str = "2021-06"
    
    # More aggressive early stopping for production
    patience: int = 15
    lr_patience: int = 8


@dataclass
class TestConfig(BaseConfig):
    """Test configuration optimized for fast execution during testing."""
    
    # Test-optimized parameters for speed
    epochs: int = 3
    batch_size: int = 1024
    lstm_units: int = 20
    sequence_length: int = 50
    prediction_length: int = 20
    
    # Shorter data range for testing
    start_date: str = "2021-01"
    end_date: str = "2021-01"
    
    # Faster early stopping for tests
    patience: int = 2
    lr_patience: int = 1
    
    # Data limits for testing
    max_rows: int = 1000
    train_split: float = 0.7


# Global configuration instances
production_config = ProductionConfig()
test_config = TestConfig()

