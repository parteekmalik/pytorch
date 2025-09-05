import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.binance_data_organizer import BinanceDataOrganizer, DataConfig, GroupedScaler
from src.utils import create_sequences


class TestDataConfig:
    """Test DataConfig dataclass"""
    
    def test_data_config_creation(self):
        """Test DataConfig creation with valid parameters"""
        config = DataConfig(
            symbol='BTCUSDT',
            timeframe='5m',
            start_time='2021-01',
            end_time='2021-01',
            sequence_length=5,
            prediction_length=1,
            train_split=0.8
        )
        
        assert config.symbol == 'BTCUSDT'
        assert config.timeframe == '5m'
        assert config.start_time == '2021-01'
        assert config.end_time == '2021-01'
        assert config.sequence_length == 5
        assert config.prediction_length == 1
        assert config.train_split == 0.8
    
    def test_data_config_defaults(self):
        """Test DataConfig with default train_split"""
        config = DataConfig(
            symbol='ETHUSDT',
            timeframe='1h',
            start_time='2021-02',
            end_time='2021-02',
            sequence_length=10,
            prediction_length=2
        )
        
        assert config.train_split == 0.8  # Default value


class TestGroupedScaler:
    """Test GroupedScaler class"""
    
    def test_grouped_scaler_initialization(self):
        """Test GroupedScaler initialization"""
        scaler = GroupedScaler()
        
        assert scaler.scalers == {}
        assert scaler.feature_groups == {}
        assert scaler.feature_names == []
        assert scaler.is_fitted == False
    
    def test_categorize_features(self):
        """Test feature categorization"""
        scaler = GroupedScaler()
        
        feature_names = ['Open', 'High', 'Low', 'Close', 'Volume']
        groups = scaler._categorize_features(feature_names)
        
        assert 'ohlc' in groups
        assert 'volume' in groups
        assert len(groups['ohlc']) == 4  # Open, High, Low, Close
        assert len(groups['volume']) == 1  # Volume
        assert groups['ohlc'] == [0, 1, 2, 3]
        assert groups['volume'] == [4]
    
    def test_fit_and_transform(self):
        """Test fit and transform methods"""
        scaler = GroupedScaler()
        
        # Create sample data
        feature_names = ['Open', 'High', 'Low', 'Close', 'Volume']
        X = np.random.rand(100, 5, 5)  # (samples, timesteps, features)
        
        # Fit scaler
        scaler.fit(X, feature_names)
        
        assert scaler.is_fitted == True
        assert 'ohlc' in scaler.scalers
        assert 'volume' in scaler.scalers
        
        # Transform data
        X_scaled = scaler.transform(X)
        
        assert X_scaled.shape == X.shape
        assert np.all(X_scaled >= 0)  # MinMaxScaler should produce values >= 0
        assert np.all(X_scaled <= 1.0001)  # MinMaxScaler should produce values <= 1 (with small tolerance)
    
    def test_inverse_transform(self):
        """Test inverse transform method"""
        scaler = GroupedScaler()
        
        feature_names = ['Open', 'High', 'Low', 'Close', 'Volume']
        X = np.random.rand(50, 5, 5)
        
        # Fit and transform
        scaler.fit(X, feature_names)
        X_scaled = scaler.transform(X)
        
        # Inverse transform
        X_inverse = scaler.inverse_transform(X_scaled)
        
        assert X_inverse.shape == X.shape
        np.testing.assert_array_almost_equal(X, X_inverse, decimal=5)
    
    def test_get_feature_groups(self):
        """Test get_feature_groups method"""
        scaler = GroupedScaler()
        
        feature_names = ['Open', 'High', 'Low', 'Close', 'Volume']
        X = np.random.rand(30, 5, 5)
        
        scaler.fit(X, feature_names)
        groups = scaler.get_feature_groups()
        
        assert 'ohlc' in groups
        assert 'volume' in groups
        assert groups['ohlc'] == ['Open', 'High', 'Low', 'Close']
        assert groups['volume'] == ['Volume']
    
    def test_fit_transform(self):
        """Test fit_transform method"""
        scaler = GroupedScaler()
        
        feature_names = ['Open', 'High', 'Low', 'Close', 'Volume']
        X = np.random.rand(40, 5, 5)
        
        X_scaled = scaler.fit_transform(X, feature_names)
        
        assert scaler.is_fitted == True
        assert X_scaled.shape == X.shape
        assert np.all(X_scaled >= 0)
        assert np.all(X_scaled <= 1.0001)  # Allow small floating point tolerance


class TestCreateSequences:
    """Test create_sequences function"""
    
    def test_create_sequences_basic(self):
        """Test basic sequence creation"""
        # Create sample OHLCV data
        data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'High': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'Low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            'Close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
            'Volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        })
        
        X, y, feature_cols = create_sequences(data, sequence_length=3, prediction_length=1)
        
        assert X.shape[0] == 7  # 10 - 3 - 1 + 1 = 7 sequences
        assert X.shape[1] == 3  # sequence_length
        assert X.shape[2] == 5  # 5 features (OHLCV)
        assert y.shape[0] == 7  # same number of sequences
        assert y.shape[1] == 5  # 5 target values (OHLCV)
        assert feature_cols == ['Open', 'High', 'Low', 'Close', 'Volume']
    
    def test_create_sequences_validation(self):
        """Test sequence creation validation"""
        # Test with empty data
        with pytest.raises(ValueError, match="Input data is empty or None"):
            create_sequences(pd.DataFrame(), 5, 1)
        
        # Test with None data
        with pytest.raises(ValueError, match="Input data is empty or None"):
            create_sequences(None, 5, 1)
        
        # Test with missing columns
        bad_data = pd.DataFrame({'Open': [1, 2, 3], 'High': [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            create_sequences(bad_data, 5, 1)
        
        # Test with insufficient data
        small_data = pd.DataFrame({
            'Open': [1, 2, 3], 'High': [1, 2, 3], 'Low': [1, 2, 3],
            'Close': [1, 2, 3], 'Volume': [1, 2, 3]
        })
        with pytest.raises(ValueError, match="Not enough data after cleaning"):
            create_sequences(small_data, 5, 1)
    
    def test_create_sequences_with_nan(self):
        """Test sequence creation with NaN values"""
        data = pd.DataFrame({
            'Open': [100, 101, np.nan, 103, 104, 105, 106, 107, 108, 109],
            'High': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'Low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            'Close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
            'Volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        })
        
        X, y, feature_cols = create_sequences(data, sequence_length=3, prediction_length=1)
        
        # Should have fewer sequences due to NaN removal
        assert X.shape[0] == 6  # 9 valid rows - 3 - 1 + 1 = 6 sequences
        assert X.shape[1] == 3
        assert X.shape[2] == 5


class TestBinanceDataOrganizer:
    """Test BinanceDataOrganizer class"""
    
    @patch('src.binance_data_organizer.download_binance_data')
    def test_organizer_initialization(self, mock_download):
        """Test BinanceDataOrganizer initialization"""
        # Mock download function
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [101, 102, 103],
            'Low': [99, 100, 101],
            'Close': [100.5, 101.5, 102.5],
            'Volume': [1000, 1100, 1200]
        })
        mock_download.return_value = mock_data
        
        config = DataConfig(
            symbol='BTCUSDT',
            timeframe='5m',
            start_time='2021-01',
            end_time='2021-01',
            sequence_length=5,
            prediction_length=1
        )
        
        organizer = BinanceDataOrganizer(config)
        
        assert organizer.config == config
        assert organizer.raw_data is not None
        assert organizer.scaler_X is None
        assert organizer.scaler_y is None
        assert organizer.is_scalers_fitted == False
        mock_download.assert_called_once()
    
    @patch('src.binance_data_organizer.download_binance_data')
    def test_organizer_validation(self, mock_download):
        """Test data validation on initialization"""
        # Mock download with missing columns
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [101, 102, 103]
        })
        mock_download.return_value = mock_data
        
        config = DataConfig(
            symbol='BTCUSDT',
            timeframe='5m',
            start_time='2021-01',
            end_time='2021-01',
            sequence_length=5,
            prediction_length=1
        )
        
        with pytest.raises(ValueError, match="Data must contain columns"):
            BinanceDataOrganizer(config)
    
    @patch('src.binance_data_organizer.download_binance_data')
    def test_process_all(self, mock_download):
        """Test process_all method"""
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'High': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'Low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            'Close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
            'Volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        })
        mock_download.return_value = mock_data
        
        config = DataConfig(
            symbol='BTCUSDT',
            timeframe='5m',
            start_time='2021-01',
            end_time='2021-01',
            sequence_length=3,
            prediction_length=1
        )
        
        organizer = BinanceDataOrganizer(config)
        result = organizer.process_all()
        
        assert result == True
    
    @patch('src.binance_data_organizer.download_binance_data')
    def test_get_unscaled_data(self, mock_download):
        """Test get_unscaled_data method"""
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'High': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'Low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            'Close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
            'Volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        })
        mock_download.return_value = mock_data
        
        config = DataConfig(
            symbol='BTCUSDT',
            timeframe='5m',
            start_time='2021-01',
            end_time='2021-01',
            sequence_length=3,
            prediction_length=1,
            train_split=0.8
        )
        
        organizer = BinanceDataOrganizer(config)
        
        # Test 'all' data type
        data = organizer.get_unscaled_data('all')
        
        assert 'X_train' in data
        assert 'X_test' in data
        assert 'y_train' in data
        assert 'y_test' in data
        assert data['X_train'].shape[0] + data['X_test'].shape[0] == 7  # 10 - 3 - 1 + 1 = 7 total sequences
        
        # Test 'train' data type
        train_data = organizer.get_unscaled_data('train')
        assert 'X_train' in train_data
        assert 'y_train' in train_data
        assert 'X_test' not in train_data
        
        # Test 'test' data type
        test_data = organizer.get_unscaled_data('test')
        assert 'X_test' in test_data
        assert 'y_test' in test_data
        assert 'X_train' not in test_data
    
    @patch('src.binance_data_organizer.download_binance_data')
    def test_get_scaled_data(self, mock_download):
        """Test get_scaled_data method"""
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'High': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'Low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            'Close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
            'Volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        })
        mock_download.return_value = mock_data
        
        config = DataConfig(
            symbol='BTCUSDT',
            timeframe='5m',
            start_time='2021-01',
            end_time='2021-01',
            sequence_length=3,
            prediction_length=1,
            train_split=0.8
        )
        
        organizer = BinanceDataOrganizer(config)
        
        # Test scaled data generation
        scaled_data = organizer.get_scaled_data('all')
        
        assert 'X_train_scaled' in scaled_data
        assert 'X_test_scaled' in scaled_data
        assert 'y_train_scaled' in scaled_data
        assert 'y_test_scaled' in scaled_data
        
        # Check that scalers are fitted
        assert organizer.is_scalers_fitted == True
        assert organizer.scaler_X is not None
        assert organizer.scaler_y is not None
        
        # Check scaling ranges
        assert np.all(scaled_data['X_train_scaled'] >= 0)
        assert np.all(scaled_data['X_train_scaled'] <= 1.0001)  # Allow small floating point tolerance
        assert np.all(scaled_data['y_train_scaled'] >= 0)
        assert np.all(scaled_data['y_train_scaled'] <= 1)
    
    @patch('src.binance_data_organizer.download_binance_data')
    def test_get_scalers(self, mock_download):
        """Test get_scalers method"""
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'High': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'Low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            'Close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
            'Volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        })
        mock_download.return_value = mock_data
        
        config = DataConfig(
            symbol='BTCUSDT',
            timeframe='5m',
            start_time='2021-01',
            end_time='2021-01',
            sequence_length=3,
            prediction_length=1
        )
        
        organizer = BinanceDataOrganizer(config)
        
        # Test before scalers are fitted
        with pytest.raises(ValueError, match="Scalers not fitted"):
            organizer.get_scalers()
        
        # Fit scalers
        organizer.get_scaled_data('all')
        
        # Test after scalers are fitted
        scalers = organizer.get_scalers()
        assert 'X' in scalers
        assert 'y' in scalers
        assert scalers['X'] is not None
        assert scalers['y'] is not None
    
    @patch('src.binance_data_organizer.download_binance_data')
    def test_inverse_transform_targets(self, mock_download):
        """Test inverse_transform_targets method"""
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'High': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'Low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            'Close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
            'Volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        })
        mock_download.return_value = mock_data
        
        config = DataConfig(
            symbol='BTCUSDT',
            timeframe='5m',
            start_time='2021-01',
            end_time='2021-01',
            sequence_length=3,
            prediction_length=1
        )
        
        organizer = BinanceDataOrganizer(config)
        
        # Test before scalers are fitted
        with pytest.raises(ValueError, match="Scalers not fitted"):
            organizer.inverse_transform_targets(np.array([[0.5, 0.5, 0.5, 0.5, 0.5]]))
        
        # Fit scalers
        scaled_data = organizer.get_scaled_data('all')
        
        # Test inverse transform
        y_scaled = scaled_data['y_train_scaled'][:5]  # First 5 samples
        y_original = organizer.inverse_transform_targets(y_scaled)
        
        assert y_original.shape == y_scaled.shape
        assert y_original.dtype == np.float64
    
    @patch('src.binance_data_organizer.download_binance_data')
    def test_get_data_summary(self, mock_download):
        """Test get_data_summary method"""
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [101, 102, 103, 104, 105],
            'Low': [99, 100, 101, 102, 103],
            'Close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'Volume': [1000, 1100, 1200, 1300, 1400]
        })
        mock_download.return_value = mock_data
        
        config = DataConfig(
            symbol='BTCUSDT',
            timeframe='5m',
            start_time='2021-01',
            end_time='2021-01',
            sequence_length=3,
            prediction_length=1
        )
        
        organizer = BinanceDataOrganizer(config)
        summary = organizer.get_data_summary()
        
        assert summary['symbol'] == 'BTCUSDT'
        assert summary['timeframe'] == '5m'
        assert summary['total_rows'] == 5
        assert summary['columns'] == ['Open', 'High', 'Low', 'Close', 'Volume']
        assert 'price_range' in summary
        assert 'volume_range' in summary
        assert summary['price_range']['min'] == 99
        assert summary['price_range']['max'] == 105
        assert summary['volume_range']['min'] == 1000
        assert summary['volume_range']['max'] == 1400
    
    @patch('src.binance_data_organizer.download_binance_data')
    def test_get_sequence_info(self, mock_download):
        """Test get_sequence_info method"""
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'High': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'Low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            'Close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
            'Volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        })
        mock_download.return_value = mock_data
        
        config = DataConfig(
            symbol='BTCUSDT',
            timeframe='5m',
            start_time='2021-01',
            end_time='2021-01',
            sequence_length=3,
            prediction_length=1,
            train_split=0.8
        )
        
        organizer = BinanceDataOrganizer(config)
        seq_info = organizer.get_sequence_info()
        
        assert seq_info['total_sequences'] == 7  # 10 - 3 - 1 + 1 = 7
        assert seq_info['train_sequences'] == 5  # 7 * 0.8 = 5.6 -> 5
        assert seq_info['test_sequences'] == 2  # 7 - 5 = 2
        assert seq_info['sequence_length'] == 3
        assert seq_info['prediction_length'] == 1
        assert seq_info['features_per_timestep'] == 5
    
    @patch('src.binance_data_organizer.download_binance_data')
    def test_get_feature_info(self, mock_download):
        """Test get_feature_info method"""
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'High': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'Low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            'Close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
            'Volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        })
        mock_download.return_value = mock_data
        
        config = DataConfig(
            symbol='BTCUSDT',
            timeframe='5m',
            start_time='2021-01',
            end_time='2021-01',
            sequence_length=3,
            prediction_length=1
        )
        
        organizer = BinanceDataOrganizer(config)
        feature_info = organizer.get_feature_info()
        
        assert feature_info['feature_columns'] == ['Open', 'High', 'Low', 'Close', 'Volume']
        assert feature_info['num_features'] == 5
        assert feature_info['sequence_length'] == 3
        assert feature_info['prediction_length'] == 1
        assert feature_info['data_shape'] == (10, 5)
        assert feature_info['total_sequences'] == 7


if __name__ == '__main__':
    pytest.main([__file__])
