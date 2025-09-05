import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os
import io

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import create_sequences, download_binance_data, _download_single_month


class TestCreateSequences:
    """Test create_sequences function"""
    
    def test_create_sequences_basic(self):
        """Test basic sequence creation with valid data"""
        data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'High': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'Low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            'Close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
            'Volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        })
        
        X, y, feature_cols = create_sequences(data, sequence_length=3, prediction_length=1)
        
        assert X.shape == (7, 3, 5)  # 10 - 3 - 1 + 1 = 7 sequences
        assert y.shape == (7, 5)  # 5 target values per sequence
        assert feature_cols == ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Check that sequences are properly aligned
        assert np.array_equal(X[0], data[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[0:3].values)
        assert np.array_equal(y[0], data[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[3:4].values.flatten())
    
    def test_create_sequences_multiple_predictions(self):
        """Test sequence creation with multiple prediction steps"""
        data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'High': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'Low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'Close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5, 110.5],
            'Volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
        })
        
        X, y, feature_cols = create_sequences(data, sequence_length=3, prediction_length=2)
        
        assert X.shape == (7, 3, 5)  # 11 - 3 - 2 + 1 = 7 sequences
        assert y.shape == (7, 10)  # 2 timesteps * 5 features = 10 target values
        assert feature_cols == ['Open', 'High', 'Low', 'Close', 'Volume']
    
    def test_create_sequences_custom_targets(self):
        """Test sequence creation with custom target columns"""
        data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'High': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'Low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            'Close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
            'Volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        })
        
        # Test with only price targets
        X, y, feature_cols = create_sequences(
            data, 
            sequence_length=3, 
            prediction_length=1,
            target_cols=['Open', 'High', 'Low', 'Close']
        )
        
        assert X.shape == (7, 3, 5)  # Still uses all features for input
        assert y.shape == (7, 4)  # Only 4 target values (4 columns * 1 timestep)
        assert feature_cols == ['Open', 'High', 'Low', 'Close', 'Volume']
    
    def test_create_sequences_validation_empty_data(self):
        """Test validation with empty data"""
        with pytest.raises(ValueError, match="Input data is empty or None"):
            create_sequences(pd.DataFrame(), 3, 1)
        
        with pytest.raises(ValueError, match="Input data is empty or None"):
            create_sequences(None, 3, 1)
    
    def test_create_sequences_validation_missing_columns(self):
        """Test validation with missing required columns"""
        # Missing some columns
        bad_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [101, 102, 103]
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            create_sequences(bad_data, 3, 1)
    
    def test_create_sequences_validation_insufficient_data(self):
        """Test validation with insufficient data"""
        # Not enough data for sequence length + prediction length
        small_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [101, 102, 103],
            'Low': [99, 100, 101],
            'Close': [100.5, 101.5, 102.5],
            'Volume': [1000, 1100, 1200]
        })
        
        with pytest.raises(ValueError, match="Not enough data after cleaning"):
            create_sequences(small_data, 3, 1)  # Need at least 4 rows (3 + 1)
    
    def test_create_sequences_with_nan_values(self):
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
        assert y.shape[0] == 6
        assert y.shape[1] == 5
    
    def test_create_sequences_data_types(self):
        """Test that output data types are correct"""
        data = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            'High': [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
            'Low': [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
            'Close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
            'Volume': [1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0, 1900.0]
        })
        
        X, y, feature_cols = create_sequences(data, sequence_length=3, prediction_length=1)
        
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(feature_cols, list)
        assert X.dtype == np.float64
        assert y.dtype == np.float64
        assert all(isinstance(col, str) for col in feature_cols)


class TestDownloadBinanceData:
    """Test download_binance_data function"""
    
    @patch('utils._download_single_month')
    def test_download_binance_data_single_month(self, mock_download_single):
        """Test download_binance_data with single month"""
        # Mock single month download (with all columns that _download_single_month returns)
        mock_data = pd.DataFrame({
            'Open time': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03']),
            'Open': [100, 101, 102],
            'High': [101, 102, 103],
            'Low': [99, 100, 101],
            'Close': [100.5, 101.5, 102.5],
            'Volume': [1000, 1100, 1200],
            'Close time': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03']),
            'Quote asset volume': [2000, 2100, 2200],
            'Number of trades': [10, 11, 12],
            'Taker buy base asset volume': [500, 550, 600],
            'Taker buy quote asset volume': [600, 650, 700],
            'Ignore': [0, 0, 0]
        })
        mock_download_single.return_value = mock_data
        
        result = download_binance_data('BTCUSDT', '5m', '2021-01', '2021-01')
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']
        assert len(result) == 3
        mock_download_single.assert_called_once_with('BTCUSDT', '5m', '2021', '01')
    
    @patch('utils._download_single_month')
    def test_download_binance_data_multiple_months(self, mock_download_single):
        """Test download_binance_data with multiple months"""
        # Mock data for different months (with all columns that _download_single_month returns)
        mock_data1 = pd.DataFrame({
            'Open time': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03']),
            'Open': [100, 101, 102],
            'High': [101, 102, 103],
            'Low': [99, 100, 101],
            'Close': [100.5, 101.5, 102.5],
            'Volume': [1000, 1100, 1200],
            'Close time': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03']),
            'Quote asset volume': [2000, 2100, 2200],
            'Number of trades': [10, 11, 12],
            'Taker buy base asset volume': [500, 550, 600],
            'Taker buy quote asset volume': [600, 650, 700],
            'Ignore': [0, 0, 0]
        })
        
        mock_data2 = pd.DataFrame({
            'Open time': pd.to_datetime(['2021-02-01', '2021-02-02', '2021-02-03']),
            'Open': [103, 104, 105],
            'High': [104, 105, 106],
            'Low': [102, 103, 104],
            'Close': [103.5, 104.5, 105.5],
            'Volume': [1300, 1400, 1500],
            'Close time': pd.to_datetime(['2021-02-01', '2021-02-02', '2021-02-03']),
            'Quote asset volume': [2300, 2400, 2500],
            'Number of trades': [13, 14, 15],
            'Taker buy base asset volume': [700, 750, 800],
            'Taker buy quote asset volume': [800, 850, 900],
            'Ignore': [0, 0, 0]
        })
        
        mock_download_single.side_effect = [mock_data1, mock_data2]
        
        result = download_binance_data('BTCUSDT', '5m', '2021-01', '2021-02')
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']
        assert len(result) == 6  # 3 + 3
        assert mock_download_single.call_count == 2
    
    @patch('utils._download_single_month')
    def test_download_binance_data_no_data(self, mock_download_single):
        """Test download_binance_data when no data is downloaded"""
        mock_download_single.return_value = None
        
        with pytest.raises(ValueError, match="No data downloaded"):
            download_binance_data('BTCUSDT', '5m', '2021-01', '2021-01')
    
    @patch('utils._download_single_month')
    def test_download_binance_data_error_handling(self, mock_download_single):
        """Test download_binance_data error handling"""
        mock_download_single.side_effect = Exception("Network error")
        
        with pytest.raises(ValueError, match="Download failed"):
            download_binance_data('BTCUSDT', '5m', '2021-01', '2021-01')
    
    def test_download_binance_data_date_parsing(self):
        """Test download_binance_data date parsing"""
        with pytest.raises(ValueError, match="Download failed"):
            # Invalid date format should cause parsing error
            download_binance_data('BTCUSDT', '5m', 'invalid-date', '2021-01')


class TestDownloadSingleMonth:
    """Test _download_single_month function"""
    
    @patch('requests.get')
    @patch('zipfile.ZipFile')
    def test_download_single_month_success(self, mock_zipfile, mock_get):
        """Test successful single month download"""
        # Mock response
        mock_response = MagicMock()
        mock_response.content = b'fake zip content'
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Mock zipfile
        mock_zip = MagicMock()
        mock_zip.namelist.return_value = ['data.csv']
        
        # Create a proper CSV string with all required columns
        csv_data = '1609459200000,100,101,99,100.5,1000,1609459260000,2000,10,500,600,0\n'
        
        # Mock the file object properly
        mock_file = MagicMock()
        mock_file.__enter__.return_value = io.StringIO(csv_data)
        mock_zip.open.return_value = mock_file
        mock_zipfile.return_value.__enter__.return_value = mock_zip
        
        result = _download_single_month('BTCUSDT', '5m', '2021', '01')
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        # _download_single_month returns all columns, not just OHLCV
        expected_columns = [
            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close time', 'Quote asset volume', 'Number of trades',
            'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
        ]
        assert list(result.columns) == expected_columns
        assert len(result) == 1
    
    @patch('requests.get')
    def test_download_single_month_network_error(self, mock_get):
        """Test single month download with network error"""
        mock_get.side_effect = Exception("Network error")
        
        with pytest.raises(ValueError, match="Error downloading 2021-01: Network error"):
            _download_single_month('BTCUSDT', '5m', '2021', '01')
    
    @patch('requests.get')
    @patch('zipfile.ZipFile')
    def test_download_single_month_no_csv(self, mock_zipfile, mock_get):
        """Test single month download with no CSV file"""
        # Mock response
        mock_response = MagicMock()
        mock_response.content = b'fake zip content'
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Mock zipfile with no CSV files
        mock_zip = MagicMock()
        mock_zip.namelist.return_value = ['data.txt']
        mock_zipfile.return_value.__enter__.return_value = mock_zip
        
        result = _download_single_month('BTCUSDT', '5m', '2021', '01')
        
        assert result is None


if __name__ == '__main__':
    pytest.main([__file__])
