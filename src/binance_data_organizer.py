import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, TYPE_CHECKING
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler
import warnings
from .utils import download_binance_data, create_sequences

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning)

if TYPE_CHECKING:
    import tensorflow as tf

@dataclass
class DataConfig:
    symbol: str
    timeframe: str
    start_time: str
    end_time: str
    sequence_length: int
    prediction_length: int
    train_split: float = 0.8

class GroupedScaler:
    def __init__(self):
        self.scalers = {}
        self.feature_groups = {}
        self.feature_names = []
        self.is_fitted = False
    
    def _categorize_features(self, feature_names: List[str]) -> Dict[str, List[int]]:
        groups = {
            'ohlc': [],
            'volume': []
        }
        
        for i, col in enumerate(feature_names):
            col_lower = col.lower()
            
            if col_lower in ['open', 'high', 'low', 'close']:
                groups['ohlc'].append(i)
            elif col_lower == 'volume':
                groups['volume'].append(i)
        
        return {k: v for k, v in groups.items() if v}
    
    
    def fit(self, X: np.ndarray, feature_names: List[str]) -> 'GroupedScaler':
        self.feature_names = feature_names
        
        if X.ndim == 3:
            X_reshaped = X.reshape(-1, X.shape[-1])
        else:
            X_reshaped = X
        
        self.feature_groups = self._categorize_features(feature_names)
        
        for group_name, feature_indices in self.feature_groups.items():
            group_data = X_reshaped[:, feature_indices]
            
            if group_name == 'ohlc':
                # OHLC scaled together
                scaler = MinMaxScaler()
                scaler.fit(group_data)
                self.scalers[group_name] = scaler
                
            elif group_name == 'volume':
                # Volume scaled separately
                scaler = MinMaxScaler()
                scaler.fit(group_data)
                self.scalers[group_name] = scaler
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        if X.ndim == 3:
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_scaled = X_reshaped.copy()
        else:
            X_reshaped = X
            X_scaled = X.copy()
        
        for group_name, feature_indices in self.feature_groups.items():
            scaler = self.scalers[group_name]
            X_scaled[:, feature_indices] = scaler.transform(X_reshaped[:, feature_indices])
        
        if X.ndim == 3:
            X_scaled = X_scaled.reshape(X.shape)
        
        return X_scaled
    
    def fit_transform(self, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        return self.fit(X, feature_names).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse_transform")
        
        if X.ndim == 3:
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_inverse = X_reshaped.copy()
        else:
            X_reshaped = X
            X_inverse = X.copy()
        
        for group_name, feature_indices in self.feature_groups.items():
            scaler = self.scalers[group_name]
            X_inverse[:, feature_indices] = scaler.inverse_transform(X_reshaped[:, feature_indices])
        
        if X.ndim == 3:
            X_inverse = X_inverse.reshape(X.shape)
        
        return X_inverse
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted first")
        
        return {
            group_name: [self.feature_names[i] for i in feature_indices]
            for group_name, feature_indices in self.feature_groups.items()
        }


def scale_data(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, GroupedScaler, MinMaxScaler]:
    scaler_X = GroupedScaler()
    X_train_scaled = scaler_X.fit_transform(X_train, feature_cols)
    X_test_scaled = scaler_X.transform(X_test)
    
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y


class BinanceDataOrganizer:
    def __init__(self, config: DataConfig):
        self.config = config
        self.scaler_X: Optional[GroupedScaler] = None
        self.scaler_y: Optional[MinMaxScaler] = None
        self.is_scalers_fitted = False
        self.raw_data = download_binance_data(
            symbol=self.config.symbol,
            interval=self.config.timeframe,
            data_from=self.config.start_time,
            data_to=self.config.end_time
        )
        
        # Validate data format
        if self.raw_data is not None:
            expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in self.raw_data.columns for col in expected_cols):
                raise ValueError(f"Data must contain columns: {expected_cols}")
    
    def process_all(self) -> bool:
        # Data is already processed during download (OHLCV only)
        return self.raw_data is not None
    
    def _prepare_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        return create_sequences(
            data=data,
            sequence_length=self.config.sequence_length,
            prediction_length=self.config.prediction_length
        )
    
    def _fit_scalers(self, X_train: np.ndarray, y_train: np.ndarray, feature_columns: List[str]) -> None:
        self.scaler_X = GroupedScaler()
        self.scaler_X.fit(X_train, feature_columns)
        
        # Target scaling: allow values beyond [0,1] range for better model performance
        self.scaler_y = MinMaxScaler()
        self.scaler_y.fit(y_train)
        
        self.is_scalers_fitted = True
    
    def get_unscaled_data(self, data_type: str = 'all') -> Dict[str, np.ndarray]:
        if self.raw_data is None:
            raise ValueError("No data available")
        
        X, y, feature_cols = self._prepare_sequences(self.raw_data)
        
        split_idx = int(len(X) * self.config.train_split)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        result = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
        
        if data_type == 'train':
            return {'X_train': X_train, 'y_train': y_train}
        elif data_type == 'test':
            return {'X_test': X_test, 'y_test': y_test}
        else:
            return result
    
    def get_scaled_data(self, data_type: str = 'all') -> Dict[str, np.ndarray]:
        unscaled_data = self.get_unscaled_data(data_type)
        
        if not self.is_scalers_fitted:
            X_train = unscaled_data['X_train']
            y_train = unscaled_data['y_train']
            _, _, feature_cols = self._prepare_sequences(self.raw_data)
            self._fit_scalers(X_train, y_train, feature_cols)
        
        if data_type == 'train':
            X_train_scaled = self.scaler_X.transform(unscaled_data['X_train'])
            y_train_scaled = self.scaler_y.transform(unscaled_data['y_train'])
            return {
                'X_train_scaled': X_train_scaled,
                'y_train_scaled': y_train_scaled
            }
        elif data_type == 'test':
            X_test_scaled = self.scaler_X.transform(unscaled_data['X_test'])
            y_test_scaled = self.scaler_y.transform(unscaled_data['y_test'])
            return {
                'X_test_scaled': X_test_scaled,
                'y_test_scaled': y_test_scaled
            }
        else:
            X_train_scaled = self.scaler_X.transform(unscaled_data['X_train'])
            X_test_scaled = self.scaler_X.transform(unscaled_data['X_test'])
            y_train_scaled = self.scaler_y.transform(unscaled_data['y_train'])
            y_test_scaled = self.scaler_y.transform(unscaled_data['y_test'])
            return {
                'X_train_scaled': X_train_scaled,
                'X_test_scaled': X_test_scaled,
                'y_train_scaled': y_train_scaled,
                'y_test_scaled': y_test_scaled
            }
    
    def get_data_in_range(self, start_time: str, end_time: str, scaled: bool = True) -> Optional[Dict[str, np.ndarray]]:
        if self.raw_data is None:
            raise ValueError("No data available")
        
        # Since we don't have time columns, return all data
        filtered_data = self.raw_data
        
        if len(filtered_data) < self.config.sequence_length:
            return None
        
        X, y, feature_cols = self._prepare_sequences(filtered_data)
        
        result = {'X': X, 'y': y}
        
        if scaled and self.is_scalers_fitted:
            X_scaled = self.scaler_X.transform(X)
            y_scaled = self.scaler_y.transform(y)
            result.update({'X_scaled': X_scaled, 'y_scaled': y_scaled})
        
        return result
    
    def get_scalers(self) -> Dict[str, Union[GroupedScaler, MinMaxScaler]]:
        if not self.is_scalers_fitted:
            raise ValueError("Scalers not fitted. Call get_scaled_data() first.")
        
        return {'X': self.scaler_X, 'y': self.scaler_y}
    
    def inverse_transform_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        """Inverse transform target values from scaled to original scale"""
        if not self.is_scalers_fitted:
            raise ValueError("Scalers not fitted. Call get_scaled_data() first.")
        
        return self.scaler_y.inverse_transform(y_scaled)
    
    def get_feature_info(self) -> Dict[str, Union[List[str], int, Tuple[int, int]]]:
        if self.raw_data is None:
            raise ValueError("No data available")
        
        _, _, feature_cols = self._prepare_sequences(self.raw_data)
        
        return {
            'feature_columns': feature_cols,
            'num_features': len(feature_cols),
            'sequence_length': self.config.sequence_length,
            'prediction_length': self.config.prediction_length,
            'data_shape': self.raw_data.shape,
            'total_sequences': len(self.raw_data) - self.config.sequence_length - self.config.prediction_length + 1
        }
    
    def get_data_summary(self) -> Dict[str, Union[str, int, float]]:
        """Get summary information about the loaded data"""
        if self.raw_data is None:
            raise ValueError("No data available")
        
        return {
            'symbol': self.config.symbol,
            'timeframe': self.config.timeframe,
            'date_range': f"{self.config.start_time} to {self.config.end_time}",
            'total_rows': len(self.raw_data),
            'columns': list(self.raw_data.columns),
            'price_range': {
                'min': self.raw_data[['Open', 'High', 'Low', 'Close']].min().min(),
                'max': self.raw_data[['Open', 'High', 'Low', 'Close']].max().max()
            },
            'volume_range': {
                'min': self.raw_data['Volume'].min(),
                'max': self.raw_data['Volume'].max()
            }
        }
    
    def get_sequence_info(self) -> Dict[str, int]:
        """Get information about sequence generation"""
        if self.raw_data is None:
            raise ValueError("No data available")
        
        total_sequences = len(self.raw_data) - self.config.sequence_length - self.config.prediction_length + 1
        train_sequences = int(total_sequences * self.config.train_split)
        test_sequences = total_sequences - train_sequences
        
        return {
            'total_sequences': total_sequences,
            'train_sequences': train_sequences,
            'test_sequences': test_sequences,
            'sequence_length': self.config.sequence_length,
            'prediction_length': self.config.prediction_length,
            'features_per_timestep': 5  # OHLCV
        }

