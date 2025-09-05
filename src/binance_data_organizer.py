#!/usr/bin/env python3

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, TYPE_CHECKING
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler
import warnings
import zipfile
import io

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
    max_rows: int = 50000
    train_split: float = 0.8


def download_binance_data(symbol: str, interval: str, data_from: str, data_to: str, max_rows: int = 50000) -> Optional[pd.DataFrame]:
    print("📥 DOWNLOADING CRYPTOCURRENCY DATA")
    print(f"   Symbol: {symbol}")
    print(f"   Interval: {interval}")
    print(f"   Date range: {data_from} to {data_to}")
    print(f"   Max rows: {max_rows:,}")
    print("=" * 50)
    
    year, months = _parse_date_range(data_from, data_to)
    print(f"📅 Date range: {year} months {months[0]} to {months[-1]}")
    print(f"📊 Will download months: {months}")
    
    df = _load_multiple_months_data(symbol, interval, year, months, max_rows)
    
    if df is not None and not df.empty:
        print(f"\n✅ Download completed successfully!")
        print(f"📊 Final data shape: {df.shape}")
        print(f"📅 Date range: {df['Open time'].min()} to {df['Open time'].max()}")
        return df
    else:
        print("❌ Download failed!")
        return None


def _parse_date_range(data_from: str, data_to: str) -> Tuple[str, List[str]]:
    try:
        if ' ' in data_from:
            start_date = datetime.strptime(data_from, '%Y %m')
        else:
            start_date = datetime.strptime(data_from, '%Y-%m-%d')
        
        if ' ' in data_to:
            end_date = datetime.strptime(data_to, '%Y %m')
        else:
            end_date = datetime.strptime(data_to, '%Y-%m-%d')
        
        months = []
        current = start_date.replace(day=1)
        end = end_date.replace(day=1)
        
        while current <= end:
            months.append(current.strftime('%m'))
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        
        year = start_date.strftime('%Y')
        return year, months
        
    except ValueError as e:
        print(f"❌ Error parsing date range: {e}")
        return "2021", ["01"]


def _load_multiple_months_data(symbol: str, interval: str, year: str, months: List[str], max_rows: int = 50000) -> Optional[pd.DataFrame]:
    print(f"🔄 Loading data with memory efficiency (max {max_rows:,} rows)...")
    
    all_data = []
    total_rows = 0
    
    for month in months:
        if total_rows >= max_rows:
            print(f"   Reached memory limit of {max_rows:,} rows, stopping...")
            break
            
        df = _download_binance_vision_data(symbol, interval, year, month)
        if df is not None:
            remaining_capacity = max_rows - total_rows
            if len(df) > remaining_capacity:
                df = df.head(remaining_capacity)
                print(f"   Truncated to {remaining_capacity} rows to stay within memory limit")
            
            all_data.append(df)
            total_rows += len(df)
            print(f"   Total rows: {total_rows:,}")
    
    if not all_data:
        print("❌ No data loaded")
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sort_values('Open time').reset_index(drop=True)
    
    print(f"✅ Data loading completed!")
    print(f"   Final shape: {combined_df.shape}")
    
    return combined_df


def _download_binance_vision_data(symbol: str, interval: str, year: str, month: str) -> Optional[pd.DataFrame]:
    try:
        month = month.zfill(2)
        url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{year}-{month}.zip"
        
        print(f"📥 Downloading {symbol} {interval} data for {year}-{month} from Binance Vision...")
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
            if not csv_files:
                print(f"❌ No CSV file found in {symbol}-{interval}-{year}-{month}.zip")
                return None
            
            with z.open(csv_files[0]) as f:
                df = pd.read_csv(f, header=None)
        
        df.columns = [
            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close time', 'Quote asset volume', 'Number of trades',
            'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
        ]
        
        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
        df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')
        
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote asset volume', 
                       'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.drop('Ignore', axis=1)
        
        print(f"✅ Downloaded {len(df)} rows for {year}-{month}")
        return df
        
    except Exception as e:
        print(f"❌ Error downloading {symbol} data for {year}-{month}: {e}")
        return None


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    
    data['Minutes_of_day'] = data['Open time'].dt.hour * 60 + data['Open time'].dt.minute
    data['Price_Range'] = data['High'] - data['Low']
    data['Price_Change'] = data['Close'] - data['Open']
    data['Price_Change_Pct'] = data['Price_Change'] / data['Open']
    data['Volume_MA_5'] = data['Volume'].rolling(window=5).mean()
    data['Volume_MA_10'] = data['Volume'].rolling(window=10).mean()
    
    return data


def create_sequences(data: pd.DataFrame, sequence_length: int, prediction_length: int, target_cols: List[str] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if target_cols is None:
        target_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    feature_cols = [col for col in data.columns if col not in ['Open time', 'Close time'] + target_cols]
    data_clean = data.dropna()
    
    if len(data_clean) < sequence_length:
        raise ValueError(f"Not enough data after cleaning. Need at least {sequence_length} rows, got {len(data_clean)}")
    
    X, y = [], []
    for i in range(len(data_clean) - sequence_length - prediction_length + 1):
        X.append(data_clean[feature_cols].iloc[i:i+sequence_length].values)
        y.append(data_clean[target_cols].iloc[i+sequence_length:i+sequence_length+prediction_length].values.flatten())
    
    return np.array(X), np.array(y), feature_cols


class GroupedScaler:
    def __init__(self):
        self.scalers = {}
        self.feature_groups = {}
        self.feature_names = []
        self.is_fitted = False
    
    def _categorize_features(self, feature_names: List[str]) -> Dict[str, List[int]]:
        groups = {
            'price': [],
            'volume': [],
            'time': [],
            'other': []
        }
        
        for i, col in enumerate(feature_names):
            col_lower = col.lower()
            
            if any(price_word in col_lower for price_word in 
                   ['price', 'open', 'high', 'low', 'close', 'change', 'range']):
                groups['price'].append(i)
            elif any(vol_word in col_lower for vol_word in 
                     ['volume', 'taker', 'trades']):
                groups['volume'].append(i)
            elif any(time_word in col_lower for time_word in 
                     ['minutes_of_day', 'hour', 'minute', 'interval', 'sin', 'cos']):
                groups['time'].append(i)
            else:
                groups['other'].append(i)
        
        return {k: v for k, v in groups.items() if v}
    
    def fit(self, X: np.ndarray, feature_names: List[str]) -> 'GroupedScaler':
        self.feature_names = feature_names
        
        if X.ndim == 3:
            X_reshaped = X.reshape(-1, X.shape[-1])
        else:
            X_reshaped = X
        
        self.feature_groups = self._categorize_features(feature_names)
        
        for group_name, feature_indices in self.feature_groups.items():
            scaler = MinMaxScaler()
            group_data = X_reshaped[:, feature_indices]
            
            if group_name == 'time':
                time_feature_names = [feature_names[i] for i in feature_indices]
                if any('minutes_of_day' in name.lower() for name in time_feature_names):
                    print(f"   ⏰ Using fixed range [0, 1439] for Minutes_of_day feature")
                    scaler.fit([[0], [1439]])
                else:
                    scaler.fit(group_data)
            else:
                scaler.fit(group_data)
            
            self.scalers[group_name] = scaler
            print(f"✅ {group_name.capitalize()} group scaler fitted on {len(feature_indices)} features")
        
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
    
    def get_scaling_info(self) -> Dict[str, Dict[str, float]]:
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted first")
        
        info = {}
        for group_name, scaler in self.scalers.items():
            info[group_name] = {
                'min_': scaler.min_,
                'scale_': scaler.scale_,
                'data_min_': scaler.data_min_,
                'data_max_': scaler.data_max_
            }
        
        return info


def scale_data(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, GroupedScaler, MinMaxScaler]:
    print("🔢 Scaling time series data with grouped normalization...")
    
    scaler_X = GroupedScaler()
    X_train_scaled = scaler_X.fit_transform(X_train, feature_cols)
    X_test_scaled = scaler_X.transform(X_test)
    
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    print("✅ Grouped scaling completed!")
    print(f"📊 Scaled X_train shape: {X_train_scaled.shape}")
    print(f"📊 Scaled y_train shape: {y_train_scaled.shape}")
    print(f"📊 X_train range: {X_train_scaled.min():.4f} to {X_train_scaled.max():.4f}")
    print(f"📊 y_train range: {y_train_scaled.min():.4f} to {y_train_scaled.max():.4f}")
    
    feature_groups = scaler_X.get_feature_groups()
    for group_name, features in feature_groups.items():
        print(f"📊 {group_name.capitalize()} group ({len(features)}): {features}")
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y


class BinanceDataOrganizer:
    def __init__(self, config: DataConfig):
        self.config = config
        self.raw_data: Optional[pd.DataFrame] = None
        self.features_data: Optional[pd.DataFrame] = None
        self.scaler_X: Optional[GroupedScaler] = None
        self.scaler_y: Optional[MinMaxScaler] = None
        self.is_features_created = False
        self.is_scalers_fitted = False
    
    def load_data(self) -> bool:
        print(f"🔄 Loading Binance data for {self.config.symbol}...")
        print(f"   Timeframe: {self.config.timeframe}")
        print(f"   Period: {self.config.start_time} to {self.config.end_time}")
        print(f"   Max rows: {self.config.max_rows}")
        
        start_date = self.config.start_time.replace('-', ' ')
        end_date = self.config.end_time.replace('-', ' ')
        
        self.raw_data = download_binance_data(
            symbol=self.config.symbol,
            interval=self.config.timeframe,
            data_from=start_date,
            data_to=end_date,
            max_rows=self.config.max_rows
        )
        
        if self.raw_data is not None and not self.raw_data.empty:
            print(f"✅ Data loaded successfully!")
            print(f"   Shape: {self.raw_data.shape}")
            print(f"   Date range: {self.raw_data['Open time'].min()} to {self.raw_data['Open time'].max()}")
            return True
        else:
            print("❌ Failed to load data!")
            return False
    
    def create_features(self) -> bool:
        if self.raw_data is None:
            print("❌ No raw data available. Call load_data() first.")
            return False
        
        print("🔧 Creating time series features...")
        self.features_data = create_features(self.raw_data)
        self.is_features_created = True
        
        print(f"✅ Features created! Shape: {self.features_data.shape}")
        return True
    
    def process_all(self) -> bool:
        print("🚀 Starting complete data processing pipeline...")
        
        if not self.load_data():
            return False
        
        if not self.create_features():
            return False
        
        print("✅ Complete data processing pipeline finished!")
        print("💡 Use get_unscaled_data() or get_scaled_data() to generate sequences on-demand.")
        return True
    
    def _prepare_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        return create_sequences(
            data=data,
            sequence_length=self.config.sequence_length,
            prediction_length=self.config.prediction_length
        )
    
    def _fit_scalers(self, X_train: np.ndarray, y_train: np.ndarray, feature_columns: List[str]) -> None:
        print("🔢 Fitting scalers on training data...")
        
        self.scaler_X = GroupedScaler()
        self.scaler_X.fit(X_train, feature_columns)
        
        self.scaler_y = MinMaxScaler()
        self.scaler_y.fit(y_train)
        
        self.is_scalers_fitted = True
        print("✅ Scalers fitted!")
    
    def get_unscaled_data(self, data_type: str = 'all') -> Dict[str, np.ndarray]:
        if not self.is_features_created:
            raise ValueError("No features created. Call create_features() first.")
        
        X, y, feature_cols = self._prepare_sequences(self.features_data)
        
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
            _, _, feature_cols = self._prepare_sequences(self.features_data)
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
        if not self.is_features_created:
            raise ValueError("No features created. Call create_features() first.")
        
        start_dt = pd.to_datetime(start_time)
        end_dt = pd.to_datetime(end_time)
        
        mask = (self.features_data['Open time'] >= start_dt) & (self.features_data['Open time'] <= end_dt)
        filtered_data = self.features_data[mask]
        
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
    
    def get_feature_info(self) -> Dict[str, Union[List[str], int, Tuple[int, int]]]:
        if not self.is_features_created:
            raise ValueError("No features created. Call create_features() first.")
        
        _, _, feature_cols = self._prepare_sequences(self.features_data)
        
        return {
            'feature_columns': feature_cols,
            'num_features': len(feature_cols),
            'sequence_length': self.config.sequence_length,
            'prediction_length': self.config.prediction_length,
            'data_shape': self.features_data.shape,
            'total_sequences': len(self.features_data) - self.config.sequence_length - self.config.prediction_length + 1
        }


# Legacy compatibility
def create_minimal_features(df: pd.DataFrame, lag_period: int = 3) -> pd.DataFrame:
    return create_features(df)

def create_sliding_windows(data: pd.DataFrame, sequence_length: int = 5, target_cols: List[str] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    return create_sequences(data, sequence_length, 1, target_cols)

def download_crypto_data(symbol: str, interval: str, data_from: str, data_to: str, max_rows: int = 50000) -> Optional[pd.DataFrame]:
    return download_binance_data(symbol, interval, data_from, data_to, max_rows)

def scale_time_series_data_grouped(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, GroupedScaler, MinMaxScaler]:
    return scale_data(X_train, X_test, y_train, y_test, feature_cols)

def predict_with_grouped_scaler(model: 'tf.keras.Model', new_data_df: pd.DataFrame, scaler_X: GroupedScaler, scaler_y: MinMaxScaler, feature_cols: List[str], timesteps: int = 1) -> Optional[pd.DataFrame]:
    if new_data_df is None or new_data_df.empty:
        print("Error: No new data provided for prediction.")
        return None
    
    features_df = create_features(new_data_df)
    features_df = features_df.dropna()
    features = features_df[feature_cols].values
    
    if len(features) < timesteps:
        print(f"Error: Need at least {timesteps} rows for prediction, got {len(features)}")
        return None
    
    latest_features = features[-timesteps:]
    latest_features_scaled = scaler_X.transform(latest_features.reshape(1, timesteps, -1))
    latest_features_lstm = latest_features_scaled.reshape((1, timesteps, latest_features_scaled.shape[-1]))
    
    predictions_scaled = model.predict(latest_features_lstm, verbose=0)
    predictions = scaler_y.inverse_transform(predictions_scaled)
    
    target_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    prediction_df = pd.DataFrame(predictions, columns=target_columns)
    
    return prediction_df