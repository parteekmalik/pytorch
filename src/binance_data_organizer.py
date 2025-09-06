import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import MinMaxScaler
import warnings

from src.config import BaseConfig
from .utils import download_binance_data, create_sequences

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning)

class OHLCPaddedScaler:
    """
    Scaler that groups OHLC features together with padding for unified scaling.
    
    LOGIC:
    - OHLC features (Open, High, Low, Close) are treated as a single group
    - All OHLC values are combined into one flat array and scaled together
    - Input data is scaled to range [padding, 1-padding] to leave room for output
    - Target data is scaled using same min/max as input but clamped to [0, 1] range
    - This preserves relative relationships between Open, High, Low, Close within each sequence
    - Each sequence is scaled independently
    
    WHY THIS APPROACH:
    - Prevents High-Low relationship distortion that occurs when scaling OHLC separately
    - Maintains candlestick chart integrity after scaling
    - Each sequence gets its own min/max range for optimal scaling
    - Padding prevents output from exceeding valid range while allowing flexibility

    INPUT FORMAT:
    - input_sequences: 3D array (num_sequences, sequence_length, 4) - OHLC historical data
    - target_sequences: 3D array (num_sequences, prediction_length, 3) - HLC future data
    
    OUTPUT FORMAT:
    - Returns same shapes as input
    - Input values scaled to [padding, 1-padding] range
    - Target values scaled using input min/max but clamped to [0, 1] range
    """
    
    def __init__(self, padding_factor: float = 0.5):
        """
        Initialize scaler with padding factor.
        
        Args:
            padding_factor: Factor to reduce scaling range (e.g., 0.5 means scale to [0.25, 0.75])
        """
        self.padding_factor = padding_factor
        self.padding_per_side = padding_factor / 2.0
    
    def _validate_input(self, input_sequences: np.ndarray, target_sequences: np.ndarray) -> None:
        """Validate input format before processing."""
        if input_sequences.ndim != 3:
            raise ValueError(f"Expected 3D input array, got {input_sequences.ndim}D array")
        
        if target_sequences.ndim != 2:
            raise ValueError(f"Expected 2D target array (flattened), got {target_sequences.ndim}D array")
        
        if input_sequences.shape[0] != target_sequences.shape[0]:
            raise ValueError(f"Number of sequences mismatch: input={input_sequences.shape[0]}, target={target_sequences.shape[0]}")
        
        if input_sequences.shape[2] != 4:
            raise ValueError(f"Expected 4 features (OHLC) in input, got {input_sequences.shape[2]} features")
        
        # Check if target can be reshaped to (num_sequences, prediction_length, 3)
        if target_sequences.shape[1] % 3 != 0:
            raise ValueError(f"Target array second dimension must be divisible by 3 (HLC), got {target_sequences.shape[1]}")
        
        if input_sequences.size == 0 or target_sequences.size == 0:
            raise ValueError("Cannot scale empty arrays")
        
        if np.any(np.isnan(input_sequences)) or np.any(np.isinf(input_sequences)):
            raise ValueError("Input array contains NaN or infinite values")
        
        if np.any(np.isnan(target_sequences)) or np.any(np.isinf(target_sequences)):
            raise ValueError("Target array contains NaN or infinite values")
    
    def scale_sequences_with_padding(self, input_sequences: np.ndarray, target_sequences: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale both input and target sequences with padding.
        
        PROCESS:
        1. For each sequence pair: find min/max from input OHLC data
        2. Scale input to range [padding, 1-padding] (e.g., [0.25, 0.75] if padding=0.5)
        3. Scale target using same min/max values but clamp to [0, 1] range
        4. This ensures target can't exceed valid range while maintaining relationships
        
        Args:
            input_sequences: 3D array (num_sequences, sequence_length, 4) - OHLC historical data
            target_sequences: 2D array (num_sequences, prediction_length * 3) - flattened HLC future data
            
        Returns:
            Tuple of (scaled_input_sequences, scaled_target_sequences) with same shapes as input
        """
        self._validate_input(input_sequences, target_sequences)
        
        scaled_input_sequences = []
        scaled_target_sequences = []
        
        # Calculate prediction_length from target shape
        prediction_length = target_sequences.shape[1] // 3
        
        for i in range(len(input_sequences)):
            input_sequence = input_sequences[i]  # Shape: (sequence_length, 4)
            target_sequence = target_sequences[i]  # Shape: (prediction_length * 3,)
            
            # Reshape target to (prediction_length, 3) for processing
            target_reshaped = target_sequence.reshape(prediction_length, 3)
            
            # Step 1: Find min/max from input OHLC data
            ohlc_data = input_sequence  # All 4 columns are OHLC
            ohlc_min = ohlc_data.min()
            ohlc_max = ohlc_data.max()
            
            # Step 2: Scale input to [padding, 1-padding] range
            if ohlc_max == ohlc_min:
                # Handle case where all values are the same
                scaled_input = np.full_like(input_sequence, self.padding_per_side)
            else:
                # Normal scaling to [padding_per_side, 1-padding_per_side] range
                scaled_input = (input_sequence - ohlc_min) / (ohlc_max - ohlc_min)
                scaled_input = scaled_input * (1 - self.padding_factor) + self.padding_per_side
            
            # Step 3: Scale target using same min/max and same range as input
            if ohlc_max == ohlc_min:
                # Handle case where all values are the same
                scaled_target = np.full_like(target_reshaped, self.padding_per_side)
            else:
                # Scale target using input's min/max values to same range as input
                scaled_target = (target_reshaped - ohlc_min) / (ohlc_max - ohlc_min)
                scaled_target = scaled_target * (1 - self.padding_factor) + self.padding_per_side
                # Clamp to [0, 1] range to prevent exceeding valid range
                scaled_target = np.clip(scaled_target, 0.0, 1.0)
            
            # Flatten target back to original format
            scaled_target_flat = scaled_target.flatten()
            
            scaled_input_sequences.append(scaled_input)
            scaled_target_sequences.append(scaled_target_flat)
        
        return np.array(scaled_input_sequences), np.array(scaled_target_sequences)


class BinanceDataOrganizer:
    def __init__(self, config: BaseConfig):
        self.config = config
        self.custom_scaler = OHLCPaddedScaler(padding_factor=config.scaling_padding_factor)
        self.raw_data = download_binance_data(
            symbol=self.config.symbol,
            interval=self.config.timeframe,
            data_from=self.config.start_date,
            data_to=self.config.end_date
        )
        
        # Validate data format
        if self.raw_data is not None:
            expected_cols = ['Open', 'High', 'Low', 'Close']
            if not all(col in self.raw_data.columns for col in expected_cols):
                raise ValueError(f"Data must contain columns: {expected_cols}")
    
    def _prepare_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        return create_sequences(
            data,
            self.config.sequence_length,
            self.config.prediction_length
        )
    
    def get_unscaled_split_data(self) -> Dict[str, np.ndarray]:
        input, output, feature_cols = self._prepare_sequences(self.raw_data)
        
        split_idx = int(len(input) * self.config.train_split)
        input_train, input_test = input[:split_idx], input[split_idx:]
        output_train, output_test = output[:split_idx], output[split_idx:]
        
        return {
            'input_train': input_train,
            'input_test': input_test,
            'output_train': output_train,
            'output_test': output_test
        }
    
    def get_scaled_data(self) -> Dict[str, np.ndarray]:
        unscaled_data = self.get_unscaled_split_data()

        # Scale both input and target data using the new unified scaler
        input_train_scaled, output_train_scaled = self.custom_scaler.scale_sequences_with_padding(
            unscaled_data['input_train'], 
            unscaled_data['output_train']
        )
        input_test_scaled, output_test_scaled = self.custom_scaler.scale_sequences_with_padding(
            unscaled_data['input_test'], 
            unscaled_data['output_test']
        )
        
        return {
            'input_train_scaled': input_train_scaled,
            'input_test_scaled': input_test_scaled,
            'output_train_scaled': output_train_scaled,
            'output_test_scaled': output_test_scaled
        }
    
    
    def add_open_to_output(self, input_sequence: np.ndarray, output_sequence: np.ndarray, prediction_length: int) -> np.ndarray:
        """
        Add Open column to output sequence for proper candlestick charting.
        
        LOGIC:
        - Output data only contains High, Low, Close (HLC) values
        - Open values are derived from the previous Close
        - First Open = last Close from input sequence
        - Subsequent Opens = previous Close from output sequence
        
        INPUT:
        - input_sequence: 2D array (timesteps, 4) with OHLC data
        - output_sequence: 1D array (prediction_length * 3) with flattened HLC data
        - prediction_length: number of prediction timesteps
        
        OUTPUT:
        - 2D array (prediction_length, 4) with complete OHLC data
        """
        # Reshape output from flattened to (prediction_length, 3) format (HLC only)
        output_reshaped = output_sequence.reshape(prediction_length, 3)
        
        # Create output with Open column
        output_with_open = np.zeros((prediction_length, 4))
        output_with_open[0, 0] = input_sequence[-1, 3]  # First Open = last input Close
        output_with_open[:, 1] = output_reshaped[:, 0]  # High
        output_with_open[:, 2] = output_reshaped[:, 1]  # Low  
        output_with_open[:, 3] = output_reshaped[:, 2]  # Close
        
        # Set Open for remaining timesteps (each Open = previous Close)
        for i in range(1, prediction_length):
            output_with_open[i, 0] = output_with_open[i-1, 3]  # Open = previous Close
        
        return output_with_open
    
    def combine_input_output_for_chart(self, input_sequence: np.ndarray, output_sequence: np.ndarray) -> np.ndarray:
        """
        Combine input and output sequences for continuous candlestick charting.
        
        LOGIC:
        - Takes input sequence (OHLC) and output sequence (flattened HLC)
        - Adds proper Open values to output sequence
        - Combines them into continuous OHLC data for charting
        
        INPUT:
        - input_sequence: 2D array (sequence_length, 4) with OHLC data
        - output_sequence: 1D array (prediction_length * 3) with flattened HLC data
        
        OUTPUT:
        - 2D array (sequence_length + prediction_length, 4) with continuous OHLC data
        """
        # Add Open column to output sequence
        output_with_open = self.add_open_to_output(input_sequence, output_sequence, self.config.prediction_length)
        
        # Combine input and output
        combined = np.vstack([input_sequence, output_with_open])
        return combined

