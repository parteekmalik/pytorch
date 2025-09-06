import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import MinMaxScaler
import warnings

from src.config import BaseConfig
from .utils import download_binance_data, create_sequences

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning)

class GroupedScaler:
    """
    Scaler that groups OHLC features together for unified scaling.
    
    LOGIC:
    - OHLC features (Open, High, Low, Close) are treated as a single group
    - All OHLC values are combined into one flat array and scaled together
    - This preserves relative relationships between Open, High, Low, Close within each sequence
    - Each sequence is scaled independently (called from _scale_sequences)
    
    WHY THIS APPROACH:
    - Prevents High-Low relationship distortion that occurs when scaling OHLC separately
    - Maintains candlestick chart integrity after scaling
    - Each sequence gets its own min/max range for optimal scaling
    
    INPUT FORMAT:
    - Expects 2D array with shape (timesteps, 4) where columns are [Open, High, Low, Close]
    - Each call to transform() processes one sequence independently
    
    OUTPUT FORMAT:
    - Returns 2D array with same shape as input
    - All values scaled to 0-1 range
    - OHLC values scaled together as a unified group
    """
    
    def _is_fit(self, input: np.ndarray) -> None:
        """Validate input format before processing."""
        if input.ndim != 2:
            raise ValueError(f"Expected 2D array, got {input.ndim}D array")
        
        if input.shape[1] != 4:
            raise ValueError(f"Expected 4 features (OHLC), got {input.shape[1]} features")
        
        if input.size == 0:
            raise ValueError("Cannot fit on empty array")
        
        if np.any(np.isnan(input)) or np.any(np.isinf(input)):
            raise ValueError("Array contains NaN or infinite values")
    
    def transform(self, input: np.ndarray) -> np.ndarray:
        """
        Scale a single sequence using grouped OHLC scaling.
        
        PROCESS:
        1. Combine all OHLC values into one flat array for unified scaling
        2. Scale OHLC combined array using single MinMaxScaler
        3. Reshape OHLC back to original (timesteps, 4) shape
        
        This ensures all OHLC values in the sequence use the same min/max range,
        preserving their relative relationships.
        """
        self._is_fit(input)
        
        input_scaled = input.copy()
        
        # Step 1: Combine all OHLC values into one flat array for unified scaling
        ohlc_combined = input.flatten().reshape(-1, 1)
        
        # Step 2: Scale OHLC with single scaler
        ohlc_scaler = MinMaxScaler()
        ohlc_scaled = ohlc_scaler.fit_transform(ohlc_combined)
        
        # Step 3: Reshape OHLC back to original (timesteps, 4) shape
        ohlc_scaled_reshaped = ohlc_scaled.reshape(input.shape[0], 4)
        
        # Step 4: Return scaled OHLC data
        input_scaled = ohlc_scaled_reshaped
        
        return input_scaled


class BinanceDataOrganizer:
    def __init__(self, config: BaseConfig):
        self.config = config
        self.custom_scaler = GroupedScaler()
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

        # Scale input data and get scaling parameters
        input_train_scaled, input_scaling_params = self._scale_sequences_with_params(unscaled_data['input_train'])
        input_test_scaled, _ = self._scale_sequences_with_params(unscaled_data['input_test'])
        
        # Scale output data using input scaling parameters
        output_train_scaled = self._scale_output_with_input_params(unscaled_data['output_train'], input_scaling_params)
        output_test_scaled = self._scale_output_with_input_params(unscaled_data['output_test'], input_scaling_params)
        
        return {
            'input_train_scaled': input_train_scaled,
            'input_test_scaled': input_test_scaled,
            'output_train_scaled': output_train_scaled,
            'output_test_scaled': output_test_scaled
        }
    
    def _scale_sequences(self, sequences: np.ndarray) -> np.ndarray:
        """
        Scale each sequence independently using GroupedScaler.
        
        WHY INDEPENDENT SCALING:
        - Each sequence gets its own min/max range for optimal scaling
        - Prevents one sequence with extreme values from affecting others
        - Maintains relative relationships within each sequence
        - Essential for time series prediction where each sequence is independent
        
        PROCESS:
        1. Iterate through each sequence in the batch
        2. Create new GroupedScaler for each sequence
        3. Scale each sequence independently
        4. Combine all scaled sequences back into 3D array
        
        INPUT: 3D array (num_sequences, sequence_length, features)
        OUTPUT: 3D array (num_sequences, sequence_length, features) - same shape, scaled values
        """
        scaled_sequences = []
        
        for i in range(len(sequences)):
            # Get individual sequence (shape: [sequence_length, features])
            sequence = sequences[i]
            
            # Scale this sequence independently using GroupedScaler
            scaler = GroupedScaler()
            scaled_sequence = scaler.transform(sequence)
            scaled_sequences.append(scaled_sequence)
        
        return np.array(scaled_sequences)
    
    def _scale_sequences_with_params(self, sequences: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Scale input sequences and return scaling parameters for output scaling.
        
        LOGIC:
        - Scale each input sequence independently
        - Store the min/max values used for each sequence
        - Return both scaled sequences and scaling parameters
        
        OUTPUT:
        - scaled_sequences: 3D array of scaled input data
        - scaling_params: List of dicts with min/max values for each sequence
        """
        scaled_sequences = []
        scaling_params = []
        
        for i in range(len(sequences)):
            sequence = sequences[i]
            
            # Get min/max values from input sequence (OHLC only)
            ohlc_data = sequence[:, [0, 1, 2, 3]]
            
            # Calculate min/max for OHLC (combined)
            ohlc_min = ohlc_data.min()
            ohlc_max = ohlc_data.max()
            
            # Store scaling parameters
            params = {
                'ohlc_min': ohlc_min,
                'ohlc_max': ohlc_max
            }
            scaling_params.append(params)
            
            # Scale the sequence
            scaler = GroupedScaler()
            scaled_sequence = scaler.transform(sequence)
            scaled_sequences.append(scaled_sequence)
        
        return np.array(scaled_sequences), scaling_params
    
    def _scale_output_with_input_params(self, output_data: np.ndarray, input_scaling_params: List[Dict]) -> np.ndarray:
        """
        Scale output data using input scaling parameters.
        
        LOGIC:
        - Use the same min/max values from input data to scale output
        - Output can go beyond 0-1 range if output values exceed input range
        - This maintains the relationship between input and output scaling
        
        PROCESS:
        1. For each output sequence, get corresponding input scaling params
        2. Apply same OHLC min/max to output OHLC values
        3. This allows output to potentially exceed 0-1 range
        """
        scaled_output = output_data.copy()
        
        for i in range(len(output_data)):
            output_sequence = output_data[i]
            params = input_scaling_params[i]
            
            # Reshape output sequence to (prediction_length, 3) for HLC
            # Output is flattened: [H, L, C, H, L, C, ...] for prediction_length timesteps
            prediction_length = len(output_sequence) // 3
            output_reshaped = output_sequence.reshape(prediction_length, 3)
            
            # Get HLC from reshaped output
            ohlc_output = output_reshaped  # H, L, C (no Open in output)
            
            # Scale using input min/max values
            ohlc_scaled = (ohlc_output - params['ohlc_min']) / (params['ohlc_max'] - params['ohlc_min'])
            
            # Flatten scaled OHLC back to original format
            scaled_sequence = ohlc_scaled.flatten()
            scaled_output[i] = scaled_sequence
        
        return scaled_output
    
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

