"""
Plotting utilities for cryptocurrency prediction visualization
Updated to use improved charting functions from test notebook
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


def draw_candlestick_chart(data, title, ylabel, ylim=None, show_full_range=False):
    """
    Draw candlestick chart with automatic y-axis scaling.
    
    FEATURES:
    - Automatically detects Y-axis range using data min/max values
    - 5% padding for better visualization
    - Green/red candlesticks based on open vs close
    - Proper high-low wicks and open-close bodies
    
    INPUT:
    - data: 2D array (timesteps, 4) with OHLC data
    - title: Chart title
    - ylabel: Y-axis label
    - ylim: Optional fixed y-axis limits
    - show_full_range: If True, show full 0.25-0.75 range (for scaled data)
    """
    fig, ax = plt.subplots(figsize=(15, 6))
    
    for i in range(len(data)):
        open_price = data[i, 0]
        high_price = data[i, 1]
        low_price = data[i, 2]
        close_price = data[i, 3]
        
        # Determine color based on open vs close
        color = 'green' if close_price >= open_price else 'red'
        
        # Draw the high-low line
        ax.plot([i, i], [low_price, high_price], color='black', linewidth=0.8)
        
        # Draw the open-close rectangle
        body_height = abs(close_price - open_price)
        body_bottom = min(open_price, close_price)
        
        if body_height > 0:
            rect = Rectangle((i-0.3, body_bottom), 0.6, body_height, 
                            facecolor=color, edgecolor='black', alpha=0.7)
            ax.add_patch(rect)
        else:
            # Doji - just a line
            ax.plot([i-0.3, i+0.3], [open_price, open_price], color='black', linewidth=2)
    
    # Calculate actual min/max from candlestick data
    ohlc_data = data[:, [0, 1, 2, 3]]  # All OHLC values
    data_min = ohlc_data.min()
    data_max = ohlc_data.max()
    data_range = data_max - data_min
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Index')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    
    # Set y-axis range based on parameters
    if ylim:
        ax.set_ylim(ylim)
        print(f"   📊 Chart Y-axis: {ylim[0]:.6f} to {ylim[1]:.6f} (fixed range)")
    elif show_full_range:
        # Always show full configured range (0.25-0.75)
        ax.set_ylim(0.25, 0.75)
        print(f"   📊 Chart Y-axis: 0.250000 to 0.750000 (full configured range)")
        print(f"   📊 Data range: {data_min:.6f} to {data_max:.6f} (within full range)")
    else:
        # Use dynamic range based on actual data
        padding = data_range * 0.05
        y_min = data_min - padding
        y_max = data_max + padding
        ax.set_ylim(y_min, y_max)
        print(f"   📊 Chart Y-axis: {y_min:.6f} to {y_max:.6f} (data range: {data_min:.6f} to {data_max:.6f})")
    
    plt.tight_layout()
    plt.show()


def plot_combined_input_output_charts(unscaled_data, scaled_data, config, title_prefix="Data Visualization"):
    """
    Plot combined input + output candlestick charts.
    
    INPUT:
    - unscaled_data: 2D array (timesteps, 4) with unscaled OHLC data
    - scaled_data: 2D array (timesteps, 4) with scaled OHLC data
    - config: Configuration object with sequence_length and prediction_length
    - title_prefix: Prefix for chart titles
    """
    print(f"\n📊 {title_prefix.upper()}")
    print("=" * 60)
    
    # Chart 1: Unscaled Data
    print("\n1️⃣ Unscaled Data Candlestick Chart (Input + Output):")
    draw_candlestick_chart(
        unscaled_data, 
        f'{title_prefix} - Unscaled OHLC Values', 
        'Price (USDT)'
        # Uses automatic Y-axis detection by default
    )
    
    # Chart 2: Scaled Data
    print("\n2️⃣ Scaled Data Candlestick Chart (Input + Output):")
    draw_candlestick_chart(
        scaled_data, 
        f'{title_prefix} - Scaled OHLC Values (Padded Scaling)', 
        'Scaled Value'
        # Uses automatic Y-axis detection by default
    )
    
    # Summary Statistics
    print(f"\n📊 CHART SUMMARY STATISTICS:")
    print(f"   Unscaled OHLC range: {unscaled_data[:, [0,1,2,3]].min():.2f} to {unscaled_data[:, [0,1,2,3]].max():.2f}")
    print(f"   Scaled OHLC range: {scaled_data[:, [0,1,2,3]].min():.6f} to {scaled_data[:, [0,1,2,3]].max():.6f}")
    
    # Check High-Low relationships
    unscaled_high_low = unscaled_data[:, 1] - unscaled_data[:, 2]
    scaled_high_low = scaled_data[:, 1] - scaled_data[:, 2]
    
    print(f"\n📊 High-Low Relationship Check:")
    print(f"   Unscaled - All positive: {np.all(unscaled_high_low >= 0)}")
    print(f"   Scaled - All positive: {np.all(scaled_high_low >= 0)}")
    
    # Show input vs output ranges
    input_end = config.sequence_length
    print(f"\n📊 Input vs Output Analysis:")
    print(f"   Input timesteps (0-{input_end-1}):")
    print(f"     Unscaled range: {unscaled_data[:input_end, [0,1,2,3]].min():.2f} to {unscaled_data[:input_end, [0,1,2,3]].max():.2f}")
    print(f"     Scaled range: {scaled_data[:input_end, [0,1,2,3]].min():.6f} to {scaled_data[:input_end, [0,1,2,3]].max():.6f}")
    print(f"   Output timesteps ({input_end}-{len(unscaled_data)-1}):")
    print(f"     Unscaled range: {unscaled_data[input_end:, [0,1,2,3]].min():.2f} to {unscaled_data[input_end:, [0,1,2,3]].max():.2f}")
    print(f"     Scaled range: {scaled_data[input_end:, [0,1,2,3]].min():.6f} to {scaled_data[input_end:, [0,1,2,3]].max():.6f}")
    
    print(f"\n✅ All candlestick charts displayed successfully!")
    print(f"   Charts show combined Input + Output data for continuous visualization")
    print(f"   Output scaling uses expanded range approach for 0-1 consistency")


def plot_sample_data_comparison(unscaled_data, scaled_data, config, num_samples=10):
    """
    Print sample data comparison for combined input + output.
    
    INPUT:
    - unscaled_data: 2D array (timesteps, 4) with unscaled OHLC data
    - scaled_data: 2D array (timesteps, 4) with scaled OHLC data
    - config: Configuration object with sequence_length
    - num_samples: Number of sample rows to display
    """
    print(f"\n📊 SAMPLE DATA COMPARISON - COMBINED INPUT + OUTPUT")
    print("=" * 70)
    
    # Print unscaled data
    print("🔍 UNSCALED DATA (first 10 rows - Input + Output):")
    print("Index    Open      High      Low       Close")
    print("-" * 45)
    for i in range(min(num_samples, len(unscaled_data))):
        row = unscaled_data[i]
        print(f"{i:5d}  {row[0]:8.2f}  {row[1]:8.2f}  {row[2]:8.2f}  {row[3]:8.2f}")
    
    # Print scaled data
    print("\n🔍 SCALED DATA (first 10 rows - Input + Output):")
    print("Index    Open      High      Low       Close")
    print("-" * 45)
    for i in range(min(num_samples, len(scaled_data))):
        row = scaled_data[i]
        print(f"{i:5d}  {row[0]:8.6f}  {row[1]:8.6f}  {row[2]:8.6f}  {row[3]:8.6f}")
    
    # Show transition from input to output
    input_end = config.sequence_length
    print(f"\n🔄 INPUT TO OUTPUT TRANSITION:")
    print(f"   Input ends at timestep {input_end-1}, Output starts at timestep {input_end}")
    print(f"   Last input Close: {unscaled_data[input_end-1, 3]:.2f}")
    print(f"   First output Open: {unscaled_data[input_end, 0]:.2f} (should match)")
    
    print("\n📈 DATA RANGES:")
    print(f"   Unscaled OHLC: {unscaled_data[:, [0,1,2,3]].min():.2f} to {unscaled_data[:, [0,1,2,3]].max():.2f}")
    print(f"   Scaled OHLC:   {scaled_data[:, [0,1,2,3]].min():.6f} to {scaled_data[:, [0,1,2,3]].max():.6f}")
    
    print(f"\n✅ Combined data shows continuous Input + Output visualization")
    print(f"   Output scaling preserves relationship with input data")


def plot_training_history(history):
    """Plot training and validation loss/accuracy over epochs"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss', color='blue')
    ax1.plot(history.history['val_loss'], label='Validation Loss', color='red')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot MAE
    ax2.plot(history.history['mae'], label='Training MAE', color='blue')
    ax2.plot(history.history['val_mae'], label='Validation MAE', color='red')
    ax2.set_title('Model MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_predictions_vs_actual(predictions, actual, target_columns, max_samples=1000):
    """Plot predictions vs actual values for each target column"""
    n_targets = len(target_columns)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Limit samples for better visualization
    n_samples = min(max_samples, len(predictions))
    indices = np.linspace(0, len(predictions)-1, n_samples, dtype=int)
    
    for i, col in enumerate(target_columns):
        if i < len(axes):
            ax = axes[i]
            ax.scatter(actual[indices, i], predictions[indices, i], alpha=0.6)
            ax.plot([actual[:, i].min(), actual[:, i].max()], 
                   [actual[:, i].min(), actual[:, i].max()], 'r--', lw=2)
            ax.set_xlabel(f'Actual {col}')
            ax.set_ylabel(f'Predicted {col}')
            ax.set_title(f'{col} - Predictions vs Actual')
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(target_columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_prediction_accuracy(predictions, actual, timesteps_ahead=1):
    """Plot prediction accuracy over time"""
    if predictions.ndim == 3:
        predictions = predictions.reshape(-1, predictions.shape[-1])
    if actual.ndim == 3:
        actual = actual.reshape(-1, actual.shape[-1])
    
    mae_per_sample = np.mean(np.abs(actual - predictions), axis=1)
    
    plt.figure(figsize=(15, 6))
    plt.plot(mae_per_sample, alpha=0.7)
    plt.title(f'Prediction Accuracy Over Time (MAE per sample)')
    plt.xlabel('Sample Index')
    plt.ylabel('Mean Absolute Error')
    plt.grid(True, alpha=0.3)
    plt.show()


def create_interactive_candlestick_chart(data, title="Candlestick Chart"):
    """
    Create interactive candlestick chart using Plotly.
    
    INPUT:
    - data: 2D array (timesteps, 4) with OHLC data
    - title: Chart title
    """
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(data, columns=['Open', 'High', 'Low', 'Close'])
    df['Time'] = range(len(df))
    
    fig = go.Figure(data=go.Candlestick(
        x=df['Time'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="OHLC"
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time Index',
        yaxis_title='Price',
        template='plotly_white',
        height=600
    )
    
    fig.show()


def plot_feature_importance(feature_names, importance_scores, title="Feature Importance"):
    """Plot feature importance scores"""
    plt.figure(figsize=(12, 8))
    bars = plt.bar(feature_names, importance_scores)
    plt.title(title)
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def plot_residuals(predictions, actual, title="Residuals Plot"):
    """Plot residuals (prediction errors)"""
    residuals = actual - predictions
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Residuals over time
    ax1.plot(residuals, alpha=0.7)
    ax1.set_title('Residuals Over Time')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Residual')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='r', linestyle='--')
    
    # Residuals histogram
    ax2.hist(residuals.flatten(), bins=50, alpha=0.7, edgecolor='black')
    ax2.set_title('Residuals Distribution')
    ax2.set_xlabel('Residual Value')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()