"""
Plotting utilities for cryptocurrency prediction visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


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
            
            # Plot actual vs predicted
            ax.scatter(actual[indices, i], predictions[indices, i], alpha=0.6, s=20)
            
            # Perfect prediction line
            min_val = min(actual[:, i].min(), predictions[:, i].min())
            max_val = max(actual[:, i].max(), predictions[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            ax.set_xlabel(f'Actual {col}')
            ax.set_ylabel(f'Predicted {col}')
            ax.set_title(f'{col}: Predictions vs Actual')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(target_columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_price_data(data, title="Price Data", max_points=2000):
    """Plot original price data (OHLC)"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Limit data points for better visualization
    n_points = min(max_points, len(data))
    data_subset = data.iloc[-n_points:]
    
    # Plot OHLC
    ax1.plot(data_subset.index, data_subset['Open'], label='Open', alpha=0.7)
    ax1.plot(data_subset.index, data_subset['High'], label='High', alpha=0.7)
    ax1.plot(data_subset.index, data_subset['Low'], label='Low', alpha=0.7)
    ax1.plot(data_subset.index, data_subset['Close'], label='Close', linewidth=2)
    ax1.set_title(f'{title} - OHLC Prices')
    ax1.set_ylabel('Price (USDT)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot Volume
    ax2.bar(data_subset.index, data_subset['Volume'], alpha=0.7, color='orange')
    ax2.set_title(f'{title} - Volume')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Volume')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_prediction_errors(predictions, actual, target_columns, max_samples=1000):
    """Plot prediction errors for each target column"""
    n_targets = len(target_columns)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Limit samples for better visualization
    n_samples = min(max_samples, len(predictions))
    indices = np.linspace(0, len(predictions)-1, n_samples, dtype=int)
    
    for i, col in enumerate(target_columns):
        if i < len(axes):
            ax = axes[i]
            
            # Calculate errors
            errors = predictions[indices, i] - actual[indices, i]
            
            # Plot error distribution
            ax.hist(errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
            ax.axvline(errors.mean(), color='orange', linestyle='-', linewidth=2, label=f'Mean Error: {errors.mean():.2f}')
            
            ax.set_xlabel(f'Prediction Error ({col})')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{col}: Error Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(target_columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_feature_importance(feature_names, feature_importance=None):
    """Plot feature importance if available"""
    if feature_importance is None:
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort features by importance
    sorted_indices = np.argsort(feature_importance)[::-1]
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_importance = feature_importance[sorted_indices]
    
    # Plot horizontal bar chart
    bars = ax.barh(range(len(sorted_features)), sorted_importance, color='skyblue', edgecolor='navy')
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel('Feature Importance')
    ax.set_title('Feature Importance Analysis')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, importance) in enumerate(zip(bars, sorted_importance)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{importance:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    plt.show()


def plot_feature_analysis(feature_info):
    """Plot feature analysis including groups and feature names"""
    feature_names = feature_info['feature_columns']
    groups = feature_info['feature_groups']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Feature groups count
    group_names = list(groups.keys())
    group_counts = [len(features) for features in groups.values()]
    
    bars1 = ax1.bar(group_names, group_counts, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax1.set_title('Feature Groups Distribution')
    ax1.set_ylabel('Number of Features')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars1, group_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(count), ha='center', va='bottom')
    
    # Feature names (top 10)
    top_features = feature_names[:10]
    y_pos = np.arange(len(top_features))
    
    bars2 = ax2.barh(y_pos, [1]*len(top_features), color='lightblue')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_features)
    ax2.set_xlabel('Features')
    ax2.set_title('Top 10 Features Used')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_candlestick_chart(ohlcv_data, title="Candlestick Chart", max_candles=100):
    """Plot candlestick chart for OHLCV data using Plotly"""
    # Limit data for better visualization
    n_candles = min(max_candles, len(ohlcv_data))
    data = ohlcv_data.iloc[-n_candles:] if hasattr(ohlcv_data, 'iloc') else ohlcv_data[-n_candles:]
    
    # Extract OHLC data
    if hasattr(data, 'values'):
        ohlc = data[['Open', 'High', 'Low', 'Close']].values
        volume = data['Volume'].values
    else:
        ohlc = data[:, :4]  # First 4 columns are OHLC
        volume = data[:, 4] if data.shape[1] > 4 else np.zeros(len(data))
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f'{title} - OHLC Candlesticks', 'Volume'),
        row_heights=[0.7, 0.3]
    )
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=list(range(len(ohlc))),
        open=ohlc[:, 0],
        high=ohlc[:, 1],
        low=ohlc[:, 2],
        close=ohlc[:, 3],
        name="Price",
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ), row=1, col=1)
    
    # Add volume chart
    fig.add_trace(go.Bar(
        x=list(range(len(volume))),
        y=volume,
        name="Volume",
        marker_color='orange',
        opacity=0.7
    ), row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title=f'{title} - OHLCV Chart',
        xaxis_title="Time Steps",
        yaxis_title="Price (USDT)",
        height=600,
        showlegend=True,
        template="plotly_dark"
    )
    
    # Update x-axis for volume
    fig.update_xaxes(title_text="Time Steps", row=2, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    fig.show()


def plot_prediction_accuracy_distribution(prediction_errors, title="Prediction Accuracy Distribution"):
    """Plot distribution of prediction errors"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Error distribution
    ax1.hist(prediction_errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(np.mean(prediction_errors), color='red', linestyle='--', 
                label=f'Mean: {np.mean(prediction_errors):.2f}')
    ax1.axvline(np.median(prediction_errors), color='orange', linestyle='--', 
                label=f'Median: {np.median(prediction_errors):.2f}')
    ax1.set_xlabel('RMSE Error')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Error Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Error over time (sample)
    sample_size = min(1000, len(prediction_errors))
    sample_errors = prediction_errors[:sample_size]
    ax2.plot(sample_errors, alpha=0.7, color='purple')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('RMSE Error')
    ax2.set_title('Error Over Time (Sample)')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_prediction_candlestick(pred_data, actual_data, input_data, title, rmse_error):
    """Create interactive candlestick chart for input, prediction vs actual data using Plotly"""
    
    # Reshape data for candlestick plotting
    # Input data: 5 features (OHLCV)
    input_ohlc = input_data.reshape(-1, 5)[:, :4]  # OHLC only
    input_volume = input_data.reshape(-1, 5)[:, 4]  # Volume
    
    # Predicted/Actual data: 4 features (HLCV) - Open is derived from previous Close
    pred_hlcv = pred_data.reshape(-1, 4)  # HLCV
    actual_hlcv = actual_data.reshape(-1, 4)  # HLCV
    
    # Reconstruct Open prices for predicted and actual data
    # Open = previous Close (from input data's last Close, then subsequent Closes)
    pred_ohlc = np.zeros((len(pred_hlcv), 4))
    actual_ohlc = np.zeros((len(actual_hlcv), 4))
    
    # First Open = last Close from input data
    last_input_close = input_ohlc[-1, 3]  # Last Close from input
    pred_ohlc[0, 0] = last_input_close  # First predicted Open
    actual_ohlc[0, 0] = last_input_close  # First actual Open
    
    # Subsequent Opens = previous Close
    for i in range(1, len(pred_hlcv)):
        pred_ohlc[i, 0] = pred_hlcv[i-1, 2]  # Open = previous Close
        actual_ohlc[i, 0] = actual_hlcv[i-1, 2]  # Open = previous Close
    
    # Fill in HLCV data
    pred_ohlc[:, 1:] = pred_hlcv[:, :3]  # High, Low, Close (first 3 columns)
    actual_ohlc[:, 1:] = actual_hlcv[:, :3]  # High, Low, Close (first 3 columns)
    
    pred_volume = pred_hlcv[:, 3]  # Volume
    actual_volume = actual_hlcv[:, 3]  # Volume
    
    # Calculate positions for plotting
    input_len = len(input_ohlc)
    pred_len = len(pred_ohlc)
    total_len = input_len + pred_len
    
    # Create time indices
    input_times = list(range(input_len))
    pred_times = list(range(input_len, total_len))
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            f'{title} - OHLC Candlesticks (RMSE: {rmse_error:.2f}) - Scaled Data',
            'Volume Comparison - Scaled Data'
        ),
        row_heights=[0.7, 0.3]
    )
    
    # Add input candlesticks
    fig.add_trace(go.Candlestick(
        x=input_times,
        open=input_ohlc[:, 0],
        high=input_ohlc[:, 1],
        low=input_ohlc[:, 2],
        close=input_ohlc[:, 3],
        name="Input Data",
        increasing_line_color='#4A90E2',
        decreasing_line_color='#2E5BBA',
        increasing_fillcolor='#4A90E2',
        decreasing_fillcolor='#2E5BBA'
    ), row=1, col=1)
    
    # Add predicted candlesticks
    fig.add_trace(go.Candlestick(
        x=pred_times,
        open=pred_ohlc[:, 0],
        high=pred_ohlc[:, 1],
        low=pred_ohlc[:, 2],
        close=pred_ohlc[:, 3],
        name="Predicted",
        increasing_line_color='#50C878',
        decreasing_line_color='#3A9B5A',
        increasing_fillcolor='#50C878',
        decreasing_fillcolor='#3A9B5A'
    ), row=1, col=1)
    
    # Add actual candlesticks
    fig.add_trace(go.Candlestick(
        x=pred_times,
        open=actual_ohlc[:, 0],
        high=actual_ohlc[:, 1],
        low=actual_ohlc[:, 2],
        close=actual_ohlc[:, 3],
        name="Actual",
        increasing_line_color='#FF6B6B',
        decreasing_line_color='#CC5555',
        increasing_fillcolor='#FF6B6B',
        decreasing_fillcolor='#CC5555'
    ), row=1, col=1)
    
    # Add vertical line to separate input from prediction
    fig.add_vline(
        x=input_len - 0.5, 
        line_dash="dash", 
        line_color="red", 
        line_width=2,
        annotation_text="Prediction Start",
        annotation_position="top"
    )
    
    # Add volume bars
    fig.add_trace(go.Bar(
        x=input_times,
        y=input_volume,
        name="Input Volume",
        marker_color='#4A90E2',
        opacity=0.7
    ), row=2, col=1)
    
    fig.add_trace(go.Bar(
        x=pred_times,
        y=pred_volume,
        name="Predicted Volume",
        marker_color='#50C878',
        opacity=0.7
    ), row=2, col=1)
    
    fig.add_trace(go.Bar(
        x=pred_times,
        y=actual_volume,
        name="Actual Volume",
        marker_color='#FF6B6B',
        opacity=0.7
    ), row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title=f'{title} - Interactive Candlestick Chart',
        xaxis_title="Time Steps (Input → Prediction)",
        yaxis_title="Scaled Price (0-1)",
        height=700,
        showlegend=True,
        template="plotly_dark",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update x-axis for volume
    fig.update_xaxes(title_text="Time Steps (Input → Prediction)", row=2, col=1)
    fig.update_yaxes(title_text="Scaled Volume (0-1)", row=2, col=1)
    
    # Update y-axis for price
    fig.update_yaxes(title_text="Scaled Price (0-1)", row=1, col=1)
    
    fig.show()


def plot_test_performance_metrics(evaluation_results, title="Test Performance Metrics"):
    """Plot test performance metrics and trade-off analysis"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Performance metrics bar chart
    metrics = ['MAE', 'MAPE', 'RMSE']
    values = [evaluation_results['test_mae'], evaluation_results['test_mape'], evaluation_results['rmse']]
    bars = ax1.bar(metrics, values, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax1.set_title('Test Performance Metrics')
    ax1.set_ylabel('Value')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01, 
                f'{value:.2f}', ha='center', va='bottom')
    
    # Model complexity vs performance scatter
    ax2.scatter([evaluation_results['test_mae']], [evaluation_results['test_mape']], 
                s=200, color='red', alpha=0.7)
    ax2.set_xlabel('MAE')
    ax2.set_ylabel('MAPE (%)')
    ax2.set_title('Performance Trade-off')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_model_performance_summary(history, title="Model Performance Summary"):
    """Plot training history - loss and MAE over epochs"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Training history - Loss
    ax1.plot(history.history['loss'], label='Training Loss', color='blue')
    ax1.plot(history.history['val_loss'], label='Validation Loss', color='red')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Training history - MAE
    ax2.plot(history.history['mae'], label='Training MAE', color='blue')
    ax2.plot(history.history['val_mae'], label='Validation MAE', color='red')
    ax2.set_title('Model MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()