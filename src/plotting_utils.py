"""
Plotting utilities for cryptocurrency prediction visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    """Plot candlestick chart for OHLCV data"""
    import matplotlib.patches as patches
    
    # Limit data for better visualization
    n_candles = min(max_candles, len(ohlcv_data))
    data = ohlcv_data.iloc[-n_candles:] if hasattr(ohlcv_data, 'iloc') else ohlcv_data[-n_candles:]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Extract OHLC data
    if hasattr(data, 'values'):
        ohlc = data[['Open', 'High', 'Low', 'Close']].values
        volume = data['Volume'].values
    else:
        ohlc = data[:, :4]  # First 4 columns are OHLC
        volume = data[:, 4] if data.shape[1] > 4 else np.zeros(len(data))
    
    # Plot candlesticks
    for i, (open_price, high, low, close) in enumerate(ohlc):
        color = 'green' if close >= open_price else 'red'
        
        # Draw the high-low line
        ax1.plot([i, i], [low, high], color='black', linewidth=0.8)
        
        # Draw the open-close rectangle
        height = abs(close - open_price)
        bottom = min(open_price, close)
        rect = patches.Rectangle((i-0.4, bottom), 0.8, height, 
                               facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.7)
        ax1.add_patch(rect)
    
    ax1.set_title(f'{title} - OHLC Candlesticks')
    ax1.set_ylabel('Price (USDT)')
    ax1.grid(True, alpha=0.3)
    
    # Plot volume
    ax2.bar(range(len(volume)), volume, color='orange', alpha=0.7)
    ax2.set_title('Volume')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Volume')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_prediction_comparison(y_pred, y_true, best_idx, worst_idx, title="Best vs Worst Predictions"):
    """Plot comparison between best and worst predictions"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Handle multi-step predictions - take only the first timestep
    pred_best = y_pred[best_idx]
    true_best = y_true[best_idx]
    pred_worst = y_pred[worst_idx]
    true_worst = y_true[worst_idx]
    
    # If predictions are multi-step, reshape and take first timestep
    if pred_best.shape[0] > 5:  # Multi-step prediction
        n_steps = pred_best.shape[0] // 5
        pred_best = pred_best[:5]  # First timestep
        true_best = true_best[:5]  # First timestep
        pred_worst = pred_worst[:5]  # First timestep
        true_worst = true_worst[:5]  # First timestep
    
    # Best prediction
    ax1.bar(['Open', 'High', 'Low', 'Close', 'Volume'], pred_best, 
            alpha=0.7, color='green', label='Predicted')
    ax1.bar(['Open', 'High', 'Low', 'Close', 'Volume'], true_best, 
            alpha=0.5, color='blue', label='Actual')
    ax1.set_title(f'Best Prediction (Index {best_idx}) - First Timestep')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Worst prediction
    ax2.bar(['Open', 'High', 'Low', 'Close', 'Volume'], pred_worst, 
            alpha=0.7, color='red', label='Predicted')
    ax2.bar(['Open', 'High', 'Low', 'Close', 'Volume'], true_worst, 
            alpha=0.5, color='blue', label='Actual')
    ax2.set_title(f'Worst Prediction (Index {worst_idx}) - First Timestep')
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


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


def plot_model_performance_summary(history, evaluation_results, title="Model Performance Summary"):
    """Plot comprehensive model performance summary"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
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
    
    # Performance metrics
    metrics = ['MAE', 'MAPE', 'RMSE']
    values = [evaluation_results['test_mae'], evaluation_results['test_mape'], evaluation_results['rmse']]
    bars = ax3.bar(metrics, values, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax3.set_title('Test Performance Metrics')
    ax3.set_ylabel('Value')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01, 
                f'{value:.2f}', ha='center', va='bottom')
    
    # Model complexity vs performance
    ax4.scatter([evaluation_results['test_mae']], [evaluation_results['test_mape']], 
                s=200, color='red', alpha=0.7)
    ax4.set_xlabel('MAE')
    ax4.set_ylabel('MAPE (%)')
    ax4.set_title('Performance Trade-off')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()