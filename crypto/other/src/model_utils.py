import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from typing import Tuple, Dict, Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_lstm_model(input_shape: Tuple[int, int], lstm_units: int = 50, dropout_rate: float = 0.2, learning_rate: float = 0.001, prediction_length: int = 1) -> Sequential:
    """
    Create LSTM model for cryptocurrency price prediction.
    
    UPDATED FOR NEW APPROACH:
    - Input: OHLC data (4 features per timestep)
    - Output: HLC data (3 features per timestep) - Open is derived from previous Close
    - Uses expanded range scaling approach
    - Constrains output to 0-1 range with sigmoid activation
    
    Args:
        input_shape: Input shape (sequence_length, features)
        lstm_units: Number of LSTM units
        dropout_rate: Dropout rate
        learning_rate: Learning rate
        prediction_length: Number of timesteps to predict
    """
    # Calculate output size: prediction_length * 3 (HLC values - Open is derived from previous Close)
    output_size = prediction_length * 3
    
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(lstm_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(25, activation='relu'),
        Dense(output_size, activation='sigmoid')  # Constrain output to 0-1 range
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def visualize_model_architecture(model, save_path: str = "model_architecture.png"):
    """
    Visualize the neural network architecture with layer details and connections.
    
    Args:
        model: Trained or untrained Keras model
        save_path: Path to save the visualization image
    """
    try:
        from tensorflow.keras.utils import plot_model
        
        # Create detailed model visualization
        plot_model(
            model,
            to_file=save_path,
            show_shapes=True,           # Show input/output shapes
            show_layer_names=True,      # Show layer names
            rankdir='TB',              # Top to bottom layout
            expand_nested=True,        # Show nested layers
            dpi=300,                   # High resolution
            layer_range=None,          # Show all layers
            show_layer_activations=True # Show activation functions
        )
        
        print(f"✅ Model architecture saved to: {save_path}")
        print(f"📊 Model Summary:")
        print(f"   Total Layers: {len(model.layers)}")
        print(f"   Total Parameters: {model.count_params():,}")
        print(f"   Trainable Parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
        
        # Print layer details
        print(f"\n📋 Layer Details:")
        for i, layer in enumerate(model.layers):
            print(f"   {i+1:2d}. {layer.name:20s} | {str(layer.output_shape):20s} | {layer.count_params():8,} params")
            
    except ImportError:
        print("❌ Graphviz not installed. Install with: pip install graphviz")
        print("📊 Showing text-based model summary instead:")
        model.summary()
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        print("📊 Showing text-based model summary instead:")
        model.summary()

def plot_training_history(history, save_path: str = None):
    """
    Plot training and validation metrics over epochs.
    
    BEST PRACTICES:
    - Clear, uncluttered design
    - Consistent color scheme
    - Multiple metrics in subplots
    - Interactive elements for exploration
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Training History', fontsize=16, fontweight='bold')
    
    # Loss plot
    axes[0, 0].plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
    if 'val_loss' in history.history:
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', color='red', linewidth=2)
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE plot
    axes[0, 1].plot(history.history['mae'], label='Training MAE', color='green', linewidth=2)
    if 'val_mae' in history.history:
        axes[0, 1].plot(history.history['val_mae'], label='Validation MAE', color='orange', linewidth=2)
    axes[0, 1].set_title('Mean Absolute Error')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate plot (if available)
    if 'lr' in history.history:
        axes[1, 0].plot(history.history['lr'], color='purple', linewidth=2)
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Learning Rate\nNot Available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Learning Rate Schedule')
    
    # Performance summary
    final_loss = history.history['loss'][-1]
    final_mae = history.history['mae'][-1]
    if 'val_loss' in history.history:
        final_val_loss = history.history['val_loss'][-1]
        final_val_mae = history.history['val_mae'][-1]
        
        summary_text = f"""Final Performance:
Training Loss: {final_loss:.6f}
Validation Loss: {final_val_loss:.6f}
Training MAE: {final_mae:.6f}
Validation MAE: {final_val_mae:.6f}
Overfitting: {'Yes' if final_val_loss > final_loss * 1.1 else 'No'}"""
    else:
        summary_text = f"""Final Performance:
Training Loss: {final_loss:.6f}
Training MAE: {final_mae:.6f}"""
    
    axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes, 
                   fontsize=12, verticalalignment='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    axes[1, 1].set_title('Performance Summary')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training history saved to: {save_path}")
    plt.show()

def plot_prediction_analysis(y_true, y_pred, feature_names=['High', 'Low', 'Close'], save_path: str = None):
    """
    Comprehensive prediction analysis visualization.
    
    BEST PRACTICES:
    - Residual plots for error analysis
    - Prediction vs actual scatter plots
    - Error distribution analysis
    - Feature-wise performance breakdown
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Prediction Analysis', fontsize=16, fontweight='bold')
    
    # Reshape data for analysis
    prediction_length = y_true.shape[1] // 3
    y_true_reshaped = y_true.reshape(-1, prediction_length, 3)
    y_pred_reshaped = y_pred.reshape(-1, prediction_length, 3)
    
    # 1. Prediction vs Actual (Overall)
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    axes[0, 0].scatter(y_true_flat, y_pred_flat, alpha=0.6, s=20)
    axes[0, 0].plot([y_true_flat.min(), y_true_flat.max()], 
                    [y_true_flat.min(), y_true_flat.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title('Prediction vs Actual (All Features)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Calculate R²
    r2 = 1 - np.sum((y_true_flat - y_pred_flat)**2) / np.sum((y_true_flat - np.mean(y_true_flat))**2)
    axes[0, 0].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[0, 0].transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 2. Residuals plot
    residuals = y_true_flat - y_pred_flat
    axes[0, 1].scatter(y_pred_flat, residuals, alpha=0.6, s=20)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Error distribution
    axes[0, 2].hist(residuals, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 2].axvline(x=0, color='r', linestyle='--')
    axes[0, 2].set_xlabel('Residuals')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Error Distribution')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4-6. Feature-wise analysis
    colors = ['red', 'green', 'blue']
    for i, (feature, color) in enumerate(zip(feature_names, colors)):
        row = 1
        col = i
        
        true_feature = y_true_reshaped[:, :, i].flatten()
        pred_feature = y_pred_reshaped[:, :, i].flatten()
        
        axes[row, col].scatter(true_feature, pred_feature, alpha=0.6, s=20, color=color)
        axes[row, col].plot([true_feature.min(), true_feature.max()], 
                           [true_feature.min(), true_feature.max()], 'k--', lw=2)
        axes[row, col].set_xlabel(f'Actual {feature}')
        axes[row, col].set_ylabel(f'Predicted {feature}')
        axes[row, col].set_title(f'{feature} Prediction Accuracy')
        axes[row, col].grid(True, alpha=0.3)
        
        # Calculate metrics for this feature
        mae = mean_absolute_error(true_feature, pred_feature)
        mse = mean_squared_error(true_feature, pred_feature)
        axes[row, col].text(0.05, 0.95, f'MAE: {mae:.4f}\nMSE: {mse:.4f}', 
                           transform=axes[row, col].transAxes,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Prediction analysis saved to: {save_path}")
    plt.show()

def create_interactive_prediction_chart(y_true, y_pred, timesteps=None, save_path: str = None):
    """
    Create interactive prediction visualization using Plotly.
    
    BEST PRACTICES:
    - Interactive elements for exploration
    - Multiple data series
    - Zoom and pan capabilities
    - Hover information
    """
    if timesteps is None:
        timesteps = list(range(len(y_true)))
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=['High', 'Low', 'Close'],
        vertical_spacing=0.1
    )
    
    feature_names = ['High', 'Low', 'Close']
    colors = ['red', 'green', 'blue']
    
    for i, (feature, color) in enumerate(zip(feature_names, colors)):
        # Reshape data for this feature
        prediction_length = y_true.shape[1] // 3
        true_feature = y_true.reshape(-1, prediction_length, 3)[:, :, i].flatten()
        pred_feature = y_pred.reshape(-1, prediction_length, 3)[:, :, i].flatten()
        
        # Create timesteps for this feature
        feature_timesteps = []
        for t in timesteps:
            feature_timesteps.extend([f"{t}_{j}" for j in range(prediction_length)])
        
        # Add traces
        fig.add_trace(
            go.Scatter(
                x=feature_timesteps,
                y=true_feature,
                mode='lines+markers',
                name=f'Actual {feature}',
                line=dict(color=color, width=2),
                marker=dict(size=4)
            ),
            row=i+1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=feature_timesteps,
                y=pred_feature,
                mode='lines+markers',
                name=f'Predicted {feature}',
                line=dict(color=color, width=2, dash='dash'),
                marker=dict(size=4)
            ),
            row=i+1, col=1
        )
    
    fig.update_layout(
        title='Interactive Prediction Analysis',
        height=900,
        showlegend=True,
        hovermode='x unified'
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"✅ Interactive chart saved to: {save_path}")
    
    fig.show()

def plot_feature_importance_analysis(model, input_data, feature_names=['Open', 'High', 'Low', 'Close'], save_path: str = None):
    """
    Analyze feature importance using permutation importance.
    
    BEST PRACTICES:
    - Permutation-based importance
    - Statistical significance testing
    - Clear visualization of impact
    """
    from sklearn.inspection import permutation_importance
    
    # Create a simple wrapper for the model
    def model_predict(X):
        return model.predict(X, verbose=0)
    
    # Calculate permutation importance
    perm_importance = permutation_importance(
        model_predict, input_data, 
        n_repeats=10, random_state=42, n_jobs=-1
    )
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
    
    # Bar plot of importance scores
    importance_scores = perm_importance.importances_mean
    importance_std = perm_importance.importances_std
    
    bars = ax1.bar(feature_names, importance_scores, yerr=importance_std, 
                   capsize=5, color='skyblue', edgecolor='navy', alpha=0.7)
    ax1.set_title('Permutation Importance Scores')
    ax1.set_ylabel('Importance Score')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, importance_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + importance_std[0]/2,
                f'{score:.4f}', ha='center', va='bottom')
    
    # Box plot of importance distributions
    ax2.boxplot(perm_importance.importances.T, labels=feature_names)
    ax2.set_title('Importance Score Distributions')
    ax2.set_ylabel('Importance Score')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Feature importance analysis saved to: {save_path}")
    plt.show()
    
    return perm_importance
