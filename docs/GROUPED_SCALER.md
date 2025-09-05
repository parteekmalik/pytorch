# Grouped Scaler

## Overview
The `GroupedScaler` class provides intelligent normalization for time series data by applying different scaling strategies to different feature groups. This approach ensures optimal scaling for heterogeneous features while maintaining the relationships within each group.

## Why Grouped Scaling?

### Problem with Standard Scaling
Traditional MinMaxScaler applies the same scaling to all features, which can be problematic for time series data with different scales:

- **Price data**: Ranges from 0 to 100,000+ (e.g., BTC price)
- **Volume data**: Ranges from 0 to millions
- **Time data**: Ranges from 0 to 1439 (minutes of day)
- **Percentage data**: Ranges from -1 to +1

### Solution: Grouped Scaling
GroupedScaler categorizes features into logical groups and applies appropriate scaling to each group:

- **Price Group**: Scales price-related features together
- **Volume Group**: Scales volume-related features together  
- **Time Group**: Uses fixed ranges for time features
- **Other Group**: Handles remaining features

## Feature Categorization

### Price Group
Features containing price-related keywords:
- `Open`, `High`, `Low`, `Close`
- `Price_Range`, `Price_Change`, `Price_Change_Pct`
- Any feature with 'price' in the name

### Volume Group
Features containing volume-related keywords:
- `Volume`, `Volume_MA_5`, `Volume_MA_10`
- `Quote asset volume`, `Taker buy base asset volume`
- Any feature with 'volume' or 'trades' in the name

### Time Group
Features containing time-related keywords:
- `Minutes_of_day`, `Hour`, `Minute`
- `Interval_5min`, `Interval_sin`, `Interval_cos`
- Any feature with 'time', 'interval', 'sin', 'cos' in the name

### Other Group
All remaining features that don't match the above categories.

## Special Handling

### Time Features
Time features receive special treatment:

```python
# Minutes_of_day uses fixed range [0, 1439]
if 'minutes_of_day' in feature_name.lower():
    scaler.fit([[0], [1439]])  # Fixed range
```

This ensures consistent scaling regardless of the actual data range.

### Cyclical Features
Features like `Interval_sin` and `Interval_cos` are handled as regular features but benefit from being in the time group.

## Usage

### Basic Usage

```python
from utils import GroupedScaler

# Create scaler
scaler = GroupedScaler()

# Fit on training data
X_train_scaled = scaler.fit_transform(X_train, feature_names)

# Transform test data
X_test_scaled = scaler.transform(X_test)

# Inverse transform predictions
predictions_original = scaler.inverse_transform(predictions_scaled)
```

### With BinanceDataOrganizer

```python
from utils import BinanceDataOrganizer, DataConfig

# Create organizer
config = DataConfig('BTCUSDT', '5m', '2021-01-01', '2021-01-31', 10, 1)
organizer = BinanceDataOrganizer(config)
organizer.process_all()

# Get scaled data (automatically uses GroupedScaler)
data = organizer.get_scaled_data('all')
X_train_scaled = data['X_train_scaled']
X_test_scaled = data['X_test_scaled']

# Get scaler for inverse transformations
scalers = organizer.get_scalers()
scaler_X = scalers['X']  # GroupedScaler
scaler_y = scalers['y']  # MinMaxScaler
```

## API Reference

### Methods

#### `fit(X, feature_names)`
Fits the scaler on training data.

**Parameters:**
- `X`: Training data array (2D or 3D)
- `feature_names`: List of feature column names

**Returns:** Self (for method chaining)

#### `transform(X)`
Transforms data using fitted scalers.

**Parameters:**
- `X`: Data array to transform (2D or 3D)

**Returns:** Scaled data array

#### `fit_transform(X, feature_names)`
Fits and transforms data in one step.

**Parameters:**
- `X`: Training data array (2D or 3D)
- `feature_names`: List of feature column names

**Returns:** Scaled data array

#### `inverse_transform(X)`
Inverse transforms scaled data back to original scale.

**Parameters:**
- `X`: Scaled data array (2D or 3D)

**Returns:** Original scale data array

#### `get_feature_groups()`
Returns feature groups with their names.

**Returns:** Dictionary mapping group names to feature lists

#### `get_scaling_info()`
Returns scaling information for each group.

**Returns:** Dictionary with scaling parameters for each group

### Properties

#### `is_fitted`
Boolean indicating if the scaler has been fitted.

#### `feature_groups`
Dictionary mapping group names to feature indices.

#### `scalers`
Dictionary mapping group names to MinMaxScaler instances.

## Advanced Usage

### Custom Feature Groups
You can extend the categorization logic by modifying the `_categorize_features` method:

```python
def _categorize_features(self, feature_names):
    groups = {
        'price': [],
        'volume': [],
        'time': [],
        'custom': [],  # Add custom group
        'other': []
    }
    
    for i, col in enumerate(feature_names):
        col_lower = col.lower()
        
        # Custom categorization logic
        if 'custom_keyword' in col_lower:
            groups['custom'].append(i)
        # ... existing logic
    
    return {k: v for k, v in groups.items() if v}
```

### Handling New Data
When making predictions on new data, ensure the feature names match the training data:

```python
# New data must have same feature names
new_data = create_features(new_raw_data)
new_features = new_data[feature_names].values

# Scale using fitted scaler
new_features_scaled = scaler.transform(new_features.reshape(1, -1))
```

## Performance Considerations

### Memory Usage
- Each group has its own MinMaxScaler instance
- Memory usage scales with number of groups
- Consider grouping similar features together

### Speed
- Grouped scaling is slightly slower than single scaler
- Benefits outweigh the small performance cost
- Optimized for time series data

### Accuracy
- Maintains relationships within feature groups
- Prevents one feature from dominating others
- Improves model training stability

## Troubleshooting

### Common Issues

**Feature Name Mismatch:**
```
ValueError: Feature names don't match training data
```
- Ensure feature names are identical between training and prediction
- Check for typos in feature names

**Unfitted Scaler:**
```
ValueError: Scaler must be fitted before transform
```
- Call `fit()` or `fit_transform()` before `transform()`
- Check `is_fitted` property

**Empty Groups:**
- Some groups may be empty if no features match the criteria
- This is normal and handled automatically

### Debug Information
The scaler provides detailed logging:

```python
# Enable verbose output
scaler = GroupedScaler()
scaler.fit(X_train, feature_names)

# Output:
# ✅ Price group scaler fitted on 3 features
# ✅ Volume group scaler fitted on 6 features  
# ⏰ Using fixed range [0, 1439] for Minutes_of_day feature
# ✅ Time group scaler fitted on 1 features
```

## Best Practices

1. **Consistent Feature Names**: Use consistent naming conventions
2. **Group Similar Features**: Keep related features in the same group
3. **Handle Time Features**: Use fixed ranges for cyclical time features
4. **Validate Scaling**: Check scaled data ranges and distributions
5. **Save Scalers**: Persist fitted scalers for production use

## Integration Examples

### With LSTM Models
```python
# Train model with grouped scaling
X_train_scaled = scaler_X.fit_transform(X_train, feature_names)
model.fit(X_train_scaled, y_train)

# Make predictions
predictions_scaled = model.predict(X_test_scaled)
predictions = scaler_y.inverse_transform(predictions_scaled)
```

### With Production Systems
```python
# Save fitted scaler
import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Load in production
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
```
