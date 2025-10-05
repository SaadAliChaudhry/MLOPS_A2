
import tensorflow as tf
import tensorflow_transform as tft

_FEATURE_KEYS = [
    'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g'
]
_LABEL_KEY = 'species'

def preprocessing_fn(inputs):
    """
    Preprocessing function for Transform component.
    This ensures consistent transformations at training and serving time.

    Args:
        inputs: Dictionary of input features

    Returns:
        Dictionary of transformed features
    """
    outputs = {}

    # Normalize all numeric features to z-score (mean=0, std=1)
    for key in _FEATURE_KEYS:
        outputs[f"{key}_normalized"] = tft.scale_to_z_score(inputs[key])

    # Pass through the label unchanged
    outputs[_LABEL_KEY] = inputs[_LABEL_KEY]

    return outputs
