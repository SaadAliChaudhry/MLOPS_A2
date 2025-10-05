
from typing import List
from absl import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow_transform.tf_metadata import schema_utils

from tfx import v1 as tfx
from tfx_bsl.public import tfxio
from tensorflow_metadata.proto.v0 import schema_pb2
import tensorflow_transform as tft

_FEATURE_KEYS = [
    'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g'
]
_LABEL_KEY = 'species'
_TRAIN_BATCH_SIZE = 32
_EVAL_BATCH_SIZE = 16


def _input_fn(file_pattern: List[str],
              data_accessor: tfx.components.DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int) -> tf.data.Dataset:  # Remove default value
    """
    Generates features and label for training using transformed data.
    """
    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(
            batch_size=batch_size, label_key=_LABEL_KEY),
        schema=tf_transform_output.transformed_metadata.schema).repeat()


def _build_keras_model() -> tf.keras.Model:
    """
    Tuned model with improved architecture and regularization.
    """
    transformed_feature_keys = [f"{key}_normalized" for key in _FEATURE_KEYS]
    
    inputs = [keras.layers.Input(shape=(1,), name=f) for f in transformed_feature_keys]
    d = keras.layers.concatenate(inputs)
    
    # Deeper network with more units
    for _ in range(4):
        d = keras.layers.Dense(32, activation='relu',
                              kernel_regularizer=keras.regularizers.l2(0.01))(d)
        d = keras.layers.Dropout(0.3)(d)
    
    outputs = keras.layers.Dense(3)(d)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(5e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()])

    model.summary(print_fn=logging.info)
    return model


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """
    Returns a function that parses raw examples and applies transformations.
    """
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(_LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)

    return serve_tf_examples_fn


def run_fn(fn_args: tfx.components.FnArgs):
    """
    Train the model based on given args using transformed features.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size=_TRAIN_BATCH_SIZE)
    
    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size=_EVAL_BATCH_SIZE)

    model = _build_keras_model()
    
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps)

    signatures = {
        'serving_default':
            _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
                tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')),
    }
    
    # FIXED: Use tf.saved_model.save() for Keras 3 compatibility
    tf.saved_model.save(model, fn_args.serving_model_dir, signatures=signatures)
