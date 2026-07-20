import tensorflow as tf

from config import (
    CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LEARNING_RATE,
)


def build_augmentation():

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(
                factor=0.02,
                fill_mode="constant",
                fill_value=0.0,
            ),

            tf.keras.layers.RandomTranslation(
                height_factor=0.04,
                width_factor=0.04,
                fill_mode="constant",
                fill_value=0.0,
            ),

            tf.keras.layers.RandomZoom(
                height_factor=0.05,
                width_factor=0.05,
                fill_mode="constant",
                fill_value=0.0,
            ),

            tf.keras.layers.RandomContrast(
                factor=0.10,
            ),
        ],
        name="training_augmentation",
    )


def convolution_block(
    inputs,
    filters,
    use_pooling=True,
):

    outputs = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=3,
        padding="same",
        use_bias=False,
        kernel_regularizer=(
            tf.keras.regularizers.l2(
                1e-4
            )
        ),
    )(
        inputs
    )


    outputs = (
        tf.keras.layers.BatchNormalization()
        (
            outputs
        )
    )


    outputs = tf.keras.layers.ReLU()(
        outputs
    )


    if use_pooling:

        outputs = (
            tf.keras.layers.MaxPooling2D(
                pool_size=2
            )
            (
                outputs
            )
        )


    return outputs


def build_model(
    use_augmentation=True,
):

    inputs = tf.keras.Input(
        shape=(
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            CHANNELS,
        ),
        name="chest_xray",
    )


    outputs = inputs


    if use_augmentation:

        outputs = build_augmentation()(
            outputs
        )


    outputs = convolution_block(
        inputs=outputs,
        filters=32,
        use_pooling=True,
    )


    outputs = convolution_block(
        inputs=outputs,
        filters=64,
        use_pooling=True,
    )


    outputs = convolution_block(
        inputs=outputs,
        filters=128,
        use_pooling=True,
    )


    outputs = convolution_block(
        inputs=outputs,
        filters=256,
        use_pooling=False,
    )


    outputs = (
        tf.keras.layers.GlobalAveragePooling2D()
        (
            outputs
        )
    )


    outputs = tf.keras.layers.Dropout(
        rate=0.30
    )(
        outputs
    )


    outputs = tf.keras.layers.Dense(
        units=1,
        activation="sigmoid",
        name="opacity_probability",
    )(
        outputs
    )


    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="rsna_federated_cnn",
    )


    return model


def compile_model(
    model,
    learning_rate=LEARNING_RATE,
):

    model.compile(
        optimizer=(
            tf.keras.optimizers.Adam(
                learning_rate=(
                    learning_rate
                )
            )
        ),

        loss=(
            tf.keras.losses.BinaryCrossentropy()
        ),

        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy",
            ),

            tf.keras.metrics.Precision(
                name="precision",
            ),

            tf.keras.metrics.Recall(
                name="sensitivity",
            ),

            tf.keras.metrics.AUC(
                name="roc_auc",
                curve="ROC",
            ),

            tf.keras.metrics.AUC(
                name="pr_auc",
                curve="PR",
            ),
        ],
    )


    return model