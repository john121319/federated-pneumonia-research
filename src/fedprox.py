import tensorflow as tf


def snapshot_trainable_variables(model):
    return [
        tf.identity(variable)
        for variable in model.trainable_variables
    ]


def validate_reference_variables(
    model,
    reference_variables,
):
    if len(model.trainable_variables) != len(reference_variables):
        raise ValueError(
            "The trainable-variable count does not match "
            "the global reference."
        )

    for variable_index, (
        variable,
        reference_variable,
    ) in enumerate(
        zip(
            model.trainable_variables,
            reference_variables,
        )
    ):
        if tuple(variable.shape) != tuple(reference_variable.shape):
            raise ValueError(
                "Trainable-variable shapes do not match at "
                f"index {variable_index}."
            )


def compute_proximal_penalty(
    model,
    reference_variables,
    mu,
):
    mu = float(mu)

    if mu < 0.0:
        raise ValueError(
            "FedProx mu must be non-negative."
        )

    validate_reference_variables(
        model=model,
        reference_variables=reference_variables,
    )

    if mu == 0.0:
        return tf.constant(
            0.0,
            dtype=tf.float32,
        )

    squared_distances = []

    for variable, reference_variable in zip(
        model.trainable_variables,
        reference_variables,
    ):
        reference_variable = tf.cast(
            reference_variable,
            variable.dtype,
        )

        squared_distances.append(
            tf.reduce_sum(
                tf.square(
                    variable
                    - reference_variable
                )
            )
        )

    return tf.cast(
        0.5
        * mu,
        tf.float32,
    ) * tf.cast(
        tf.add_n(
            squared_distances
        ),
        tf.float32,
    )


def train_fedprox_client(
    model,
    dataset,
    class_weights,
    mu,
    local_epochs,
    learning_rate,
):
    mu = float(mu)
    local_epochs = int(local_epochs)
    learning_rate = float(learning_rate)

    if mu <= 0.0:
        raise ValueError(
            "FedProx mu must be greater than zero."
        )

    if local_epochs <= 0:
        raise ValueError(
            "Local epochs must be positive."
        )

    if learning_rate <= 0.0:
        raise ValueError(
            "Learning rate must be positive."
        )

    if 0 not in class_weights or 1 not in class_weights:
        raise ValueError(
            "Class weights must contain keys 0 and 1."
        )

    negative_weight = float(
        class_weights[0]
    )

    positive_weight = float(
        class_weights[1]
    )

    if negative_weight <= 0.0 or positive_weight <= 0.0:
        raise ValueError(
            "Class weights must be positive."
        )

    reference_variables = snapshot_trainable_variables(
        model
    )

    validate_reference_variables(
        model=model,
        reference_variables=reference_variables,
    )

    initial_proximal_loss = float(
        compute_proximal_penalty(
            model=model,
            reference_variables=reference_variables,
            mu=mu,
        ).numpy()
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    binary_crossentropy = (
        tf.keras.losses.BinaryCrossentropy(
            reduction=(
                tf.keras.losses.Reduction.NONE
            )
        )
    )

    total_loss_metric = tf.keras.metrics.Mean(
        name="loss"
    )

    classification_loss_metric = tf.keras.metrics.Mean(
        name="classification_loss"
    )

    regularization_loss_metric = tf.keras.metrics.Mean(
        name="regularization_loss"
    )

    proximal_loss_metric = tf.keras.metrics.Mean(
        name="proximal_loss"
    )

    accuracy_metric = tf.keras.metrics.BinaryAccuracy(
        name="accuracy"
    )

    precision_metric = tf.keras.metrics.Precision(
        name="precision"
    )

    sensitivity_metric = tf.keras.metrics.Recall(
        name="sensitivity"
    )

    roc_auc_metric = tf.keras.metrics.AUC(
        name="roc_auc",
        curve="ROC",
    )

    pr_auc_metric = tf.keras.metrics.AUC(
        name="pr_auc",
        curve="PR",
    )

    mu_tensor = tf.constant(
        mu,
        dtype=tf.float32,
    )

    negative_weight_tensor = tf.constant(
        negative_weight,
        dtype=tf.float32,
    )

    positive_weight_tensor = tf.constant(
        positive_weight,
        dtype=tf.float32,
    )

    @tf.function
    def train_step(
        images,
        labels,
    ):
        labels = tf.cast(
            tf.reshape(
                labels,
                (
                    -1,
                    1,
                ),
            ),
            tf.float32,
        )

        with tf.GradientTape() as tape:
            probabilities = model(
                images,
                training=True,
            )

            per_example_loss = binary_crossentropy(
                labels,
                probabilities,
            )

            sample_weights = tf.where(
                labels[:, 0] >= 0.5,
                positive_weight_tensor,
                negative_weight_tensor,
            )

            classification_loss = tf.reduce_mean(
                per_example_loss
                * sample_weights
            )

            if model.losses:
                regularization_loss = tf.add_n(
                    model.losses
                )
            else:
                regularization_loss = tf.constant(
                    0.0,
                    dtype=tf.float32,
                )

            squared_distances = []

            for variable, reference_variable in zip(
                model.trainable_variables,
                reference_variables,
            ):
                squared_distances.append(
                    tf.reduce_sum(
                        tf.square(
                            variable
                            - tf.cast(
                                reference_variable,
                                variable.dtype,
                            )
                        )
                    )
                )

            proximal_loss = (
                0.5
                * mu_tensor
                * tf.cast(
                    tf.add_n(
                        squared_distances
                    ),
                    tf.float32,
                )
            )

            total_loss = (
                classification_loss
                + tf.cast(
                    regularization_loss,
                    tf.float32,
                )
                + proximal_loss
            )

        gradients = tape.gradient(
            total_loss,
            model.trainable_variables,
        )

        if any(
            gradient is None
            for gradient
            in gradients
        ):
            raise RuntimeError(
                "A trainable variable received no gradient."
            )

        for gradient in gradients:
            tf.debugging.assert_all_finite(
                gradient,
                "A FedProx gradient contained a non-finite value.",
            )

        optimizer.apply_gradients(
            zip(
                gradients,
                model.trainable_variables,
            )
        )

        batch_size = tf.cast(
            tf.shape(
                labels
            )[0],
            tf.float32,
        )

        total_loss_metric.update_state(
            total_loss,
            sample_weight=batch_size,
        )

        classification_loss_metric.update_state(
            classification_loss,
            sample_weight=batch_size,
        )

        regularization_loss_metric.update_state(
            regularization_loss,
            sample_weight=batch_size,
        )

        proximal_loss_metric.update_state(
            proximal_loss,
            sample_weight=batch_size,
        )

        accuracy_metric.update_state(
            labels,
            probabilities,
        )

        precision_metric.update_state(
            labels,
            probabilities,
        )

        sensitivity_metric.update_state(
            labels,
            probabilities,
        )

        roc_auc_metric.update_state(
            labels,
            probabilities,
        )

        pr_auc_metric.update_state(
            labels,
            probabilities,
        )

    epoch_history = []

    metrics = [
        total_loss_metric,
        classification_loss_metric,
        regularization_loss_metric,
        proximal_loss_metric,
        accuracy_metric,
        precision_metric,
        sensitivity_metric,
        roc_auc_metric,
        pr_auc_metric,
    ]

    for epoch_number in range(
        1,
        local_epochs + 1,
    ):
        for metric in metrics:
            metric.reset_state()

        batch_count = 0

        for images, labels in dataset:
            train_step(
                images,
                labels,
            )

            batch_count += 1

        if batch_count == 0:
            raise RuntimeError(
                "The client dataset produced no batches."
            )

        epoch_result = {
            "epoch": int(
                epoch_number
            ),
            "loss": float(
                total_loss_metric.result().numpy()
            ),
            "classification_loss": float(
                classification_loss_metric.result().numpy()
            ),
            "regularization_loss": float(
                regularization_loss_metric.result().numpy()
            ),
            "proximal_loss": float(
                proximal_loss_metric.result().numpy()
            ),
            "accuracy": float(
                accuracy_metric.result().numpy()
            ),
            "precision": float(
                precision_metric.result().numpy()
            ),
            "sensitivity": float(
                sensitivity_metric.result().numpy()
            ),
            "roc_auc": float(
                roc_auc_metric.result().numpy()
            ),
            "pr_auc": float(
                pr_auc_metric.result().numpy()
            ),
            "batch_count": int(
                batch_count
            ),
        }

        if not all(
            tf.math.is_finite(
                tf.constant(
                    value,
                    dtype=tf.float32,
                )
            ).numpy()
            for key, value
            in epoch_result.items()
            if key not in {
                "epoch",
                "batch_count",
            }
        ):
            raise RuntimeError(
                "FedProx training produced a non-finite metric."
            )

        epoch_history.append(
            epoch_result
        )

    final_proximal_loss = float(
        compute_proximal_penalty(
            model=model,
            reference_variables=reference_variables,
            mu=mu,
        ).numpy()
    )

    if final_proximal_loss <= 0.0:
        raise RuntimeError(
            "The final FedProx proximal penalty did not become positive."
        )

    return {
        "mu": float(
            mu
        ),
        "initial_proximal_loss": float(
            initial_proximal_loss
        ),
        "final_proximal_loss": float(
            final_proximal_loss
        ),
        "epoch_history": (
            epoch_history
        ),
        "final_epoch": (
            epoch_history[-1]
        ),
    }
