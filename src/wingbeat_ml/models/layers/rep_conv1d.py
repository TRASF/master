from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


@keras.utils.register_keras_serializable(
    package="wingbeat_ml"
)
class RepConv1D(keras.layers.Layer):
    """
    Re-parameterizable 1-D convolution.

    Training:
        N identical Conv1D + BN branches
        -> sum
        -> activation

    Deployment:
        one fused Conv1D
        -> activation
    """

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        strides: int = 1,
        padding: str = "valid",
        dilation_rate: int = 1,
        branches: int = 2,
        activation: str | None = "relu",
        use_batch_norm: bool = True,
        bn_momentum: float = 0.99,
        bn_epsilon: float = 1e-3,
        kernel_regularizer=None,
        deploy: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        if filters <= 0:
            raise ValueError(
                "filters must be > 0"
            )

        if branches <= 0:
            raise ValueError(
                "branches must be > 0"
            )

        if (
            strides > 1
            and dilation_rate > 1
        ):
            raise ValueError(
                "strides > 1 is incompatible with "
                "dilation_rate > 1."
            )

        self.filters = int(filters)
        self.kernel_size = int(kernel_size)
        self.strides = int(strides)
        self.padding = str(padding)
        self.dilation_rate = int(
            dilation_rate
        )
        self.branches = int(branches)

        self.activation_name = activation
        self.activation = (
            keras.activations.get(
                activation
            )
            if activation
            else None
        )

        self.use_batch_norm = bool(
            use_batch_norm
        )

        self.bn_momentum = float(
            bn_momentum
        )

        self.bn_epsilon = float(
            bn_epsilon
        )

        self.kernel_regularizer = (
            keras.regularizers.get(
                kernel_regularizer
            )
        )

        self.deploy = bool(deploy)

        self.conv_branches = []
        self.bn_branches = []

        self.reparam_conv = None

        if self.deploy:
            self.reparam_conv = (
                keras.layers.Conv1D(
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    strides=self.strides,
                    padding=self.padding,
                    dilation_rate=(
                        self.dilation_rate
                    ),
                    activation=None,
                    use_bias=True,
                    name="reparam_conv",
                )
            )

        else:
            for index in range(
                self.branches
            ):
                conv = keras.layers.Conv1D(
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    strides=self.strides,
                    padding=self.padding,
                    dilation_rate=(
                        self.dilation_rate
                    ),
                    activation=None,
                    use_bias=(
                        not self.use_batch_norm
                    ),
                    kernel_regularizer=(
                        self.kernel_regularizer
                    ),
                    name=f"conv_{index}",
                )

                self.conv_branches.append(
                    conv
                )

                if self.use_batch_norm:
                    bn = (
                        keras.layers.BatchNormalization(
                            momentum=(
                                self.bn_momentum
                            ),
                            epsilon=(
                                self.bn_epsilon
                            ),
                            name=f"bn_{index}",
                        )
                    )
                else:
                    bn = None

                self.bn_branches.append(
                    bn
                )

    def build(self, input_shape):
        if self.deploy:
            self.reparam_conv.build(
                input_shape
            )

        else:
            for conv, bn in zip(
                self.conv_branches,
                self.bn_branches,
            ):
                conv.build(input_shape)

                if bn is not None:
                    output_shape = (
                        conv.compute_output_shape(
                            input_shape
                        )
                    )

                    bn.build(output_shape)

        super().build(input_shape)

    def call(
        self,
        inputs,
        training=None,
    ):
        if self.deploy:
            x = self.reparam_conv(
                inputs
            )

        else:
            outputs = []

            for conv, bn in zip(
                self.conv_branches,
                self.bn_branches,
            ):
                branch = conv(inputs)

                if bn is not None:
                    branch = bn(
                        branch,
                        training=training,
                    )

                outputs.append(branch)

            x = tf.add_n(outputs)

        if self.activation is not None:
            x = self.activation(x)

        return x

    @staticmethod
    def _fuse_conv_bn(
        conv,
        bn,
    ):
        kernel = conv.kernel

        if conv.use_bias:
            bias = conv.bias
        else:
            bias = tf.zeros(
                shape=(kernel.shape[-1],),
                dtype=kernel.dtype,
            )

        if bn is None:
            return kernel, bias

        gamma = (
            bn.gamma
            if bn.scale
            else tf.ones_like(
                bn.moving_mean
            )
        )

        beta = (
            bn.beta
            if bn.center
            else tf.zeros_like(
                bn.moving_mean
            )
        )

        mean = bn.moving_mean
        variance = bn.moving_variance

        std = tf.sqrt(
            variance + bn.epsilon
        )

        scale = gamma / std

        fused_kernel = (
            kernel
            * scale[
                tf.newaxis,
                tf.newaxis,
                :
            ]
        )

        fused_bias = (
            beta
            + (bias - mean) * scale
        )

        return (
            fused_kernel,
            fused_bias,
        )

    def get_equivalent_kernel_bias(
        self,
    ):
        if self.deploy:
            weights = (
                self.reparam_conv.get_weights()
            )

            return (
                tf.convert_to_tensor(
                    weights[0]
                ),
                tf.convert_to_tensor(
                    weights[1]
                ),
            )

        kernels = []
        biases = []

        for conv, bn in zip(
            self.conv_branches,
            self.bn_branches,
        ):
            kernel, bias = (
                self._fuse_conv_bn(
                    conv,
                    bn,
                )
            )

            kernels.append(kernel)
            biases.append(bias)

        fused_kernel = tf.add_n(
            kernels
        )

        fused_bias = tf.add_n(
            biases
        )

        return (
            fused_kernel,
            fused_bias,
        )

    def to_deploy_layer(self):
        """
        Return a new RepConv1D containing a single fused Conv1D.
        """
        if not self.built:
            raise RuntimeError(
                "RepConv1D must be built before "
                "conversion."
            )

        fused_kernel, fused_bias = (
            self.get_equivalent_kernel_bias()
        )

        deploy_layer = RepConv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=(
                self.dilation_rate
            ),
            branches=self.branches,
            activation=(
                self.activation_name
            ),
            use_batch_norm=(
                self.use_batch_norm
            ),
            bn_momentum=(
                self.bn_momentum
            ),
            bn_epsilon=(
                self.bn_epsilon
            ),
            deploy=True,
            name=(
                f"{self.name}_deploy"
            ),
        )

        input_channels = int(
            self.conv_branches[
                0
            ].kernel.shape[1]
        )

        deploy_layer.build(
            (
                None,
                None,
                input_channels,
            )
        )

        deploy_layer.reparam_conv.set_weights(
            [
                fused_kernel.numpy(),
                fused_bias.numpy(),
            ]
        )

        return deploy_layer

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": (
                    self.dilation_rate
                ),
                "branches": self.branches,
                "activation": (
                    self.activation_name
                ),
                "use_batch_norm": (
                    self.use_batch_norm
                ),
                "bn_momentum": (
                    self.bn_momentum
                ),
                "bn_epsilon": (
                    self.bn_epsilon
                ),
                "kernel_regularizer": (
                    keras.regularizers.serialize(
                        self.kernel_regularizer
                    )
                ),
                "deploy": self.deploy,
            }
        )

        return config


def reparameterize_repconv_model(
    model: keras.Model,
) -> keras.Model:
    """
    Clone a Functional/Sequential model and replace each
    training RepConv1D with its single-convolution deploy form.
    """

    def clone_function(layer):
        if isinstance(
            layer,
            RepConv1D,
        ):
            config = layer.get_config()
            config["deploy"] = True

            return RepConv1D.from_config(
                config
            )

        return layer.__class__.from_config(
            layer.get_config()
        )

    deploy_model = keras.models.clone_model(
        model,
        clone_function=clone_function,
    )

    for source, target in zip(
        model.layers,
        deploy_model.layers,
    ):
        if isinstance(
            source,
            RepConv1D,
        ):
            kernel, bias = (
                source.get_equivalent_kernel_bias()
            )

            target.reparam_conv.set_weights(
                [
                    kernel.numpy(),
                    bias.numpy(),
                ]
            )

        else:
            source_weights = (
                source.get_weights()
            )

            if source_weights:
                target.set_weights(
                    source_weights
                )

    return deploy_model


__all__ = [
    "RepConv1D",
    "reparameterize_repconv_model",
]
