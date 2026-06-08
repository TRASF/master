import tensorflow as tf

class LossFactory:
    @staticmethod
    def get_loss(config: dict = None):
        """
        Retrieves the loss function based on the configuration.
        """
        config = config or {}
        loss_config = config.get("loss", {"name": "CategoricalCrossentropy"})

        loss_name = loss_config.get("name")
        loss_params = {k: v for k, v in loss_config.items() if k != "name"}

        # Normalize names (e.g., 'CategoricalFocalLoss' -> 'CategoricalFocalCrossentropy')
        aliases = {
            "CategoricalFocalLoss": "CategoricalFocalCrossentropy",
            "FocalLoss": "CategoricalFocalCrossentropy",
        }
        
        real_name = aliases.get(loss_name, loss_name)

        try:
            # Check tf.keras.losses for the class
            LossClass = getattr(tf.keras.losses, real_name)
            return LossClass(**loss_params)
        except AttributeError:
            raise ValueError(
                f"Loss function '{loss_name}' (resolved as '{real_name}') "
                f"not found in tf.keras.losses. Available include: "
                f"{[n for n in dir(tf.keras.losses) if 'Crossentropy' in n]}"
            )
