import tensorflow as tf

import yaml

class MosSongPlusModel:
    """
    Dynamically builds the MosSongPlus CNN architecture from YAML configurations.
    """
    def __init__(self, model_yaml_path="configs/model.yaml", defaults_yaml_path="configs/defaults.yaml"):
        # Load configurations
        with open(model_yaml_path, 'r') as f:
            self.model_config = yaml.safe_load(f)['model']['mosssongplus']
            
        with open(defaults_yaml_path, 'r') as f:
            self.defaults = yaml.safe_load(f)

        # Calculate shape constraints
        sample_rate = self.defaults['audio']['sample_rate']
        duration = self.defaults['audio']['duration']
        
        self.segment_size = int(sample_rate * duration)
        self.num_classes = len(self.defaults['class_dict'])

    def build(self) -> tf.keras.Sequential:
        model = tf.keras.Sequential(name="MosSongPlus")

        # 1. Input Layer
        model.add(tf.keras.layers.InputLayer(shape=(self.segment_size, 1)))

        # 2. Convolutional Layers
        for conv_params in self.model_config.get('conv', []):
            model.add(tf.keras.layers.Conv1D(**conv_params))

        # 3. Transition to Dense
        model.add(tf.keras.layers.MaxPooling1D(pool_size=self.model_config.get('pooling', 3)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dropout(rate=self.model_config.get('dropout', 0.5)))


        # 4. Dense Layers
        for dense_params in self.model_config.get('dense', []):
            model.add(tf.keras.layers.Dense(**dense_params))

        # 5. Output Layer
        model.add(tf.keras.layers.Dense(units=self.num_classes))

        return model 
