import unittest
import numpy as np
from keras import layers, models
from FEXT.commons.utils.learning.bottleneck import CompressionLayer

class TestCompressionLayer(unittest.TestCase):

    def setUp(self):
        self.units = 64
        self.dropout_rate = 0.2
        self.layer = CompressionLayer(units=self.units, dropout_rate=self.dropout_rate)

    def test_layer_output_shape(self):
        input_shape = (1, 32, 32, 3)
        inputs = np.random.random(input_shape).astype(np.float32)
        model = models.Sequential([self.layer])
        output = model.predict(inputs)
        self.assertEqual(output.shape, (1, 1024, self.units))

    def test_layer_config(self):
        config = self.layer.get_config()
        new_layer = CompressionLayer.from_config(config)
        self.assertEqual(new_layer.units, self.units)
        self.assertEqual(new_layer.dropout_rate, self.dropout_rate)

    def test_layer_serialization(self):
        config = self.layer.get_config()
        new_layer = CompressionLayer.from_config(config)
        self.assertEqual(new_layer.get_config(), config)

if __name__ == '__main__':
    unittest.main()