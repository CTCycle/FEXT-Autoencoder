import torch
import keras
from keras import layers

from FEXT.commons.constants import CONFIG
from FEXT.commons.logger import logger


# [LOSS FUNCTION]
###############################################################################
class StructuralSimilarityIndexMeasure(keras.losses.Loss):
    
    def __init__(self, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, 
                 name='StructuralSimilarityIndexMeasure', **kwargs):
        super(StructuralSimilarityIndexMeasure, self).__init__(name=name, **kwargs)
        self.max_val = max_val
        self.filter_size = filter_size
        self.filter_sigma = filter_sigma
        self.k1 = k1
        self.k2 = k2
        
        # Compute the constants C1 and C2 based on max_val
        self.C1 = (k1 * max_val) ** 2
        self.C2 = (k2 * max_val) ** 2
        
        # Create the Gaussian window for StructuralSimilarityIndexMeasure computation
        self.window = self._create_gaussian_window(filter_size, filter_sigma)

    #--------------------------------------------------------------------------   
    def _create_gaussian_window(self, size, sigma):
        
        coords = keras.ops.arange(size) - size // 2
        g = keras.ops.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / keras.ops.sum(g)        
       
        window = keras.ops.outer(g, g)
        window = window / keras.ops.sum(window)        
        
        window = window[None, None, ...]
        return window
    
    #--------------------------------------------------------------------------   
    def _apply_conv2d(self, x, window):
        # x: [batch_size, channels, height, width]
        # window: [1, 1, filter_size, filter_size]
        
        # Ensure the input has shape [batch_size, channels, height, width]
        if keras.ops.shape(x)[1] != keras.ops.shape(window)[0]:
            x = keras.ops.transpose(x, [0, 3, 1, 2])  # Convert from NHWC to NCHW if necessary
        
        padding = self.filter_size // 2
        
        # Perform per-channel convolution by using groups
        out = keras.ops.conv(x, window.repeat(keras.ops.shape(x)[1], axis=0),  
                            strides=(1, 1),
                            padding=padding,
                            groups= keras.ops.shape(x)[1])        
        
        if keras.ops.shape(out)[1] == keras.ops.shape(x)[1]:
            out = keras.ops.transpose(out, [0, 2, 3, 1])  
        
        return out

    #-------------------------------------------------------------------------- 
    def call(self, y_true, y_pred):
        # Ensure inputs are float32
        y_true = keras.ops.cast(y_true, 'float32')
        y_pred = keras.ops.cast(y_pred, 'float32')
        
        # Reshape inputs to [batch_size, channels, height, width]
        y_true = keras.ops.transpose(y_true, [0, 3, 1, 2])  # NHWC to NCHW
        y_pred = keras.ops.transpose(y_pred, [0, 3, 1, 2])
        
        # Compute means
        mu_y_true = self._apply_conv2d(y_true, self.window)
        mu_y_pred = self._apply_conv2d(y_pred, self.window)
        
        # Compute squares of means
        mu_y_true_sq = mu_y_true ** 2
        mu_y_pred_sq = mu_y_pred ** 2
        mu_y_true_mu_y_pred = mu_y_true * mu_y_pred
        
        # Compute variances and covariance
        sigma_y_true_sq = self._apply_conv2d(y_true ** 2, self.window) - mu_y_true_sq
        sigma_y_pred_sq = self._apply_conv2d(y_pred ** 2, self.window) - mu_y_pred_sq
        sigma_y_true_y_pred = self._apply_conv2d(y_true * y_pred, self.window) - mu_y_true_mu_y_pred
        
        # Compute StructuralSimilarityIndexMeasure map
        StructuralSimilarityIndexMeasure_numerator = (2 * mu_y_true_mu_y_pred + self.C1) * (2 * sigma_y_true_y_pred + self.C2)
        StructuralSimilarityIndexMeasure_denominator = (mu_y_true_sq + mu_y_pred_sq + self.C1) * (sigma_y_true_sq + sigma_y_pred_sq + self.C2)
        StructuralSimilarityIndexMeasure_map = StructuralSimilarityIndexMeasure_numerator / StructuralSimilarityIndexMeasure_denominator
        
        # Compute the mean StructuralSimilarityIndexMeasure over the batch
        loss = 1 - keras.ops.mean(StructuralSimilarityIndexMeasure_map)
        return loss

    #-------------------------------------------------------------------------- 
    def get_config(self):
        base_config = super(StructuralSimilarityIndexMeasure, self).get_config()
        return {**base_config,
                'max_val': self.max_val,
                'filter_size': self.filter_size,
                'filter_sigma': self.filter_sigma,
                'k1': self.k1,
                'k2': self.k2}
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

# [LOSS FUNCTION]
###############################################################################
class WeightedMeanAbsoluteError(keras.losses.Loss):
    
    
    def __init__(self, name='WeightedMeanAbsoluteError', size=(128, 128), **kwargs):        
        super(WeightedMeanAbsoluteError, self).__init__(name=name, **kwargs)
        self.loss = keras.losses.MeanAbsoluteError(reduction=None)
        self.size = size
        
    #--------------------------------------------------------------------------    
    def call(self, y_true, y_pred):
        loss = self.loss(y_true, y_pred)
        penalty_factor = keras.ops.power((self.size[0] * self.size[1] / 255), 1/3)
        loss = loss * penalty_factor       

        return loss
    
    #--------------------------------------------------------------------------    
    def get_config(self):
        base_config = super(WeightedMeanAbsoluteError, self).get_config()
        return {**base_config, 'name': self.name}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)