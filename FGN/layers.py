# code taken from https://github.com/KdaiP/1D-Condition-method-pytorch/blob/main/README.md on 19/11/25
import torch
import torch.nn as nn

class ConditionalLayerNorm(nn.Module):
    def __init__(self, in_channels, cond_channels, eps=1e-5):
        """
        Conditional Layer Normalization module.
        
        Parameters:
        in_channels: The number of channels in the input feature maps.
        cond_channels: The number of channels in the conditioning input.
        eps: A small number to prevent division by zero in normalization.
        """
        super(ConditionalLayerNorm, self).__init__()
        self.eps = eps
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        
        self.weight_transform = nn.Linear(cond_channels, in_channels)
        self.bias_transform = nn.Linear(cond_channels, in_channels) 
        
        self.reset_parameters()

    def reset_parameters(self):
        #initially conditional layer norm is approx just layer norm! 
        nn.init.constant_(self.weight_transform.weight, 0.0)
        nn.init.constant_(self.weight_transform.bias, 1.0)
        nn.init.constant_(self.bias_transform.weight, 0.0)
        nn.init.constant_(self.bias_transform.bias, 0.0)

    def forward(self, x, c):
        """
        Parameters:
        x (Tensor): The input feature maps with shape [batch_size, time, in_channels].
        c (Tensor): The conditioning input with shape [batch_size, 1, cond_channels].
        
        Returns:
        Tensor: The modulated feature maps with the same shape as input x.
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False)
        
        x_normalized = (x - mean) / (std + self.eps)
        
        gamma = self.weight_transform(c)
        beta = self.bias_transform(c)
        
        out = gamma * x_normalized + beta
        
        return out
