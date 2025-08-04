"""
SEI Oracle Model for Promoter Dataset

This module contains the Sequence-based Ensemble Interpretation (SEI) model architecture
from the SEI framework (https://github.com/FunctionLab/sei-framework).

This is the oracle model used for evaluation against D3 generated sequences for promoter data.
SEI predicts chromatin accessibility and transcription factor binding from DNA sequences.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional
from scipy.interpolate import splev


def _flip(x, dim):
    """
    Reverses the elements in a given dimension `dim` of the Tensor.

    source: https://github.com/pytorch/pytorch/issues/229
    """
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(
        x.size(0), x.size(1), -1)[:, getattr(
            torch.arange(x.size(1)-1, -1, -1),
            ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


def bs(x, df=None, knots=None, degree=3, intercept=False):
    """B-spline basis functions.
    
    Args:
        x: Input data points
        df: Number of degrees of freedom for the spline
        knots: Interior knots of the spline
        degree: Degree of the piecewise polynomial (default: 3 for cubic)
        intercept: Whether to include intercept term
        
    Returns:
        B-spline basis matrix
    """

    order = degree + 1
    inner_knots = []
    
    if df is not None and knots is None:
        n_inner_knots = df - order + (1 - intercept)
        if n_inner_knots < 0:
            n_inner_knots = 0
            print(f"df was too small; have used {order - (1 - intercept)}")
        
        if n_inner_knots > 0:
            inner_knots = np.percentile(
                x, 100 * np.linspace(0, 1, n_inner_knots + 2)[1:-1]
            )
    elif knots is not None:
        inner_knots = knots
    
    all_knots = np.concatenate(
        ([np.min(x), np.max(x)] * order, inner_knots)
    )
    all_knots.sort()
    
    n_basis = len(all_knots) - (degree + 1)
    basis = np.empty((x.shape[0], n_basis), dtype=float)
    
    for i in range(n_basis):
        coefs = np.zeros((n_basis,))
        coefs[i] = 1
        basis[:, i] = splev(x, (all_knots, coefs, degree))
    
    if not intercept:
        basis = basis[:, 1:]
    return basis


def spline_factory(n: int, df: int, log: bool = False) -> torch.Tensor:
    """Factory function for creating B-spline basis tensors.
    
    Args:
        n: Number of spatial points
        df: Degrees of freedom
        log: Whether to use logarithmic spacing
        
    Returns:
        B-spline basis tensor
    """
    if log:
        dist = np.array(np.arange(n) - n/2.0)
        dist = np.log(np.abs(dist) + 1) * (2*(dist > 0) - 1)
        n_knots = df - 4
        knots = np.linspace(np.min(dist), np.max(dist), n_knots + 2)[1:-1]
        return torch.from_numpy(
            bs(dist, knots=knots, intercept=True)
        ).float()
    else:
        dist = np.arange(n)
        return torch.from_numpy(
            bs(dist, df=df, intercept=True)
        ).float()



class BSplineTransformation(nn.Module):
    """B-spline transformation layer for spatial feature transformation.
    
    Args:
        degrees_of_freedom: Number of degrees of freedom for the spline
        log: Whether to use logarithmic spacing
        scaled: Whether to scale by spatial dimension
    """
    
    def __init__(self, degrees_of_freedom: int, log: bool = False, scaled: bool = False):
        super(BSplineTransformation, self).__init__()
        self._spline_tr = None
        self._log = log
        self._scaled = scaled
        self._df = degrees_of_freedom

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass through B-spline transformation.
        
        Args:
            input: Input tensor
            
        Returns:
            Transformed tensor using B-spline basis
        """
        if self._spline_tr is None:
            spatial_dim = input.size()[-1]
            self._spline_tr = spline_factory(spatial_dim, self._df, log=self._log)
            if self._scaled:
                self._spline_tr = self._spline_tr / spatial_dim
            if input.is_cuda:
                self._spline_tr = self._spline_tr.cuda()
        
        return torch.matmul(input, self._spline_tr)



class BSplineConv1D(nn.Module):
    """1D Convolutional layer with B-spline transformation.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolution kernel
        degrees_of_freedom: Degrees of freedom for B-spline
        stride: Convolution stride
        padding: Padding
        dilation: Dilation factor
        groups: Number of groups for grouped convolution
        bias: Whether to use bias
        log: Whether to use logarithmic B-spline spacing
        scaled: Whether to scale B-spline weights
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 degrees_of_freedom: int, stride: int = 1, padding: int = 0, 
                 dilation: int = 1, groups: int = 1, bias: bool = True, 
                 log: bool = False, scaled: bool = True):
        super(BSplineConv1D, self).__init__()
        self._df = degrees_of_freedom
        self._log = log
        self._scaled = scaled
        
        # B-spline convolution layer
        self.spline = nn.Conv1d(
            1, degrees_of_freedom, kernel_size, stride, padding, dilation, bias=False
        )
        
        # Initialize with B-spline weights (frozen)
        spline_weights = spline_factory(kernel_size, self._df, log=log)
        self.spline.weight = nn.Parameter(
            spline_weights.view(self._df, 1, kernel_size)
        )
        if scaled:
            self.spline.weight.data = self.spline.weight.data / kernel_size
        self.spline.weight.requires_grad = False
        
        # Regular 1x1 convolution
        self.conv1d = nn.Conv1d(
            in_channels * degrees_of_freedom, out_channels, 1, 
            groups=groups, bias=bias
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass through B-spline convolution.
        
        Args:
            input: Input tensor of shape (batch_size, in_channels, length)
            
        Returns:
            Output tensor after B-spline convolution
        """
        batch_size, n_channels, length = input.size()
        
        # Apply B-spline transformation
        spline_out = self.spline(
            input.view(batch_size * n_channels, 1, length)
        )
        
        # Apply 1x1 convolution
        conv1d_out = self.conv1d(
            spline_out.view(batch_size, n_channels * self._df, length)
        )
        return conv1d_out


class Sei(nn.Module):
    """SEI (Sequence-based Ensemble Interpretation) model architecture.
    
    This model predicts chromatin accessibility and transcription factor binding
    from DNA sequences using a deep convolutional architecture with residual
    connections and dilated convolutions.
    
    Args:
        sequence_length: Length of input DNA sequences
        n_genomic_features: Number of genomic features to predict
    """
    
    def __init__(self, sequence_length: int = 4096, n_genomic_features: int = 21907):
        super(Sei, self).__init__()
        
        self.sequence_length = sequence_length
        self.n_genomic_features = n_genomic_features

        # First convolutional block
        self.lconv1 = nn.Sequential(
            nn.Conv1d(4, 480, kernel_size=9, padding=4),
            nn.Conv1d(480, 480, kernel_size=9, padding=4)
        )
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(480, 480, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(480, 480, kernel_size=9, padding=4),
            nn.ReLU(inplace=True)
        )
        
        # Second convolutional block
        self.lconv2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
            nn.Conv1d(480, 640, kernel_size=9, padding=4),
            nn.Conv1d(640, 640, kernel_size=9, padding=4)
        )
        
        self.conv2 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv1d(640, 640, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(640, 640, kernel_size=9, padding=4),
            nn.ReLU(inplace=True)
        )
        
        # Third convolutional block
        self.lconv3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
            nn.Conv1d(640, 960, kernel_size=9, padding=4),
            nn.Conv1d(960, 960, kernel_size=9, padding=4)
        )
        
        self.conv3 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv1d(960, 960, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(960, 960, kernel_size=9, padding=4),
            nn.ReLU(inplace=True)
        )

        # Dilated convolutional blocks for capturing long-range dependencies
        self.dconv1 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(960, 960, kernel_size=5, dilation=2, padding=4),
            nn.ReLU(inplace=True)
        )
        
        self.dconv2 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(960, 960, kernel_size=5, dilation=4, padding=8),
            nn.ReLU(inplace=True)
        )
        
        self.dconv3 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(960, 960, kernel_size=5, dilation=8, padding=16),
            nn.ReLU(inplace=True)
        )
        
        self.dconv4 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(960, 960, kernel_size=5, dilation=16, padding=32),
            nn.ReLU(inplace=True)
        )
        
        self.dconv5 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(960, 960, kernel_size=5, dilation=25, padding=50),
            nn.ReLU(inplace=True)
        )

        # B-spline transformation for spatial pooling
        self._spline_df = int(128/8)
        self.spline_tr = nn.Sequential(
            nn.Dropout(p=0.5),
            BSplineTransformation(self._spline_df, scaled=False)
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(960 * self._spline_df, n_genomic_features),
            nn.ReLU(inplace=True),
            nn.Linear(n_genomic_features, n_genomic_features),
            nn.Sigmoid()
        )



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the SEI model.
        
        Args:
            x: Input tensor of shape (batch_size, 4, sequence_length)
               representing one-hot encoded DNA sequences
               
        Returns:
            Predicted probabilities for genomic features
        """
        # First convolutional block with residual connection
        lout1 = self.lconv1(x)
        out1 = self.conv1(lout1)
        
        # Second convolutional block with residual connection
        lout2 = self.lconv2(out1 + lout1)
        out2 = self.conv2(lout2)
        
        # Third convolutional block with residual connection
        lout3 = self.lconv3(out2 + lout2)
        out3 = self.conv3(lout3)
        
        # Dilated convolutions with progressive residual connections
        dconv_out1 = self.dconv1(out3 + lout3)
        cat_out1 = out3 + dconv_out1
        
        dconv_out2 = self.dconv2(cat_out1)
        cat_out2 = cat_out1 + dconv_out2
        
        dconv_out3 = self.dconv3(cat_out2)
        cat_out3 = cat_out2 + dconv_out3
        
        dconv_out4 = self.dconv4(cat_out3)
        cat_out4 = cat_out3 + dconv_out4
        
        dconv_out5 = self.dconv5(cat_out4)
        out = cat_out4 + dconv_out5
        
        # B-spline spatial transformation
        spline_out = self.spline_tr(out)
        reshape_out = spline_out.view(spline_out.size(0), 960 * self._spline_df)
        
        # Final classification
        predictions = self.classifier(reshape_out)
        return predictions


class NonStrandSpecific(nn.Module):
    """
    A torch.nn.Module that wraps a user-specified model architecture if the
    architecture does not need to account for sequence strand-specificity.

    Parameters
    ----------
    model : torch.nn.Module
        The user-specified model architecture.
    mode : {'mean', 'max'}, optional
        Default is 'mean'. NonStrandSpecific will pass the input and the
        reverse-complement of the input into `model`. The mode specifies
        whether we should output the mean or max of the predictions as
        the non-strand specific prediction.

    Attributes
    ----------
    model : torch.nn.Module
        The user-specified model architecture.
    mode : {'mean', 'max'}
        How to handle outputting a non-strand specific prediction.

    """

    def __init__(self, model, mode="mean"):
        super(NonStrandSpecific, self).__init__()

        self.model = model

        if mode != "mean" and mode != "max":
            raise ValueError("Mode should be one of 'mean' or 'max' but was"
                             "{0}.".format(mode))
        self.mode = mode

    def forward(self, input):
        reverse_input = _flip(_flip(input, 1), 2)

        output = self.model.forward(input)
        output_from_rev = self.model.forward(reverse_input)
        if type(output) == tuple:
            output1, output2 = output
            output_from_rev1, output_from_rev2 = output_from_rev
            if self.mode == "mean":
                return ((output1 + output_from_rev1) / 2,
                        (output2 + output_from_rev2) / 2)
            else:
                return (torch.max(output1, output_from_rev1),
                        torch.max(output2, output_from_rev2))
        else:
            if self.mode == "mean":
                return (output + output_from_rev) / 2
            else:
                return torch.max(output, output_from_rev)


# =============================================================================
# Loss Function and Optimizer
# =============================================================================


def criterion() -> nn.BCELoss:
    """Get the loss function for SEI model training.
    
    Returns:
        Binary cross-entropy loss for multi-label classification
    """
    return nn.BCELoss()


def get_optimizer(lr: float) -> Tuple[type, dict]:
    """Get optimizer class and parameters for SEI model training.
    
    Args:
        lr: Learning rate
        
    Returns:
        Tuple of (optimizer_class, optimizer_params)
    """
    return (torch.optim.SGD, {
        "lr": lr, 
        "weight_decay": 1e-7, 
        "momentum": 0.9
    })


# =============================================================================
# Factory Functions
# =============================================================================


def create_sei_model(sequence_length: int = 4096, 
                     n_genomic_features: int = 21907) -> Sei:
    """Factory function to create a SEI model.
    
    Args:
        sequence_length: Length of input DNA sequences
        n_genomic_features: Number of genomic features to predict
        
    Returns:
        Initialized SEI model
    """
    return Sei(sequence_length=sequence_length, n_genomic_features=n_genomic_features)
