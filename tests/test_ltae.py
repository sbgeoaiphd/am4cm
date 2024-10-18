import torch
import pytest
from ..src.models.ltae import LTAE  # Replace with the actual path to your model

def test_ltae_model_with_decoder():
    """
    Test the LTAE model with the decoder, ensuring that it outputs the correct number of classes.
    """
    batch_size = 8
    seq_len = 24
    in_channels = 10  # Number of input features (excluding positions)
    n_classes = 20  # Number of output classes

    # Create random input tensor, with the first "band" being the positions
    x = torch.randn(batch_size, seq_len, in_channels + 1)  # (batch_size, seq_len, in_channels + position)

    # Initialize the model with the decoder part
    model = LTAE(
        in_channels=in_channels, 
        n_head=8, 
        d_k=16, 
        mlp3=[256, 128], 
        mlp4=[128, 64, 32], 
        n_classes=n_classes, 
        dropout=0.1, 
        d_model=256, 
        T=1000, 
        return_att=False
    )

    # Run the forward pass
    output = model(x)

    # Check that the output shape is correct (batch_size, n_classes)
    assert output.shape == (batch_size, n_classes), f"Expected output shape (batch_size, {n_classes}), got {output.shape}"

    # Optionally: Check that the output values are finite (no NaNs or Infs)
    assert torch.isfinite(output).all(), "Output contains non-finite values (NaN/Inf)"


# test above but with vanilla attention_type
def test_ltae_model_with_vanilla_attention():
    """
    Test the LTAE model with the vanilla attention, ensuring that it outputs the correct number of classes.
    """
    batch_size = 128
    seq_len = 24
    in_channels = 10  # Number of input features (excluding positions)
    n_classes = 20  # Number of output classes

    # Create random input tensor, with the first "band" being the positions
    x = torch.randn(batch_size, seq_len, in_channels + 1)  # (batch_size, seq_len, in_channels + position)

    # Initialize the model with the vanilla attention
    model = LTAE(
        in_channels=in_channels, 
        n_head=8, 
        d_k=16, 
        mlp3=[256, 128], 
        mlp4=[128, 64, 32], 
        n_classes=n_classes, 
        dropout=0.1, 
        d_model=256, 
        T=1000, 
        return_att=False,
        attention_type="vanilla"
    )

    # Run the forward pass
    output = model(x)

    # Check that the output shape is correct (batch_size, n_classes)
    assert output.shape == (batch_size, n_classes), f"Expected output shape (batch_size, {n_classes}), got {output.shape}"

    # Optionally: Check that the output values are finite (no NaNs or Infs)
    assert torch.isfinite(output).all(), "Output contains non-finite values (NaN/Inf)"