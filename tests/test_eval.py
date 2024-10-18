import pytest
import os
import torch
from torch.utils.data import DataLoader
from ..src.models.ltae import LTAE
from ..src.training.utils import validate_model
from ..src.data.dataset import SITSDataset, MiniBatchSITSDataset

@pytest.fixture
def pastis_data_path():
    # Expect an environment variable 'PASTIS_PATH' pointing to the dataset folder
    data_path = os.getenv('PASTIS_PATH')
    if not data_path:
        pytest.skip('PASTIS_PATH is not set')
    return data_path

@pytest.fixture
def ltae_model():
    model = LTAE(
    in_channels=10,  # Adjust according to your dataset
    n_head=8,
    d_k=16,
    mlp3=[256, 128],
    mlp4=[128, 64, 32],
    n_classes=20,
    dropout=0.1,
    d_model=256,
    T=1000
    )

    model.to("cuda")
    weights_file = "../.checkpoints/best_model_ltae.pth"
    if os.path.exists(weights_file):
        model.load_state_dict(torch.load(weights_file))
    return model

@pytest.fixture
def eval_dataset(pastis_data_path):
    dataset = SITSDataset(pastis_data_path, 5, patch_frac=0.1, pixel_frac=0.05)
    eval_dataset = MiniBatchSITSDataset(dataset, patches_per_group=32)
    return eval_dataset

@pytest.fixture
def eval_loader(eval_dataset):
    eval_loader = DataLoader(eval_dataset, batch_size=512, shuffle=False, num_workers=4)
    return eval_loader

def test_validate_model(ltae_model, eval_loader):
    validate_model(ltae_model, eval_loader, evaluate=True, evaluation_path="../evals/TEST_best_model_ltae_{}")

