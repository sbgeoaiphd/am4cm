import os
import pytest
import torch
from torch.utils.data import DataLoader
import geopandas as gpd
import numpy as np
from ..src.data.dataset import SITSDataset, MiniBatchSITSDataset

@pytest.fixture
def pastis_data_path():
    # Expect an environment variable 'PASTIS_PATH' pointing to the dataset folder
    data_path = os.getenv('PASTIS_PATH')
    if not data_path:
        pytest.skip('PASTIS_PATH is not set')
    return data_path

@pytest.fixture
def expected_npy_files(pastis_data_path):
    # This the expected first few elements of the npy file path list to be
    return [(os.path.join(pastis_data_path, 'DATA_S2', 'S2_10000.npy'), os.path.join(pastis_data_path, 'ANNOTATIONS', 'TARGET_10000.npy')),
            (os.path.join(pastis_data_path, 'DATA_S2', 'S2_10007.npy'), os.path.join(pastis_data_path, 'ANNOTATIONS', 'TARGET_10007.npy'))]


def test_init_dataset(pastis_data_path, expected_npy_files):
    # Test the initialization of the dataset with fold_id=1
    dataset = SITSDataset(pastis_data_path, fold_id=1)
    
    # Check that metadata is loaded (optional, for further validation)
    assert hasattr(dataset, 'metadata')
    assert isinstance(dataset.metadata, gpd.GeoDataFrame)

    # check has attribute patch_ids
    assert hasattr(dataset, 'patch_ids')

    # Check that the list of npy files is correct
    assert dataset.npy_file_paths[:len(expected_npy_files)] == expected_npy_files
    
    # Check that the length of the dataset is correct
    fold_1_length = 487
    assert len(dataset) == fold_1_length

# test load_patch method - return a tuple of (X, y)
def test_load_patch(pastis_data_path):
    dataset = SITSDataset(pastis_data_path, fold_id=1)

    patch = dataset.load_patch(0)
    x, y = patch

    # check that it returns a tuple
    assert isinstance(patch, tuple)

    # check that the first element of the tuple is a numpy array
    assert isinstance(x, np.ndarray)

    # check that the second element of the tuple is also a numpy array
    assert isinstance(y, np.ndarray)

    # check x is 4D (time, bands, x, y)
    assert x.ndim == 4

    # check that the 2nd dim of X is the correct shape (10 bands)
    assert x.shape[1] == 11

    # and 3rd and 4th dims are 128
    assert x.shape[2:] == (128, 128)

    # check y is 2D (x, y)
    assert y.ndim == 2

    # check that the shape of y is (128, 128)
    assert y.shape == (128, 128)



def test_getitem_dataset(pastis_data_path):
    # Test the __getitem__ method of the dataset
    dataset = SITSDataset(pastis_data_path, fold_id=1)

    x, y = dataset[0]
    
    # Check that the __getitem__ method returns a tuple
    assert isinstance(dataset[0], tuple)
    
    # Check that the first element of the tuple is a numpy array (X)
    assert isinstance(x, np.ndarray)
    
    # Check that the second element of the tuple also is a numpy array (y)
    assert isinstance(y, np.ndarray)

    # data will be flattened. input will be (samples, timesteps, bands), output will be (samples,) (with integer categories)
    # lets check the number of dims of each
    assert x.ndim == 3
    assert y.ndim == 1

    # check sample dim is same
    assert x.shape[0] == y.shape[0]

    # band dim is constant in the dataset
    assert x.shape[2] == 11 # 10 bands + doy band

    # num timesteps depends on the file - we can find out actual timeseries length of [0] of fold 1 and test
    assert x.shape[1] == 43

    # test that doy band looks right
    # first 2 values should be 24 and 29
    assert x[0, 0, 0] == 23
    assert x[0, 1, 0] == 28

    # check y is int and x is float
    assert y.dtype == np.uint8
    assert x.dtype == np.float32

    # features converted to decimals, so shouldn't be values > ca. 1
    # except in the doy band, where everything is > 1 and int-like (but stored as float)
    assert np.all(x[:, :, 1:] <= 2.5)
    assert np.all(x[:, :, 0] > 1)


### Test the MiniBatch loader
@pytest.fixture
def mock_sits_dataset(pastis_data_path):
    """
    Dataset for testing purposes.
    """
    return SITSDataset(pastis_data_path, fold_id=1)

def test_minibatch_loading(mock_sits_dataset):
    """
    Test that multiple patches are loaded at once, concatenated, and shuffled within each group.
    """
    mini_batch_dataset = MiniBatchSITSDataset(mock_sits_dataset, patches_per_group=2)
    
    # Get the first mini-batch (group of 2 patches)
    X, y = mini_batch_dataset[0]
    
    # Ensure that two patches worth of pixels are concatenated
    assert X.shape[0] == mock_sits_dataset[0][0].shape[0] + mock_sits_dataset[1][0].shape[0]
    assert y.shape[0] == mock_sits_dataset[0][1].shape[0] + mock_sits_dataset[1][1].shape[0]
    
    # Ensure X and y are converted to tensors
    assert isinstance(X, torch.Tensor)
    assert isinstance(y, torch.Tensor)

def test_pixel_shuffling(mock_sits_dataset):
    """
    Test that pixels are shuffled across patches within the group.
    Non trivial because of the time sampling
    """
    mini_batch_dataset = MiniBatchSITSDataset(mock_sits_dataset, patches_per_group=2)
    
    # Get the first mini-batch (group of 2 patches)
    X, y = mini_batch_dataset[0]
    
    # Get the indices of the patches that were selected for this mini-batch
    selected_indices = mini_batch_dataset.indices[:2]  # Since we're testing the first mini-batch

    # Get the original patches and determine which has the shortest time dimension
    patch_X_1, _ = mock_sits_dataset[selected_indices[0]]
    patch_X_2, _ = mock_sits_dataset[selected_indices[1]]

    # find which is shortest
    time_dim_1 = patch_X_1.shape[1]
    time_dim_2 = patch_X_2.shape[1]

    # If 1 is shortest, and shuffling DIDN'T happen, then the 1st half of mini_batch_dataset X should be the first patch
    if time_dim_1 < time_dim_2:
        assert not torch.allclose(X[:patch_X_1.shape[0]], torch.tensor(patch_X_1))
    # If 2 is shortest, and shuffling DIDN'T happen, then the 1st half of mini_batch_dataset X should be the second patch
    else:
        assert not torch.allclose(X[patch_X_2.shape[0]:], torch.tensor(patch_X_2))

def test_epoch_reshuffling(mock_sits_dataset):
    """
    Test that patch indices are reshuffled at the start of each epoch.
    """
    mini_batch_dataset = MiniBatchSITSDataset(mock_sits_dataset, patches_per_group=2)

    # Get the patch indices after the first epoch
    initial_indices = mini_batch_dataset.indices.copy()
    
    # Simulate a new epoch and reshuffle
    mini_batch_dataset.on_epoch_end()
    reshuffled_indices = mini_batch_dataset.indices
    
    # Ensure that the indices have been reshuffled
    assert not np.array_equal(initial_indices, reshuffled_indices)


### STATUS - need new minibatchsitsdataset class which properly handles batches and provides single samples to be compatible iwth data loader
#### BUT we could do batching ourselves - what does DataLoader provide in the case we're thinking ourselves about shuffling and batches?
# def test_dataloader_with_minibatch(mock_sits_dataset):
#     """
#     Test that DataLoader works correctly with MiniBatchSITSDataset.
#     """
#     mini_batch_dataset = MiniBatchSITSDataset(mock_sits_dataset, patches_per_group=2)
    
#     # Create a DataLoader for batching
#     dataloader = DataLoader(mini_batch_dataset, batch_size=32, shuffle=False, num_workers=0)
    
#     for batch_X, batch_y in dataloader:
#         # Ensure batch size is correct
#         assert batch_X.shape[0] == 32
#         assert batch_y.shape[0] == 32
#         # don't know time dim, but check band dim is correct and in final dim
#         assert batch_X.shape[2] == 11
#         break  # Only test one batch