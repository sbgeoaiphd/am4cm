import os
import pytest
import torch
from torch.utils.data import DataLoader
import geopandas as gpd
import numpy as np
from ..src.data.dataset import SITSDataset, MiniBatchSITSDataset, padded_collate_fn

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

def test_single_sample_loading(mock_sits_dataset):
    """
    Ensure that MiniBatchSITSDataset loads one sample at a time from the buffer.
    """
    dataset = MiniBatchSITSDataset(mock_sits_dataset, patches_per_group=2)
    
    # Call __getitem__ to load one sample
    first_sample_X, first_sample_y = dataset[0]
    
    # Ensure only one sample is returned
    assert first_sample_X.ndim == 2, "Expected a single sample (1D), but got a batch"
    assert first_sample_y.ndim == 0, "Expected a single label, but got a batch"
    
    # Ensure buffer index increments after each call
    second_sample_X, second_sample_y = dataset[1]
    assert dataset.buffer_index == 2, "Buffer index did not increment after loading a sample"
    
    # Ensure that two consecutive samples are different (since we are shuffling)
    assert not torch.equal(first_sample_X, second_sample_X), "Two consecutive samples are identical"

def test_buffer_replenishment(mock_sits_dataset):
    """
    Ensure that the buffer replenishes when exhausted and moves to the next group of patches.
    """
    dataset = MiniBatchSITSDataset(mock_sits_dataset, patches_per_group=2)
    
    # Simulate consuming the entire buffer
    while dataset.buffer_index < len(dataset.pixel_buffer):
        _ = dataset[dataset.buffer_index]
    
    # After consuming the buffer, it should reload the next group
    current_group_index = dataset.group_index
    _ = dataset[0]  # This should trigger buffer replenishment
    assert dataset.group_index == current_group_index + 1, "Group index did not increment"
    assert dataset.buffer_index == 1, "Buffer index was not reset after replenishment"

def test_epoch_end_shuffling(mock_sits_dataset):
    """
    Ensure that patch indices are reshuffled after each epoch.
    """
    dataset = MiniBatchSITSDataset(mock_sits_dataset, patches_per_group=2)
    
    # Store initial indices for comparison
    initial_indices = dataset.indices.copy()
    
    # Call on_epoch_end to trigger reshuffling
    dataset.on_epoch_end()
    
    # Check that the indices are different
    assert not np.array_equal(initial_indices, dataset.indices), "Patch indices were not reshuffled"
    
    # Ensure the buffer is reset and the group index starts over
    assert dataset.group_index == 1, "Group index did not reset after epoch end"
    assert dataset.buffer_index == 0, "Buffer index did not reset after epoch end"

def test_dataloader_integration(mock_sits_dataset):
    """
    Ensure that DataLoader works with MiniBatchSITSDataset and handles batches correctly.
    """
    dataset = MiniBatchSITSDataset(mock_sits_dataset, patches_per_group=2)
    
    # Create a DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=padded_collate_fn,
        drop_last=True
    )
    
    for batch_X, batch_y in dataloader:
        # Ensure the batch size is correct
        assert batch_X.shape[0] == 32, "Batch size is incorrect"
        assert batch_y.shape[0] == 32, "Batch size is incorrect for labels"
        
        # Ensure individual samples are in the correct shape
        assert batch_X.ndim == 3, "Expected 2D tensor (batch_size, features)"
        assert batch_y.ndim == 1, "Expected 1D tensor for labels"
        
        # Break after one batch to avoid processing the whole dataset
        break

def test_incomplete_batch_handling(mock_sits_dataset):
    """
    Ensure that incomplete batches in the buffer are correctly dropped or handled by the DataLoader.
    """
    dataset = MiniBatchSITSDataset(mock_sits_dataset, patches_per_group=2)
    
    # Simulate the buffer almost being exhausted (e.g., one sample left)
    dataset.buffer_index = len(dataset.pixel_buffer) - 1
    
    # Create a DataLoader and ensure incomplete batches are dropped
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=padded_collate_fn,
        drop_last=True
    )
    
    for batch_X, batch_y in dataloader:
        # Ensure no incomplete batches are processed (should have 32 samples)
        assert batch_X.shape[0] == 32, "Incomplete batch was not dropped"
        break