import os
import torch
from torch.utils.data import Dataset
import geopandas as gpd
import numpy as np
from datetime import datetime
import json
from torch.nn.utils.rnn import pad_sequence

class SITSDataset(Dataset):
    def __init__(self, pastis_path, fold_id=None, patch_frac=1, pixel_frac=1):
        """
        Initialize the SITSDataset by loading metadata and filtering .npy file paths based on fold ID.

        Parameters
        ----------
        pastis_path : str
            Path to the PASTIS dataset directory where metadata.geojson is located file.
        fold_id : int or list[int] or None, optional
            If provided, filter the dataset to only include files that match this fold ID or list of fold IDs.
            If None, all files are included.
            fold_id is an integer or a list of integers between 1 and 5, inclusive.
        patch_frac : float, optional
            Fraction of the patches in dataset to use (in % of patches). Default is 1 (use the entire dataset).
        pixel_frac : float, optional
            Fraction of the pixels in each patch to sample (in % of pixels). Default is 1 (use all pixels
        """

        # Load metadata
        self.pastis_path = pastis_path
        metadata_path = os.path.join(pastis_path, 'metadata.geojson')
        self.metadata = gpd.read_file(metadata_path)
        self.metadata = self.filter_patch_ids(fold_id, patch_frac)
        self.pixel_frac = pixel_frac
        
        self.patch_ids = list(self.metadata.ID_PATCH)
        self.init_patch_paths()

    def pixel_length(self):
        """
        Return total number of pixels in dataset.
        """
        # its the number of patches * number of pixels in each patch
        # patches are 128x128
        # we also sample with pixel_frac
        return int(len(self.patch_ids) * 128 * 128 * self.pixel_frac) - 1    

    def __len__(self):
        """
        Return n patches in dataset
        """
        return len(self.patch_ids)

    def __getitem__(self, idx):
        """
        Return the features and labels for the patch at the given index.
        
        Parameters
        ----------
        idx : int
            Index of the patch to load.
        """
        x, y = self.load_patch(idx)
        x, y = self.flatten_patch(x, y, self.pixel_frac)
        return x, y
    
    def filter_patch_ids(self, fold_id, patch_frac):
        """
        Filter the patch IDs based on fold ID.
        
        Parameters
        ----------
        fold_id : int or list[int] or None
            If provided, filter the dataset to only include files that match this fold ID or list of fold IDs.
            If None, all files are included.
            fold_id is an integer or a list of integers between 1 and 5, inclusive.
        patch_frac : float
            Fraction of the patches in dataset to use (in % of patches).
        """
        if fold_id is not None:
            if isinstance(fold_id, int):
                fold_id = [fold_id]
            self.metadata = self.metadata[self.metadata.Fold.isin(fold_id)]
            if patch_frac < 1:
                self.metadata = self.metadata.sample(frac=patch_frac, random_state=42)

        return self.metadata
        

    def init_patch_paths(self):
        """
        Initialize the list of .npy file paths.
        Both features (DATA_S2) and labels (ANNOTATIONS/TARGETS)
        """
        #self.npy_file_paths = [os.path.join(self.pastis_path, 'DATA_S2', f'S2_{patch_id}.npy') pfor patch_id in self.patch_ids]
        # tuple of (features, labels)
        self.npy_file_paths = [(os.path.join(self.pastis_path, 'DATA_S2', f'S2_{patch_id}.npy'),
                                os.path.join(self.pastis_path, 'ANNOTATIONS', f'TARGET_{patch_id}.npy')) for patch_id in self.patch_ids]

    
    # method to load a single patch - both X and y
    def load_patch(self, idx):
        """
        Load the features and labels for a single patch.
        
        Parameters
        ----------
        idx : int
            Index of the patch to load.
        """
        # Load the features and labels for the patch at the given index
        features_path, labels_path = self.npy_file_paths[idx]
        features = np.load(features_path)
        labels = np.load(labels_path)[0,...] # 1st channel are actual labels

        # convert to decimals
        # divide by 10000, convert to float32
        features = (features / 10000.0).astype(np.float32)

        # Add dates to features
        features = self.add_dates(features, idx)

        return features, labels
    
    def add_dates(self, x, idx, origin='2018-09-01'):
        """
        Add the date information to the features.
        
        Parameters
        ----------
        x : np.ndarray
            Features for the patch.
        idx : int
            Index of the patch to load - to get dates from metadata
        """
        # get date list from metadata - "dates-S2" column
        dates = json.loads(self.metadata.iloc[idx]['dates-S2'])
        dates = list(dates.values())

        # convert dates to datetime
        dates = [datetime.strptime(str(date), '%Y%m%d') for date in dates]
        # calculate days since 2018-09-01
        days_since_start = [(date - datetime.strptime(origin, '%Y-%m-%d')).days for date in dates]

        # add as a new channel to x in position 0
        days_since_start = np.array(days_since_start)
        n_days = len(days_since_start)

        # reshape and repeat to match x shape
        days_since_start = days_since_start.reshape(n_days, 1, 1, 1)
        days_since_start = np.repeat(days_since_start, x.shape[2], axis=2)
        days_since_start = np.repeat(days_since_start, x.shape[3], axis=3)

        # concatenate to x
        x = np.concatenate((days_since_start, x), axis=1)

        # cast to float32
        x = x.astype(np.float32)

        return x

    # method to flatten a loaded patch
    def flatten_patch(self, x, y, pixel_frac):
        """
        Flatten the features and labels for a single patch.
        
        Parameters
        ----------
        x : np.ndarray
            Features for the patch.
        y : np.ndarray
            Labels for the patch.
        pixel_frac : float
            Fraction of the pixels in the patch to sample (in % of pixels).
        """
        # Flatten the features and labels
        x_flat = x.reshape(x.shape[0], x.shape[1], -1)
        y_flat = y.flatten()

        # put sample dim first (for x), so (time, bands, samples) -> (samples, time, bands)
        x_flat = np.moveaxis(x_flat, 2, 0)

        # Sample a fraction of the pixels
        if pixel_frac < 1:
            n_pixels = x_flat.shape[0]
            n_sampled_pixels = int(n_pixels * pixel_frac)
            indices = np.random.choice(n_pixels, n_sampled_pixels, replace=False)
            x_flat = x_flat[indices]
            y_flat = y_flat[indices]

        return x_flat, y_flat
    
class MiniBatchSITSDataset(Dataset):
    def __init__(self, dataset, patches_per_group=4, shuffle=True):
        self.dataset = dataset
        self.patches_per_group = patches_per_group
        self.shuffle = shuffle

        self.indices = np.arange(len(self.dataset))
        self.group_index = 0  # To keep track of which group we're on
        self.pixel_buffer = None  # Buffer to hold shuffled pixels
        self.label_buffer = None  # Buffer to hold corresponding labels
        self.buffer_index = 0

        self.on_epoch_end()  # Initialize the indices and load the first group

    def on_epoch_end(self):
        """Reshuffle patch indices and reset the group index at the start of each epoch."""
        self.group_index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        self._load_next_group()

    def __len__(self):
        """ Return total number of samples/pixels in dataset. """
        return self.dataset.pixel_length()

    def _sample_time_dim(self, X, target_time_dim):
        """Randomly sample time steps to match the target time dimension."""
        time_dim = X.shape[1]
        if time_dim > target_time_dim:
            time_indices = np.random.choice(time_dim, target_time_dim, replace=False)
            X = X[:, time_indices, :]
        return X

    def _load_next_group(self):
        """Load the next group of patches and shuffle pixels."""
        if self.group_index * self.patches_per_group >= len(self.indices):
            # All groups have been processed
            self.pixel_buffer = None
            self.label_buffer = None
            return

        group_indices = self.indices[
            self.group_index * self.patches_per_group:
            (self.group_index + 1) * self.patches_per_group
        ]

        patches = []
        time_dims = []

        # Load patches and determine the shortest time dimension
        for idx in group_indices:
            X, y = self.dataset[idx]
            time_dims.append(X.shape[1])
            patches.append((X, y))

        target_time_dim = min(time_dims)

        # Adjust time dimensions and flatten
        X_list = []
        y_list = []
        for X, y in patches:
            X = self._sample_time_dim(X, target_time_dim)
            X_list.append(X)
            y_list.append(y)

        # Concatenate and shuffle
        X_all = np.concatenate(X_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)
        permutation = np.random.permutation(X_all.shape[0])
        self.pixel_buffer = torch.tensor(X_all[permutation]).float()
        self.label_buffer = torch.tensor(y_all[permutation]).long()
        self.buffer_index = 0

        self.group_index += 1

    def __getitem__(self, idx):
        """Return a single sample from the buffer."""
        if self.pixel_buffer is None or self.buffer_index >= len(self.pixel_buffer):
            # Buffer exhausted, load next group
            self._load_next_group()
            if self.pixel_buffer is None:
                # No more data
                raise IndexError("End of dataset")

        X = self.pixel_buffer[self.buffer_index]
        y = self.label_buffer[self.buffer_index]
        self.buffer_index += 1
        return X, y
    
def padded_collate_fn(batch):
    """
    Custom collate function that pads the sequences in the time dimension
    to match the longest sequence in the batch.
    
    batch: List of tuples (X, y), where X is the data sample and y is the label.
    """
    # Separate the batch into X (data) and y (labels)
    batch_X = [item[0] for item in batch]
    batch_y = [item[1] for item in batch]

    # Pad the sequences along the time dimension (dimension 1)
    batch_X_padded = pad_sequence(batch_X, batch_first=True)

    # Stack the labels (no need to pad as labels should all have the same shape)
    batch_y_stacked = torch.stack(batch_y, dim=0)

    return batch_X_padded, batch_y_stacked



