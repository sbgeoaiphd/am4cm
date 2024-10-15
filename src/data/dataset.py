import os
import torch
from torch.utils.data import Dataset
import geopandas as gpd
import numpy as np
from datetime import datetime
import json

class SITSDataset(Dataset):
    def __init__(self, pastis_path, fold_id=None):
        """
        Initialize the SITSDataset by loading metadata and filtering .npy file paths based on fold ID.

        Parameters
        ----------
        pastis_path : str
            Path to the PASTIS dataset directory where metadata.geojson is located file.
        fold_id : int or None, optional
            If provided, filter the dataset to only include files that match this fold ID.
            If None, all files are included.
            fold_id is an integer between 1 and 5, inclusive.
        """

        # Load metadata
        self.pastis_path = pastis_path
        metadata_path = os.path.join(pastis_path, 'metadata.geojson')
        self.metadata = gpd.read_file(metadata_path)
        self.metadata = self.filter_patch_ids(fold_id)
        
        self.patch_ids = list(self.metadata.ID_PATCH)
        self.init_patch_paths()
        

    def __len__(self):
        """
        Return the number of files.
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
        x, y = self.flatten_patch(x, y)
        return x, y
    
    def filter_patch_ids(self, fold_id):
        """
        Filter the patch IDs based on fold ID.
        
        Parameters
        ----------
        fold_id : int or None
            If provided, filter the dataset to only include files that match this fold ID.
            If None, all files are included.
            fold_id is an integer between 1 and 5, inclusive.
        """
        if fold_id is not None:
            self.metadata = self.metadata[self.metadata.Fold == fold_id]
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
    def flatten_patch(self, x, y):
        """
        Flatten the features and labels for a single patch.
        
        Parameters
        ----------
        x : np.ndarray
            Features for the patch.
        y : np.ndarray
            Labels for the patch.
        """
        # Flatten the features and labels
        x_flat = x.reshape(x.shape[0], x.shape[1], -1)
        y_flat = y.flatten()

        # put sample dim first (for x), so (time, bands, samples) -> (samples, time, bands)
        x_flat = np.moveaxis(x_flat, 2, 0)

        return x_flat, y_flat