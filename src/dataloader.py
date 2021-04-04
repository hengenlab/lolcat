from torch.utils.data import Dataset
import sklearn

from src.covariance import compute_cov, compute_edge_dist


class EdgeDistributionDataset(Dataset):
    def __init__(self, master_dataset, dataset_size=1000, scaler=sklearn.preprocessing.StandardScaler,
                 num_bins=10, **kwargs):
        self.master_dataset = master_dataset
        self.kwargs = kwargs
        self.dataset_size = dataset_size
        self.scaler = scaler
        self.num_bins = num_bins

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        X, y, m = self.master_dataset.sample(**self.kwargs)
        cov = compute_cov(X.T, scaler=self.scaler)
        edge_dist = compute_edge_dist(cov, num_bins=self.num_bins)
        return edge_dist, y
