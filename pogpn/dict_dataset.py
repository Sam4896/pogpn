from typing import Dict

import torch
from torch.utils.data import Dataset, DataLoader


class DictDataset(Dataset):
    """Class to create dag_dataset that takes in dictionary as input."""

    def __init__(self, dataset: Dict[str, torch.Tensor]):
        """Initialise the dag_dataset."""
        self.dataset = dataset
        self._validate_dict_values_length()
        length = [len(value) for value in dataset.values()]
        if len(set(length)) != 1:
            raise ValueError("Dictionary values length must be equal.")
        self.length = length[0]

    def __len__(self) -> int:
        """Return the length of the dag_dataset."""
        return self.length

    def __getitem__(self, idx):
        """Return a sample from the dag_dataset for specified index."""
        return {key: self.dataset[key][idx] for key in self.dataset.keys()}

    def get_by_key(self, key):
        """Return a sample from the dag_dataset for a specified key."""
        return self.dataset[key]

    def _validate_dict_values_length(self):
        value_lengths = [value.shape[0] for value in self.dataset.values()]
        if len(set(value_lengths)) != 1:
            raise ValueError("Dictionary values length must be equal.")


class DictDataLoader(DataLoader):
    """Class to create dag_dataset that takes in dict dag_dataset as input."""

    dataset: DictDataset  # Add this type hint

    def __init__(self, dataset: DictDataset, *args, **kwargs):
        """Initialise the dataloader."""
        super().__init__(dataset, *args, **kwargs)


if __name__ == "__main__":
    data_dict = {
        "input1": torch.tensor([[1], [2], [3], [4]]),
        "input2": torch.tensor([[10], [20], [30], [40]]),
        "targets": torch.tensor([0.1, 0.2, 0.3, 0.4]),
    }

    dataset = DictDataset(data_dict)
    dataloader = DictDataLoader(dataset=dataset, batch_size=2, shuffle=True)

    print("== Iterating over DataLoader ==")
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}:")
        for key, value in batch.items():
            print(f"  {key}: {value.tolist()}")
