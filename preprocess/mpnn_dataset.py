from features import *
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MPNNDataset(Dataset):
    def __init__(self, X, y):
        self.X = X      # (atom_features_list, bond_features_list, pair_indices_list)
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (self.X[0][idx], self.X[1][idx], self.X[2][idx]), self.y[idx]
    
def prepare_batch(batch):
    X_batch , y_batch = zip(*batch)
    atom_features_list, bond_features_list, pair_indices_list = zip(*X_batch)

    num_atoms = [atoms.shape[0] for atoms in atom_features_list]
    num_bonds = [bonds.shape[0] for bonds in bond_features_list]

    # 각 원자(atom)가 어떤 분자(molecule)에 속하는지 나타내기 위해 molecule_indicator 만들기
    # 각 분자별 원자 수가 [2,1]로 주어진다면 molecule_indicator 결과값은 [0,0,1]
    molecule_indices = torch.arange(len(num_atoms))
    molecule_indicator = torch.repeat_interleave(molecule_indices, torch.tensor(num_atoms))

    # 분자들의 원자 인덱스를 고유한 값으로 만들기 위해 increment
    increment = torch.cumsum(torch.tensor(num_atoms[:-1]), dim=0)
    if len(num_bonds) > 1:
        gather_indices = torch.repeat_interleave(molecule_indices[:-1],torch.tensor(num_bonds[1:]))
        increment_padded = torch.cat([
            torch.zeros(num_bonds[0], dtype=increment.dtype),
            torch.gather(increment, 0, gather_indices)
        ])
    else:
        increment_padded = torch.zeros(num_bonds[0], dtype=torch.long)

    atom_features = torch.cat(atom_features_list, dim=0)
    bond_features = torch.cat(bond_features_list, dim=0)

    pair_indices = torch.cat(pair_indices_list, dim=1)
    pair_indices = pair_indices + increment_padded.unsqueeze(1)

    y_batch = torch.tensor(y_batch)
    return (atom_features, bond_features, pair_indices, molecule_indicator), y_batch

def mpnn_loader(X, y, batch_size=32, shuffle=False):
    dataset = MPNNDataset(X, y)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=prepare_batch
    )
    return loader
