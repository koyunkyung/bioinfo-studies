import tensorflow as tf
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import keras
from keras import layers

## 원자 특성: 원자 종류(C, N, O, ...)는 one-hot encoding, 수소 원자 수, 형식 전하, 방향족 여부 (고리구조 있는지 없는지) ##
def one_hot_encoding(atom, possible_atoms):
    encoding = [0] * len(possible_atoms)
    atom_symbol = atom.GetSymbol()
    if atom_symbol in possible_atoms:
        encoding[possible_atoms.index(atom_symbol)] = 1
    return encoding

def get_atom_features(mol, possible_atoms=None):
    if possible_atoms is None:
        possible_atoms = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'P', 'I']
    
    atom_features = []
    for atom in mol.GetAtoms():
        symbol_encoding = one_hot_encoding(atom, possible_atoms)
        
        # 추가 원자 특징: 수소 원자 수, 형식 전하, 방향족 여부
        features = [
            atom.GetTotalNumHs(),  
            atom.GetFormalCharge(),  
            1 if atom.GetIsAromatic() else 0,
        ]
        
        atom_features.append(symbol_encoding + features)
    
    return np.array(atom_features, dtype=np.float32)

## 분자 내 결함 (엣지) 정보: 결합 종류(단일결합, 이중결합, 삼중결합, 방향족 결합), 공액 (단일결합과 이중결합 번갈아 연결된 구조) 여부 ##
def get_bond_features(mol):
    bond_features = []
    pair_indices = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        bond_type_features = [
            1 if bond.GetBondType() == Chem.rdchem.BondType.SINGLE else 0,
            1 if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE else 0,
            1 if bond.GetBondType() == Chem.rdchem.BondType.TRIPLE else 0,
            1 if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC else 0,
        ]

        bond_features.append(bond_type_features + [
            1 if bond.GetIsConjugated() else 0,
            1 if bond.IsInRing() else 0
        ])
        
        pair_indices.append([i, j])
    
    return (np.array(bond_features, dtype=np.float32), 
            np.array(pair_indices, dtype=np.int32))

## SMILES 문자열 그래프로 표현하기 ##
def smiles_to_graph_data(smiles):
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    
    atom_features = get_atom_features(mol)
    bond_features, pair_indices = get_bond_features(mol)
    
    return atom_features, bond_features, pair_indices

class MoleculeEmbeddingModel:
    def __init__(self, atom_dim=12, bond_dim=6, hidden_dim=64):
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.hidden_dim = hidden_dim
        
        self.model = self._create_embedding_model()
    
    def _create_embedding_model(self):
        # 입력층
        node_features = layers.Input(shape=(None, self.atom_dim), name='node_features')
        bond_features = layers.Input(shape=(None, self.bond_dim), name='bond_features')
        pair_indices = layers.Input(shape=(None, 2), dtype='int32', name='pair_indices')
        
        # 메시지 패싱 레이어 - 노드 특성 변환하고 통합해서 최종으로는 분자 전체 임베딩 만들기
        x = layers.Dense(self.hidden_dim, activation='relu')(node_features)
        x = layers.GlobalAveragePooling1D()(x)

        embedding = layers.Dense(self.hidden_dim, activation='tanh', name='molecule_embedding')(x)

        model = keras.Model(
            inputs=[node_features, bond_features, pair_indices],
            outputs=embedding
        )
        
        return model
    
    def embed_molecules(self, smiles_list):
        graph_data = [smiles_to_graph_data(smiles) for smiles in smiles_list]
 
        node_lengths = [data[0].shape[0] for data in graph_data]
        bond_lengths = [data[1].shape[0] for data in graph_data]

        max_nodes = max(node_lengths)
        max_bonds = max(bond_lengths)

        node_features = np.zeros((len(smiles_list), max_nodes, self.atom_dim), dtype=np.float32)
        bond_features = np.zeros((len(smiles_list), max_bonds, self.bond_dim), dtype=np.float32)
        pair_indices = np.zeros((len(smiles_list), max_bonds, 2), dtype=np.int32)

        for i, (atoms, bonds, indices) in enumerate(graph_data):
            node_features[i, :len(atoms)] = atoms
            bond_features[i, :len(bonds)] = bonds
            pair_indices[i, :len(indices)] = indices

        embeddings = self.model.predict([node_features, bond_features, pair_indices])
        return embeddings


### 함수 제대로 작동하는지 확인하는 테스트 케이스 ###
if __name__ == "__main__":
    smiles_list = [
        "CCC1(C2=C(COC1=O)C(=O)N3CC4=CC5=CC=CC=C5N=C4C3=C2)O", 
        "CN(C)CC=CC(=O)NC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl)OC4CCOC4", 
        "CNC(=O)C1=CC=CC=C1SC2=CC3=C(C=C2)C(=NN3)C=CC4=CC=CC=N4"
    ]

    embedding_model = MoleculeEmbeddingModel()
    molecule_embeddings = embedding_model.embed_molecules(smiles_list)

    print("MPNN Embeddings Shape: ", molecule_embeddings.shape)
    print("\nMPNN Embeddings:")
    for i, embedding in enumerate(molecule_embeddings):
        print(f"Molecule {i+1} Embedding:", embedding)