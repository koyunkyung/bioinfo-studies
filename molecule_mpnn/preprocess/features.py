import numpy as np

class Featurizer:
    def __init__(self, allowable_sets):
        self.dim = 0
        self.features_mapping = {}
        for k, s in allowable_sets.items():
            s = sorted(list(s))
            self.features_mapping[k] = dict(zip(s, range(self.dim, len(s) + self.dim)))
            self.dim += len(s)

    def encode(self, inputs):
        output = np.zeros((self.dim, ))
        for name_feature, feature_mapping in self.features_mapping.items():
            feature = getattr(self, name_feature)(inputs)
            if feature not in feature_mapping:
                continue
            output[feature_mapping[feature]] = 1.0
        return output

## 원자 (Atom) 특성 ##
class AtomFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)

    def symbol(self, atom):
        return atom.GetSymbol()
    
    def n_valence(self, atom):
        return atom.GetTotalValence()
    
    def n_hydrogens(self, atom):
        return atom.GetTotalNumHs()
    
    def hybridization(self, atom):
        return atom.GetHybridization().name.lower()
    

## 결합 (Bond) 특성 ##
class BondFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)
        self.dim += 1

    def encode(self, bond):
        output = np.zeros((self.dim, ))
        if bond is None:
            output[-1] = 1.0
            return output
        output = super().encode(bond)
        return output
    
    def bond_type(self, bond):
        return bond.GetBondType().name.lower()
    
    def conjugated(self, bond):
        return bond.GetIsConjugated()
    
atom_featurizer = AtomFeaturizer(
    allowable_sets={
        "symbol": {"B", "Br", "C", "Ca", "Cl", "F", "H", "I", "N", "Na", "O", "P", "S"},
        "n_valence": {0, 1, 2, 3, 4, 5, 6},
        "n_hydrogens": {0, 1, 2, 3, 4},
        "hybridization": {"s", "sp", "sp2", "sp3"},
    }
)

bond_featurizer = BondFeaturizer(
    allowable_sets={
        "bond_type": {"single", "double", "triple", "aromatic"},
        "conjugated": {True, False},
    }
)



## 테스트 케이스 ##
if __name__ == "__main__":
    from rdkit import Chem
    mol = Chem.MolFromSmiles("CO")

    print("=== Atom Featurizer Test ===")
    for atom in mol.GetAtoms():
        features = atom_featurizer.encode(atom)
        print(f"\nAtom {atom.GetSymbol()} features:")
        print(f"Feature dimension: {len(features)}")
        print(f"Number of active features: {sum(features)}")

        active_features = []
        for name, mapping in atom_featurizer.features_mapping.items():
            feature_value = getattr(atom_featurizer, name)(atom)
            if feature_value in mapping:
                active_features.append(f"{name}: {feature_value}")
        print("Active features:", active_features)

    print("\n=== Bond Featurizer Test ===")
    for bond in mol.GetBonds():
        features = bond_featurizer.encode(bond)
        print(f"\nBond between {bond.GetBeginAtom().GetSymbol()} and {bond.GetEndAtom().GetSymbol()} features:")
        print(f"Feature dimension: {len(features)}")
        print(f"Number of active features: {sum(features)}")

        active_features = []
        for name, mapping in bond_featurizer.features_mapping.items():
            feature_value = getattr(bond_featurizer, name)(bond)
            if feature_value in mapping:
                active_features.append(f"{name}: {feature_value}")
        print("Active features:", active_features)
