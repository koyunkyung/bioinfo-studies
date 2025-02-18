import safe
from safe.converter import *
from safe.tokenizer import SAFETokenizer
import torch

tokenizer = SAFETokenizer.from_pretrained("datamol-io/safe-gpt")
smiles_list = ["CCC1(C2=C(COC1=O)C(=O)N3CC4=CC5=CC=CC=C5N=C4C3=C2)O",
              "CN(C)CC=CC(=O)NC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl)OC4CCOC4", 
              "CNC(=O)C1=CC=CC=C1SC2=CC3=C(C=C2)C(=NN3)C=CC4=CC=CC=N4"]

safe_strings =  [encode(smiles, slicer="mmpa") for smiles in smiles_list]
# print(safe_strings)

encoded = tokenizer.encode(safe_strings, ids_only=False)
print(encoded)