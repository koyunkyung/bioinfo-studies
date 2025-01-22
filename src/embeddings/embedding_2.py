from transformers import AutoTokenizer, AutoModel
from rdkit import Chem
import safe
import selfies as sf
import torch
import numpy as np
import tensorflow as tf
import keras
from keras import layers, Model
from graph_env import graph_from_smiles


### Cell line name ###
# scGPT를 이용한 cell embedding



### Drug name (각각의 약물에 해당되는 SMILES 정보 이용) ###

## 화학정보 임베딩 방식 통합 클래스 ##
class ChemicalEmbeddings:
    def __init__(self, safe_model_name="datamol-io/safe-gpt"):
        self.safe_embedding = SafeEmbeddingGPT(model_name=safe_model_name)
        self.selfies_embedding = SelfiesEmbeddingGPT()

    def get_safe_embeddings(self, smiles_list):
        return self.safe_embedding.embed(smiles_list)
    
    def get_selfies_embeddings(self, smiles_list):
        return self.selfies_embedding.embed(smiles_list)
    
# safe 언어 모델을 활용한 임베딩
class SafeEmbeddingGPT:
    def __init__(self, lm_model_name="datamol-io/safe-gpt"):
        self.tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
        self.model = AutoModel.from_pretrained(lm_model_name)

    def safe(self, smiles_list):
        # 유효성 검사 후에 safe로 변환
        safe_list = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    print(f"Invalid SMILES: {smiles}")
                    continue
                safe_encoded = safe.encode(smiles)
                safe_list.append(safe_encoded)
            except safe._exception.SAFEFragmentationError as e:
                print(f"SAFE encoding failed for SMILES: {smiles} with error: {e}")
                continue
        if not safe_list:
            raise ValueError("No valid SAFE encodings generated from the provided SMILES")
        
        # safe 문자열 언어 모델 입력으로 바꿔주고 임베딩 추출
        inputs = self.tokenizer(safe_list, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state
    
# selfies 언어 모델을 활용한 임베딩 (알파벳, 정수로 매핑하지 말고 바로 one-hot encoding하기)
class SelfiesEmbeddingGPT:

    def __init__(self, lm_model_name):
        self.tokenizer = None
        self.model = None

    def selfies(self, smiles_list):
        selfies_list = []
        
## MPNN을 이용한 약물 임베딩 ##
# message passing phase 구현 - 나의 노드, 이웃노들들의 현재 상태와 그것을 연결하는 엣지들의 정보 aggregate
class EdgeNetwork(layers.Layer):
    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1]
        self.bond_dim = input_shape[1][-1]
        self.kernel = self.add_weight(
            shape=(self.bond_dim, self.atom_dim * self.atom_dim), 
            initializer="glorot_uniform",
            name="kernel",
        )
        self.bias = self.add_weight(
            shape=(self.atom_dim * self.atom_dim,),
            initializer="zeros",
            name="bias",
        )
        super().build(input_shape)

    def call(self, inputs):
        atom_features, bond_features, pair_indices = inputs
        bond_features_transformed = tf.matmul(bond_features, self.kernel) + self.bias
        bond_features_transformed = tf.reshape(
            bond_features_transformed, (-1, self.atom_dim, self.atom_dim)
        )

        # 이웃 노드의 atom features를 가져오고 aggregate하기
        atom_features_neighbors = tf.gather(atom_features, pair_indices[:, 1])
        atom_features_neighbors = tf.expand_dims(atom_features_neighbors, axis=-1)
        messages = tf.matmul(bond_features_transformed, atom_features_neighbors)
        messages = tf.squeeze(messages, axis=-1)

        aggregated_messages = tf.math.unsorted_segment_sum(
            messages,
            segment_ids=pair_indices[:, 0],
            num_segments=tf.shape(atom_features)[0],
        )
        return aggregated_messages
    
class MessagePassing(layers.Layer):
    def __init__(self, units, steps=4, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.steps = steps
        self.edge_network = None 

    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1]
        self.edge_network = EdgeNetwork() 
        self.pad_length = max(0, self.units - self.atom_dim)
        self.gru_cell = layers.GRUCell(self.units) 
        super().build(input_shape)

    def call(self, inputs):
        atom_features, bond_features, pair_indices = inputs
        atom_features = tf.pad(atom_features, [[0, 0], [0, self.pad_length]])

        # 여러번의 message passing 진행 (이웃들의 정보 받아오고, 노드 정보 업데이트 과정 모두 포함)
        for _ in range(self.steps):
            aggregated_messages = self.edge_network(
                [atom_features, bond_features, pair_indices]
            )
            atom_features, _ = self.gru_cell(aggregated_messages, [atom_features])

        return atom_features


# readout phase 구현 - message passing을 여러번 반복해서 각 노드에 대한 n번째 hidden state 얻기
class PartitionPadding(layers.Layer):
    def __init__(self, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def call(self, inputs):
        atom_features, molecule_indicator = inputs

        # subgraph 가져와서 하나하나 쌓아두기
        atom_features_partitioned = tf.dynamic_partition(
            atom_features, molecule_indicator, self.batch_size
        )
        num_atoms = [tf.shape(f)[0] for f in atom_features_partitioned]
        max_num_atoms = tf.reduce_max(num_atoms)
        atom_features_stacked = tf.stack(
            [
                tf.pad(f, [(0, max_num_atoms - n), (0, 0)])
                for f, n in zip(atom_features_partitioned, num_atoms)
            ],
            axis=0,
        )
        # 비어있는 subgraph는 지우기 
        gather_indices = tf.where(tf.reduce_sum(atom_features_stacked, (1, 2)) != 0)
        gather_indices = tf.squeeze(gather_indices, axis=-1)
        return tf.gather(atom_features_stacked, gather_indices, axis=0)


class TransformerEncoderReadout(layers.Layer):
    def __init__(
        self, num_heads=8, embed_dim=64, dense_dim=512, batch_size=32, **kwargs
    ):
        super().__init__(**kwargs)

        self.partition_padding = PartitionPadding(batch_size)
        self.attention = layers.MultiHeadAttention(num_heads, embed_dim)
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.average_pooling = layers.GlobalAveragePooling1D()

    def call(self, inputs):
        x = self.partition_padding(inputs)
        padding_mask = tf.reduce_any(tf.not_equal(x, 0.0), axis=-1)
        padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]
        attention_output = self.attention(x, x, attention_mask=padding_mask)
        proj_input = self.layernorm_1(x + attention_output)
        proj_output = self.layernorm_2(proj_input + self.dense_proj(proj_input))
        return self.average_pooling(proj_output)


# 통합해서 MPNN 모델 구현하기
def MPNNModel(
    atom_dim,
    bond_dim,
    batch_size=32,
    message_units=64,
    message_steps=4,
    num_attention_heads=8,
    dense_units=512,
):
    atom_features = layers.Input(shape=(atom_dim,), dtype="float32", name="atom_features")
    bond_features = layers.Input(shape=(bond_dim,), dtype="float32", name="bond_features")
    pair_indices = layers.Input(shape=(2,), dtype="int32", name="pair_indices")
    molecule_indicator = layers.Input(shape=(), dtype="int32", name="molecule_indicator")

    x = MessagePassing(message_units, message_steps)(
        [atom_features, bond_features, pair_indices]
    )

    x = TransformerEncoderReadout(
        num_attention_heads, message_units, dense_units, batch_size
    )([x, molecule_indicator])

    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(
        inputs=[atom_features, bond_features, pair_indices, molecule_indicator],
        outputs=[x],
    )
    return model

# 임베딩 만들기
def mpnn_embeddings(smiles_list, model, batch_size=32):
    atom_features, bond_features, pair_indices = graph_from_smiles(smiles_list)

    # 분자구조 받아올 리스트 만들기
    molecule_indicator = []
    for idx, atoms in enumerate(atom_features):
        molecule_indicator.extend([idx] * atoms.shape[0])
    molecule_indicator = tf.convert_to_tensor(molecule_indicator, dtype=tf.int32)

    inputs = {
        "atom_features": atom_features.to_tensor(),
        "bond_features": bond_features.to_tensor(),
        "pair_indices": pair_indices.to_tensor(),
        "molecule_indicator": molecule_indicator,
    }

    embeddings = model.predict(inputs, batch_size=batch_size)
    return embeddings
    



### 함수 작동 확인 테스트 케이스 ###
if __name__ == "__main__":

    combined_cell_line = ["Camptothecin:TOP1", "Vinblastine:Microtubule destabiliser", "Cisplatin:DNA crosslinker"]
    smiles_list = ["CCC1(C2=C(COC1=O)C(=O)N3CC4=CC5=CC=CC=C5N=C4C3=C2)O", "N.N.Cl[Pt]Cl", "CC1=C(C=C(C=N1)Cl)NCC2=CC=C(S2)C(=O)NC(CC3CCCC3)C(=O)NC4CC4"]

    ## scGPT를 사용한 cell line 임베딩 ##



    # ## SAFE-GPT를 사용한 약물 임베딩 ##
    # safe_gpt_embedding = SafeEmbeddingGPT(lm_model_name="datamol-io/safe-gpt")
    # embeddings = safe_gpt_embedding.safe(smiles_list)
    # print(f"SAFE-GPT Embeddings Shape: {embeddings.shape}")
    # print(f"SAFE-GPT Embeddings Tensor: \n{embeddings}")


    ## selfies-GPT를 사용한 약물 임베딩 ##



    ## MPNN을 사용한 약물 임베딩 ##
    atom_dim = 6
    bond_dim = 4 
    batch_size = 32
    message_units = 64
    message_steps = 4
    num_attention_heads = 8
    dense_units = 512

    mpnn_model = MPNNModel(atom_dim=atom_dim, bond_dim=bond_dim, batch_size=batch_size, message_units=message_units, message_steps=message_steps, num_attention_heads=num_attention_heads,dense_units=dense_units,)
    embeddings = mpnn_embeddings(smiles_list, mpnn_model, batch_size=batch_size)

    print(f"Generated Embeddings Shape: {embeddings.shape}")
    print(f"Generated Embeddings: \n{embeddings}")
        

            
        