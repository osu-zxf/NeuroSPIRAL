import numpy as np
import torch
from tqdm import tqdm
from rdkit import Chem
from .molecule_process import mol_to_multi_level_graph
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data, DataLoader, Batch

def prepare_train_val_test_set(filtered_dataset, batch_size=32):
    train_size = int(0.8 * len(filtered_dataset))
    val_size = int(0.1 * len(filtered_dataset))
    test_size = len(filtered_dataset) - train_size - val_size

    # Shuffle and split
    np.random.shuffle(filtered_dataset)
    train_dataset_raw = filtered_dataset[:train_size]
    val_dataset_raw = filtered_dataset[train_size:(train_size + val_size)]
    test_dataset_raw = filtered_dataset[(train_size + val_size):]
    print(f"Raw splits: Train: {len(train_dataset_raw)}, Val: {len(val_dataset_raw)}, Test: {len(test_dataset_raw)}")
    print("Constructing multi-level graphs for the dataset...")

    processed_dataset = []
    all_global_features = []

    for original_data_item in tqdm(filtered_dataset, desc="Processing molecules"):
        mol = Chem.MolFromSmiles(original_data_item.smiles)
        if mol is None:
            print(f"Warning: Could not parse SMILES: {original_data_item.smiles}")
            continue
        
        multi_level_data = mol_to_multi_level_graph(mol, original_data_item)
        if multi_level_data is not None:
            processed_dataset.append(multi_level_data)
            all_global_features.append(multi_level_data.global_features.numpy())

    if len(all_global_features) > 0:
        global_features_array = np.array(all_global_features)
        scaler = MinMaxScaler()
        scaled_global_features_array = scaler.fit_transform(global_features_array)
        
        for i, data_item in enumerate(processed_dataset):
            data_item.global_features = torch.tensor(scaled_global_features_array[i], dtype=torch.float)
        print("Global features normalized using MinMaxScaler.")
    else:
        print("No global features to normalize.")

    train_dataset_processed = []
    val_dataset_processed = []
    test_dataset_processed = []

    train_smiles = {d.smiles for d in train_dataset_raw}
    val_smiles = {d.smiles for d in val_dataset_raw}
    test_smiles = {d.smiles for d in test_dataset_raw}

    for data in processed_dataset:
        if data.smiles in train_smiles:
            train_dataset_processed.append(data)
        elif data.smiles in val_smiles:
            val_dataset_processed.append(data)
        elif data.smiles in test_smiles:
            test_dataset_processed.append(data)

    train_loader = DataLoader(train_dataset_processed, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset_processed, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset_processed, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader