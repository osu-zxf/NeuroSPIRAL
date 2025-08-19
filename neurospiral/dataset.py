import torch
from torch_geometric.data import InMemoryDataset, Data
import pandas as pd
from rdkit import Chem

class TOXRICDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        df = pd.read_csv(self.raw_paths[0], dtype={'ID': str, 'SMILES': str})
        self.label_columns = [col for col in df.columns if col not in ['ID', 'SMILES']]

    @property
    def raw_file_names(self):
        return ['toxric_data.csv']
    
    @property
    def processed_file_names(self):
        return ['data.pt']

    @property
    def num_classes(self):
        return len(self.label_columns)
    
    def process(self):
        self.csv_path = self.root+'/raw/toxric_data.csv'
        self.data_df = pd.read_csv(self.csv_path, dtype={'ID': str, 'SMILES': str})
        self.data_df = self.data_df[self.data_df['SMILES'].notna()]
        self.label_columns = [col for col in self.data_df.columns if col not in ['ID', 'SMILES']]

        data_list = []
        for idx, row in self.data_df.iterrows():
            smiles = row['SMILES']
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"解析SMILES失败：{smiles}, 跳过")
                continue
            atom_features = []
            for atom in mol.GetAtoms():
                atom_features.append([atom.GetAtomicNum()])
            x = torch.tensor(atom_features, dtype=torch.float)
            y = torch.tensor([row[col] if pd.notnull(row[col]) else float('nan') for col in self.label_columns], dtype=torch.float).unsqueeze(0)
            data = Data(x=x, y=y, smiles=smiles, id=row['ID'])
            data_list.append(data)
        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])