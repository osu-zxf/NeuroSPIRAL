import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GraphConv, global_mean_pool
from torch_geometric.utils import scatter

class AtomicGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_attr_dim, heads=4):
        super(AtomicGNN, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels // heads, heads=heads, edge_dim=edge_attr_dim)
        self.conv2 = GATConv(hidden_channels, out_channels // heads, heads=heads, edge_dim=edge_attr_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return x

class SubstructureGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super(SubstructureGNN, self).__init__()
        self.heads = heads
        self.conv1 = GATConv(in_channels, hidden_channels // heads, heads=heads)
        self.conv2 = GATConv(hidden_channels, out_channels // heads, heads=heads)

    def forward(self, x_sub, edge_index_sub):
        x_sub = self.conv1(x_sub, edge_index_sub)
        x_sub = F.relu(x_sub)
        x_sub = F.dropout(x_sub, p=0.5, training=self.training)
        x_sub = self.conv2(x_sub, edge_index_sub)
        return x_sub

class FusionAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads=4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        self.atom_to_sub_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=False)
        self.sub_to_atom_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=False)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(4 * embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, atom_mol_embedding, sub_mol_embedding):
        atom_seq = atom_mol_embedding.unsqueeze(0)
        sub_seq = sub_mol_embedding.unsqueeze(0)

        attended_sub_from_atom, _ = self.atom_to_sub_attn(query=atom_seq, key=sub_seq, value=sub_seq)
        attended_sub_from_atom = attended_sub_from_atom.squeeze(0)

        attended_atom_from_sub, _ = self.sub_to_atom_attn(query=sub_seq, key=atom_seq, value=atom_seq)
        attended_atom_from_sub = attended_atom_from_sub.squeeze(0)

        combined_embeddings = torch.cat([
            atom_mol_embedding,
            attended_sub_from_atom,
            sub_mol_embedding,
            attended_atom_from_sub
        ], dim=1)

        fused_emb = self.fusion_mlp(combined_embeddings)
        fused_emb = self.norm(fused_emb)
        return fused_emb

class MultiLevelGNN(nn.Module):
    def __init__(self, atom_in_channels, sub_in_channels_original, hidden_channels, out_channels, num_tasks, edge_attr_dim, global_feature_dim, gnn_heads=4):
        super(MultiLevelGNN, self).__init__()
        self.out_channels = out_channels
        self.gnn_heads = gnn_heads

        self.atomic_gnn = AtomicGNN(atom_in_channels, hidden_channels, out_channels, edge_attr_dim, heads=self.gnn_heads)
        
        self.substructure_gnn = SubstructureGNN(sub_in_channels_original + out_channels, hidden_channels, out_channels, heads=self.gnn_heads)

        self.fusion_attention = FusionAttention(out_channels, num_heads=self.gnn_heads)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(out_channels + global_feature_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_channels, num_tasks)
        )

    def forward(self, data):
        x_atom_emb = self.atomic_gnn(data.x, data.edge_index, data.edge_attr) # (num_total_atoms, out_channels)
        atom_mol_embedding = global_mean_pool(x_atom_emb, data.batch)

        x_sub_initial = data.x_sub
        num_total_substructures_in_batch = x_sub_initial.shape[0]

        x_sub_dynamic_features = torch.zeros(num_total_substructures_in_batch,
                                             self.out_channels, # 匹配 x_atom_emb 的输出维度
                                             device=x_atom_emb.device)

        if data.edge_index_substructure_atom_map_indices.numel() > 0:
            atom_offsets = torch.cumsum(torch.cat([
                torch.tensor([0], device=data.num_atoms.device),
                data.num_atoms
            ]), dim=0)[:-1]
            sub_offsets = torch.cumsum(torch.cat([
                torch.tensor([0], device=data.num_substructures.device),
                data.num_substructures
            ]), dim=0)[:-1]

            if isinstance(data.num_sub_atom_connections, (list, tuple)):
                num_sub_atom_connections_tensor = torch.tensor(data.num_sub_atom_connections, device=data.x.device)
            else:
                num_sub_atom_connections_tensor = data.num_sub_atom_connections

            if num_sub_atom_connections_tensor.numel() > 0:
                if num_sub_atom_connections_tensor.ndim > 1:
                    num_sub_atom_connections_tensor = num_sub_atom_connections_tensor.flatten()
                
                offset_atom_indices = torch.repeat_interleave(atom_offsets, num_sub_atom_connections_tensor)
                offset_sub_indices = torch.repeat_interleave(sub_offsets, num_sub_atom_connections_tensor)

                shifted_substructure_atom_map_indices = data.edge_index_substructure_atom_map_indices.clone()
                shifted_substructure_atom_map_indices[1, :] -= offset_atom_indices
                shifted_substructure_atom_map_indices[1, :] += offset_sub_indices

                x_sub_dynamic_features_pooled = scatter(x_atom_emb[shifted_substructure_atom_map_indices[0]],
                                                        shifted_substructure_atom_map_indices[1],
                                                        dim=0,
                                                        dim_size=num_total_substructures_in_batch,
                                                        reduce='mean')
                x_sub_dynamic_features_pooled = torch.nan_to_num(x_sub_dynamic_features_pooled, nan=0.0)
                
                x_sub_dynamic_features = x_sub_dynamic_features_pooled
        
        x_sub_combined_features = torch.cat([x_sub_initial, x_sub_dynamic_features], dim=-1)

        sub_mol_embedding = torch.zeros(data.num_graphs, self.out_channels, device=data.x.device)

        if x_sub_combined_features.shape[0] > 0:
            atom_offsets_per_graph = torch.cumsum(data.num_atoms, dim=0) - data.num_atoms
            graph_indices_for_edges = torch.arange(data.num_graphs, device=data.num_sub_edges.device).repeat_interleave(data.num_sub_edges)
            atom_offsets_for_edges = atom_offsets_per_graph[graph_indices_for_edges]
            local_sub_edge_index = data.edge_index_sub - atom_offsets_for_edges
            
            substructure_offsets_per_graph = torch.cumsum(data.num_substructures, dim=0) - data.num_substructures
            substructure_offsets_for_edges = substructure_offsets_per_graph[graph_indices_for_edges]
            correctly_shifted_edge_index_sub = local_sub_edge_index + substructure_offsets_for_edges
            
            x_sub_emb = self.substructure_gnn(x_sub_combined_features, correctly_shifted_edge_index_sub)

            sub_batch_list = []
            for i in range(data.num_graphs):
                num_s = data.num_substructures[i].item()
                if num_s > 0:
                    sub_batch_list.extend([i] * num_s)
            
            if len(sub_batch_list) > 0:
                sub_batch = torch.tensor(sub_batch_list, dtype=torch.long, device=data.x_sub.device)
                pooled_sub_embeddings_present = global_mean_pool(x_sub_emb, sub_batch)
                unique_graph_ids_with_sub = torch.unique(sub_batch)
                sub_mol_embedding[unique_graph_ids_with_sub] = pooled_sub_embeddings_present

        fused_attention_embedding = self.fusion_attention(atom_mol_embedding, sub_mol_embedding)

        global_features_batch = data.global_features.view(data.num_graphs, -1)
        final_fused_embedding = torch.cat([fused_attention_embedding, global_features_batch], dim=1)
        
        out = self.fusion_mlp(final_fused_embedding)
        return out