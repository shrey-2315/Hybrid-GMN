
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hybrid AST → Graph conversion
def build_graph_from_hybrid_ast(ast):
    node_list, edge_list = [], []

    def traverse(node, parent_id=None):
        node_id = len(node_list)
        node_list.append(node.label)
        if parent_id is not None:
            edge_list.append((parent_id, node_id))
        for child in node.children:
            traverse(child, node_id)

    traverse(ast.root)
    return node_list, edge_list

# GNN model for comparing two SQL AST graphs
class GMN(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(GMN, self).__init__()
        self.encoder = GCN(in_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, graph1, graph2):
        emb1 = self.encoder(graph1)
        emb2 = self.encoder(graph2)
        combined = torch.cat([emb1, emb2], dim=1)
        return self.classifier(combined)
