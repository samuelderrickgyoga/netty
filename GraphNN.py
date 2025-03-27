import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Simulated Sensor Network Data
class SensorNetworkGenerator:
    @staticmethod
    def create_sensor_graph(n_nodes=100, n_features=10, n_classes=2):
        # Create graph
        G = nx.erdos_renyi_graph(n_nodes, 0.1)
        
        # Node features (sensor readings)
        node_features = np.random.randn(n_nodes, n_features)
        
        # Synthetic node labels based on graph structure
        node_labels = np.zeros(n_nodes, dtype=int)
        node_labels[list(nx.maximal_independent_set(G))] = 1
        
        return G, node_features, node_labels

# GNN Model
class SensorGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels=64, num_classes=2):
        super(SensorGNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Training and Evaluation
def train_gnn(graph_data):
    # Prepare data
    x = torch.FloatTensor(graph_data.node_features)
    y = torch.LongTensor(graph_data.node_labels)
    
    # Create edge index
    edge_index = torch.tensor(list(graph_data.G.edges())).t().contiguous()
    
    # Split data
    train_mask = torch.zeros(x.size(0), dtype=torch.bool)
    test_mask = torch.zeros(x.size(0), dtype=torch.bool)
    
    train_indices, test_indices = train_test_split(
        np.arange(x.size(0)), test_size=0.2, random_state=42
    )
    
    train_mask[train_indices] = True
    test_mask[test_indices] = True
    
    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, y=y, 
                train_mask=train_mask, test_mask=test_mask)
    
    # Model and optimizer
    model = SensorGNN(num_features=x.size(1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        # Evaluation
        model.eval()
        pred = out.argmax(dim=1)
        correct = pred[data.test_mask] == data.y[data.test_mask]
        acc = int(correct.sum()) / int(data.test_mask.sum())
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}, Test Accuracy: {acc:.4f}')
    
    return model, data, acc

# Visualization of graph learning
def visualize_embeddings(model, data):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    
    model.eval()
    with torch.no_grad():
        z = model.conv1(data.x, data.edge_index)
        z = TSNE(n_components=2).fit_transform(z.numpy())
        
        plt.figure(figsize=(10, 8))
        plt.scatter(z[:, 0], z[:, 1], c=data.y, cmap='viridis')
        plt.title('Sensor Node Embeddings')
        plt.colorbar()
        plt.show()

# Main execution
def main():
    # Generate sensor network
    G, node_features, node_labels = SensorNetworkGenerator.create_sensor_graph()
    
    # Create graph data object
    class GraphData:
        def __init__(self, G, node_features, node_labels):
            self.G = G
            self.node_features = node_features
            self.node_labels = node_labels
    
    graph_data = GraphData(G, node_features, node_labels)
    
    # Train GNN
    model, data, accuracy = train_gnn(graph_data)
    
    print(f"Final Test Accuracy: {accuracy:.4f}")
    
    # Optional: Visualize embeddings
    visualize_embeddings(model, data)

if __name__ == "__main__":
    main()