import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Simulated Time Series Sensor Data
class SensorDataGenerator:
    @staticmethod
    def generate_time_series(n_samples=1000, n_timesteps=100, n_features=10):
        # Generate normal data
        normal_data = np.random.normal(0, 1, (n_samples, n_timesteps, n_features))
        
        # Generate augmented versions of data
        def augment_data(data):
            # Time warping
            augmented = data.copy()
            for i in range(len(data)):
                # Random time warping
                warp_indices = np.sort(np.random.choice(
                    np.arange(n_timesteps), 
                    size=int(n_timesteps * 0.2), 
                    replace=False
                ))
                augmented[i, warp_indices] *= np.random.uniform(0.8, 1.2)
            return augmented
        
        return normal_data, augment_data(normal_data)

# Contrastive Learning Model
class ContrastiveSensorModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, embedding_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(self, x):
        return nn.functional.normalize(self.encoder(x), dim=1)
    
    def contrastive_loss(self, z1, z2, temperature=0.5):
        # NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss
        sim_matrix = torch.mm(z1, z2.t()) / temperature
        labels = torch.arange(len(z1)).to(z1.device)
        
        loss_1 = nn.functional.cross_entropy(sim_matrix, labels)
        loss_2 = nn.functional.cross_entropy(sim_matrix.t(), labels)
        
        return (loss_1 + loss_2) / 2

# Training Process
def train_contrastive_model(X1, X2, epochs=100, batch_size=64):
    # Flatten time series for processing
    X1_flat = X1.reshape(X1.shape[0], -1)
    X2_flat = X2.reshape(X2.shape[0], -1)
    
    # Preprocessing
    scaler = StandardScaler()
    X1_scaled = scaler.fit_transform(X1_flat)
    X2_scaled = scaler.transform(X2_flat)
    
    # Split data
    X1_train, X1_val, X2_train, X2_val = train_test_split(
        X1_scaled, X2_scaled, test_size=0.2, random_state=42
    )
    
    # Convert to PyTorch tensors
    X1_train = torch.FloatTensor(X1_train)
    X2_train = torch.FloatTensor(X2_train)
    
    # Model and optimizer
    model = ContrastiveSensorModel(input_dim=X1_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    losses = []
    for epoch in range(epochs):
        # Shuffle data
        indices = torch.randperm(len(X1_train))
        X1_batch = X1_train[indices[:batch_size]]
        X2_batch = X2_train[indices[:batch_size]]
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        z1 = model(X1_batch)
        z2 = model(X2_batch)
        
        # Compute contrastive loss
        loss = model.contrastive_loss(z1, z2)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Record loss
        losses.append(loss.item())
        
        # Print progress
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
    
    return model, losses

# Linear Probing for Downstream Task
def linear_probe(embeddings, labels):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42
    )
    
    # Train logistic regression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Visualization
def visualize_embeddings(model, X):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    
    # Get embeddings
    X_flat = X.reshape(X.shape[0], -1)
    X_scaled = StandardScaler().fit_transform(X_flat)
    X_tensor = torch.FloatTensor(X_scaled)
    
    with torch.no_grad():
        embeddings = model(X_tensor).numpy()
    
    # Reduce to 2D
    tsne = TSNE(n_components=2)
    embedding_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1])
    plt.title('Sensor Embeddings Visualization')
    plt.show()

# Main execution
def main():
    # Generate sensor data
    X, X_aug = SensorDataGenerator.generate_time_series()
    
    # Create synthetic labels for linear probing
    labels = np.random.randint(0, 2, size=len(X))
    
    # Train contrastive model
    model, losses = train_contrastive_model(X, X_aug)
    
    # Get embeddings
    X_flat = X.reshape(X.shape[0], -1)
    X_scaled = StandardScaler().fit_transform(X_flat)
    X_tensor = torch.FloatTensor(X_scaled)
    
    with torch.no_grad():
        embeddings = model(X_tensor).numpy()
    
    # Linear probing
    probe_accuracy = linear_probe(embeddings, labels)
    print(f'Linear Probe Accuracy: {probe_accuracy:.4f}')
    
    # Visualize embeddings
    visualize_embeddings(model, X)

if __name__ == "__main__":
    main()