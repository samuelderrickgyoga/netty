import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Simulated IoT sensor data generation
def generate_sensor_data(n_samples=1000, n_features=20):
    # Normal data
    normal_data = np.random.normal(0, 1, (n_samples, n_features))
    
    # Add some anomalies
    anomalies = np.random.normal(5, 2, (int(n_samples * 0.1), n_features))
    
    # Combine normal and anomalous data
    data = np.vstack([normal_data, anomalies])
    labels = np.concatenate([
        np.zeros(n_samples), 
        np.ones(int(n_samples * 0.1))
    ])
    
    return data, labels

# VAE Model
class VariationalAutoencoder:
    def __init__(self, input_dim, latent_dim=10):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.model = self._build_model()
    
    def _sampling(self, args):
        z_mean, z_log_var = args
        batch = tf.keras.backend.shape(z_mean)[0]
        epsilon = tf.keras.backend.random_normal(shape=(batch, self.latent_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def _build_model(self):
        # Encoder
        inputs = Input(shape=(self.input_dim,))
        x = Dense(256, activation='relu')(inputs)
        x = Dense(128, activation='relu')(x)
        
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)
        
        # Sampling layer
        z = Lambda(self._sampling, output_shape=(self.latent_dim,), 
                   name='z')([z_mean, z_log_var])
        
        # Decoder
        decoder_input = Dense(128, activation='relu')(z)
        decoder_output = Dense(self.input_dim, activation='sigmoid')(decoder_input)
        
        # Compile full VAE model
        vae = Model(inputs, decoder_output)
        
        # Custom loss
        reconstruction_loss = tf.keras.losses.binary_crossentropy(inputs, decoder_output)
        reconstruction_loss *= self.input_dim
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        
        vae.add_loss(tf.reduce_mean(reconstruction_loss + kl_loss))
        vae.compile(optimizer='adam')
        
        return vae
    
    def train(self, X_train, epochs=50, batch_size=64):
        return self.model.fit(
            X_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            shuffle=True, 
            validation_split=0.2
        )
    
    def detect_anomalies(self, X_test, threshold=2.0):
        # Reconstruction error as anomaly score
        reconstructed = self.model.predict(X_test)
        mse = np.mean(np.power(X_test - reconstructed, 2), axis=1)
        
        # Identify anomalies based on reconstruction error
        anomalies = mse > np.percentile(mse, 100 - threshold)
        return anomalies, mse

# Main execution
def main():
    # Generate synthetic data
    X, y = generate_sensor_data()
    
    # Preprocess data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Initialize and train VAE
    vae = VariationalAutoencoder(input_dim=X_train.shape[1], latent_dim=10)
    history = vae.train(X_train)
    
    # Detect anomalies
    anomalies, anomaly_scores = vae.detect_anomalies(X_test)
    
    # Evaluation
    print("Anomaly Detection Results:")
    print(f"Total test samples: {len(X_test)}")
    print(f"Detected anomalies: {np.sum(anomalies)}")
    print(f"True anomalies in test set: {np.sum(y_test)}")

if __name__ == "__main__":
    main()