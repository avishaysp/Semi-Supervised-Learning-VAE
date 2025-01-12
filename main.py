import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import os
import pickle
import matplotlib.pyplot as plt

from data_load import get_fashion_mnist, get_labeled_data, get_mnist
from vae import VAE, extract_latent_features, train_vae

MNIST_INPUT_DIM = 784
LATENT_DIM = 50
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-4
LABEL_COUNTS = [100, 600, 1000, 3000]
SEED = 42

def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.mps.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid()
    plt.savefig('models/loss_plot.png')
    plt.close()
    print("Loss plot saved as models/loss_plot.png")

def main():
    set_seed()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs("models", exist_ok=True)

    # Get train and validation loaders for MNIST
    mnist_train_loader, mnist_val_loader = get_mnist()
    
    vae = VAE(MNIST_INPUT_DIM, LATENT_DIM).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=LEARNING_RATE)
    
    print("Training VAE on MNIST...")
    train_losses, val_losses = train_vae(vae, mnist_train_loader, mnist_val_loader, optimizer, device, EPOCHS)
    
    # Plot and save the losses
    plot_losses(train_losses, val_losses)
    
    torch.save(vae.state_dict(), "models/vae_model.pth")

    vae.eval()
    
    # Now evaluate on FashionMNIST with different label counts
    fashion_train_loader, fashion_test_loader = get_fashion_mnist()
    
    test_latent, test_labels = extract_latent_features(vae, fashion_test_loader, device)
    
    results = {}
    for num_labels in LABEL_COUNTS:
        print(f"\nTraining SVM with {num_labels} labels...")
        
        labeled_dataset = get_labeled_data(fashion_train_loader.dataset, num_labels // 10)
        labeled_loader = DataLoader(labeled_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        train_latent, train_labels = extract_latent_features(vae, labeled_loader, device)
        
        # Train and evaluate SVM
        svm = SVC(kernel='rbf')
        svm.fit(train_latent, train_labels)
        
        # Save SVM model
        with open(f"models/svm_model_{num_labels}.pkl", 'wb') as f:
            pickle.dump(svm, f)
        
        predictions = svm.predict(test_latent)
        error_rate = (1 - accuracy_score(test_labels, predictions)) * 100
        
        results[num_labels] = error_rate
        print(f"Test error rate with {num_labels} labels: {error_rate:.2f}%")

    # Save results
    with open("models/results.pkl", 'wb') as f:
        pickle.dump(results, f)

    print("\nFinal Results:")
    for num_labels, error_rate in results.items():
        print(f"Labels: {num_labels}, Error Rate: {error_rate:.2f}%")

if __name__ == '__main__':
    main()