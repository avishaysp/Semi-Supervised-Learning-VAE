# Semi-Supervised Learning with VAE

This project implements the M1 scheme from the paper "Semi-supervised Learning with Deep Generative Models" by Kingma et al. The implementation uses a VAE for feature extraction followed by an SVM for classification on the Fashion MNIST dataset.

## Model Details

- VAE architecture follows the paper's MNIST network specification
- SVM uses RBF kernel for classification
- Fixed seed (42) is used for reproducibility
- Results are reported for 100, 600, 1000, and 3000 labels

## Training and Testing

1. Install requirements:
```bash
pip install torch torchvision scikit-learn numpy
```

2. Run the training:
```bash
python main.py
```

This will:
- Train the VAE on MNIST
- Extract features using the trained VAE
- Train SVM models with different label counts
- Save all models and results

## Model Storage

The following files will be created in the `models` directory:
- `vae_model.pth`: Trained VAE weights
- `svm_model_{n}.pkl`: SVM models for each label count
- `results.pkl`: Dictionary containing accuracy results

## Results

The accuracy results will be printed for each label count (100, 600, 1000, 3000) and saved in results.pkl.
