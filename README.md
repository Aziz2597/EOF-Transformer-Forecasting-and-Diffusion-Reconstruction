# EOF + Transformer Forecasting and Diffusion Reconstruction 

This repository implements a pipeline for **EOF (SVD) analysis**, **Transformer-based time-series forecasting**, and a **simple diffusion-style reconstruction** on fluid dynamics velocity fields.  
The workflow uses Principal Components (PCs) from an SVD for EOF-space forecasting, and also trains a Transformer on **PCA-reduced original data** for direct forecasting in a lower-dimensional original-space representation.

---

## Project Overview

- Load `u.mat` and `v.mat` velocity fields and stack them into a single data matrix `D`.
- Perform **SVD** to obtain EOF modes and PCs; retain a small fixed number of components (`k = 5` by default).
- Normalize PCs and create sequences for training a Transformer that forecasts EOF PCs.
- Downsample and apply **PCA** (e.g., `n_components = 50`) to the original data to create a lower-dimensional representation for an additional Transformer model trained on "original" data.
- Implement a lightweight diffusion-style reconstruction function (iterative shifting/rolling smoothing) as a baseline for filling missing values.
- Evaluate forecasting and reconstruction using **MSE** and **correlation** metrics.

---

## Key Components

### 1. Data Loading & Preprocessing
- Mount Google Drive (Colab): `drive.mount('/content/drive')`.
- Load MATLAB files: `u.mat` and `v.mat` via `scipy.io.loadmat`.
- Reshape U and V arrays and concatenate horizontally to form the data matrix `D` (shape: `n x d`).
- Optionally downsample rows for memory efficiency.

### 2. EOF Analysis (SVD)
- Compute full SVD using `scipy.linalg.svd(D, full_matrices=False)`.
- Retain a fixed number of EOF components (default `k = 5` in code2).
- Reconstruct a low-rank approximation `D_reconstructed = U_k * Sigma_k * Vt_k`.
- Extract PCs for forecasting and standardize them with `StandardScaler()`.

### 3. Sequence Preparation
- Create sequences (e.g., `seq_length = 5`) from normalized PCs for forecasting.
- Convert sequences to PyTorch tensors. In code2, tensors are prepared with shapes matching `nn.Transformer` expectations.

### 4. Transformer Forecasting (EOF PCs)
- Uses `torch.nn.Transformer` (encoder-decoder) with:
  - `d_model = input_dim` (PC dimension)
  - `num_encoder_layers = 2`, `num_decoder_layers = 2`
  - `batch_first=True`
- Model is trained to predict sequences; final training in code2 uses the output and MSE loss.

### 5. Forecasting on PCA-Reduced Original Data
- Downsample `D` (e.g., `step = 10`).
- Perform `PCA(n_components=50)` to reduce dimensionality of the original fields.
- Normalize by splitting into chunks to manage memory, concatenate normalized chunks.
- Build sequences and train a second `nn.Transformer` on this PCA-reduced representation using mini-batches (DataLoader, `batch_size = 512`).

### 6. Diffusion-style Reconstruction Baseline
- `diffusion_reconstruction(data, missing_rate=0.3)`:
  - Random mask generation to simulate missing entries.
  - Iterative simple transforms (in code2 it uses `np.roll` operations and `nan_to_num`) as a cheap reconstruction baseline.
  - **Note:** This is a heuristic smoothing baseline and **not** a PDE-based diffusion solver.

### 7. Evaluation
- Forecasting: compute **Mean Squared Error (MSE)** and **correlation coefficient** between predictions and true targets.
- Diffusion reconstruction: MSE between original and reconstructed `D`.
- Print metrics and (optionally) visualize outputs.

---

