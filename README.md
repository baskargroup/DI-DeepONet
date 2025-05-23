# Benchmarking Scientific ML Models for Flow Prediction

This repository provides the source code and training scripts for **"Derivative-Informed Neural Operators for Accurate and Generalizable High-Fidelity 3D Fluid Flow around Geometries"** The project evaluates state-of-the-art **neural operators** for predicting 3D fluid dynamics over complex geometries.


## Paper
Our study introduces a benchmark for scientific machine learning (SciML) models in predicting 3D steady-state flow across intricate geometries using high-fidelity simulation data. The full paper can be accessed here:

**"Derivative-Informed Neural Operators for Accurate and Generalizable High-Fidelity 3D Fluid Flow around Geometries"** 
- Authors: *Ali Rabeh, Adarsh Krishnamurthy, Baskar Ganapathysubramanian*

## Features

- **Benchmarking of DeepONet and Geometric-DeepONet models using 4 different loss functions**.
- **Evaluation on FlowBench 3D Lid-Driven Cavity dataset, a publicly available dataset on Hugging Face**.
- **Hyperparameter tuning with WandB Sweeps**.
- **Residual and gradient calculations using FEM-based scripts**.
- **Support for multiple loss formulations, including MSE, relative MSE, derivative-informed losses, and physics-constrained losses**.

## Datasets
This study utilizes the **FlowBench 3D Lid-Driven Cavity (LDC) dataset**, which is publicly accessible on Hugging Face: [**FlowBench LDC Dataset**](https://huggingface.co/datasets/BGLab/FlowBench/tree/main/LDC_NS_3D)

The dataset is licensed under **CC-BY-NC-4.0** and serves as a benchmark for the development and evaluation of scientific machine learning (SciML) models.

### Dataset Structure
- **Geometry representation:** SDF
- **Resolution:** 128×128x128
- **Fields:** Velocity (u, v, w)
- **Stored as:** Numpy tensors (`.npz` format)

# Installation  

## Essential Dependencies  

This repository requires the following core libraries: 

- **`torch`** – PyTorch framework for deep learning 
- **`pytorch-lightning`** – High-level PyTorch wrapper for training 
- **`omegaconf`** – Configuration management 
- **`wandb`** – Experiment tracking 
- **`numpy`** – Numerical computations 
- **`scipy`** – Scientific computing 

> **Note:** 
> We have included `venv_requirements.txt`, which lists all the libraries used in our environment. To set up the environment and install dependencies using `venv_requirements.txt`:
```bash
python3 -m venv sciml
source sciml/bin/activate 
pip install --upgrade pip setuptools wheel Cython
pip install -r venv_requirements.txt
```

## Model Training
To train a **Neural Operator**, run the following command:
```bash
python3 main.py --model "model_name" --sweep
```

Before training, you need to specify the dataset paths in the **configurations** (YAML files):
```yaml
data:
  file_path_train_x: ./data/train_x.npz
  file_path_train_y: ./data/train_y.npz
  file_path_test_x: ./data/test_x.npz
  file_path_test_y: ./data/test_y.npz
```
## Loss Functions

This repository supports multiple loss functions tailored for scientific machine learning models. The `loss_type` is set directly in `main.py` when initializing the model. The available options include:

- **`"mse"`** – Standard Mean Squared Error (MSE) loss.
- **`"relative_mse"`** – MSE loss normalized per-channel.
- **`"derivative_mse"`** – MSE loss including first-order derivatives.
- **`"relative_derivative_mse"`** – Relative MSE loss incorporating derivative information.
- **`"pure_deriv"`** – Loss function based purely on derivatives and boundary conditions.
- **`"physics_loss"`** – Physics-constrained loss incorporating derivatives, boundary conditions, and continuity constraints.

The loss function is defined in `main.py` as:
```python
params["loss_type"] = "relative_mse"  # Modify this to select a different loss
```
To change the loss function, update the `loss_type` assignment in `main.py` before training.

## Model Inference
For model inference, use the scripts in the `plotting_script` folder:
```bash
python3 process_3D.py --model "$model" --config "$config_path" --checkpoint "$checkpoint_file"
```

## Evaluation & Plotting
The `plotting_script` folder contains Python scripts for:
- **Evaluating field predictions and errors**.
- **Calculating continuity residuals using finite element methods (FEM)**.
- **Computing solution gradients**.

Example usage:
```bash
python3 plotting_script/process_3D.py --model "$model" --config "$config_path" --checkpoint "$checkpoint_file
```

## Contributing
We welcome contributions! If you’d like to improve this project, please fork the repository and submit a pull request.

## License
This repository is licensed under the MIT License.
