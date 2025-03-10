# Benchmarking Scientific ML Models for Flow Prediction

This repository provides the source code and training scripts for **"Derivative-Informed Neural Operators for Accurate and Generalizable High-Fidelity 3D Fluid Flow around Geometries"** The project evaluates state-of-the-art **neural operators** for predicting 3D fluid dynamics over complex geometries.


## Paper
Our study introduces a benchmark for scientific machine learning (SciML) models in predicting 3D steady-state flow across intricate geometries using high-fidelity simulation data. The full paper can be accessed here:

**"Derivative-Informed Neural Operators for Accurate and Generalizable High-Fidelity 3D Fluid Flow around Geometries"** 
- Authors: *Ali Rabeh, Baskar Ganapathysubramanian*

## Features
- **Benchmarking of DeepONet and Geometric-DeepONet models using 4 different loss functions**.
- **Evaluation on FlowBench 3D Lid-Driven Cavity dataset**, a publicly available dataset on Hugging Face.
- **Hyperparameter tuning with WandB Sweeps**.
- **Residual and gradient calculations using FEM-based scripts**.

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
> We have included `venv_requirements.txt`, which lists all the libraries used in our environment. However, it contains some unnecessary dependencies that are not required for running the core scripts. The essential libraries are listed above for a minimal setup.To set up the environment and install dependencies using `venv_requirements.txt`:
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
