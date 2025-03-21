import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf
import argparse
import os
import torch
import gc
import wandb
from torch.utils import data
import random
import numpy as np
import time
from data.dataset import LidDrivenDataset3D
from models.deeponet.deeponet import DeepONet
from models.geometric_deeponet.geometric_deeponet import GeometricDeepONet

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)
    
    
def main(model_name, config=None):

    if config is None:
        # Load the configuration file
        config_path = f'configs/{model_name}/conf.yaml'
        config = OmegaConf.load(config_path)
    else:
        # Merge the Wandb config with the base config
        config_path = f'configs/{model_name}/conf.yaml'
        base_config = OmegaConf.load(config_path)
        # Ensure the Wandb config is nested correctly under 'model'
        wandb_config = OmegaConf.create({'model': {k: v for k, v in config.items()}})
        config = OmegaConf.merge(base_config, wandb_config)

    seed = config.trainer.seed
    set_seed(seed)
    
    train_dataset = LidDrivenDataset3D(
        file_path_x=config.data.file_path_train_x,
        file_path_y=config.data.file_path_train_y,
        data_type=config.data.type        )
        
    test_dataset = LidDrivenDataset3D(
        file_path_x=config.data.file_path_test_x,
        file_path_y=config.data.file_path_test_y,
        data_type=config.data.type
        )
        
    train_loader = data.DataLoader(train_dataset,
                                    batch_size=config.data.batch_size,
                                    shuffle=config.data.shuffle,
                                    drop_last=False,
                                    num_workers=6
                                    )
    test_loader = data.DataLoader(test_dataset,
                                    batch_size=config.data.batch_size,
                                    shuffle=config.data.shuffle,
                                    drop_last=False,
                                    num_workers=6
                                    )

    # Define required model parameters for each model
    model_params = {
        'deeponet': ['input_channels_func', 'input_channels_loc', 'output_channels', 'branch_net_layers', 'trunk_net_layers', 'modes', 'lr', 'plot_path', 'log_file'],
        'geometric-deeponet': ['input_channels_func', 'input_channels_loc', 'output_channels', 'branch_net_layers', 'trunk_net_layers', 'modes', 'lr', 'plot_path', 'log_file'],
    }

    # Filter the parameters for the model
    if model_name not in model_params:
        raise ValueError(f"Unknown model name: {model_name}")

    params = {k: v for k, v in config.model.items() if k in model_params[model_name]}
    params["loss_type"] = "relative_mse"
    loss_type = params.pop("loss_type")

    print(f"Model parameters for {model_name} have been successfully set.")

    # Initialize the model
    if model_name == 'deeponet':
        model = DeepONet(**params)
    elif model_name == 'geometric-deeponet':
        model = GeometricDeepONet(**params)                
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.loss_type = loss_type
    # Define the callbacks
    wandb_logger = WandbLogger(
        project=model_name, 
        config=config, 
        log_model='false'
        )
    
    
    wandb_logger.experiment.config.update(OmegaConf.to_container(config, resolve=True))
    wandb_logger.experiment.save(config_path, policy="now")

    # Get the run ID to differentiate checkpoints
    run_id = wandb_logger.experiment.id  # or use .name for the run name
    # Define the checkpoint directory with the run ID to avoid overwriting
    checkpoint_dir = os.path.join(config.callbacks.checkpoint.dirpath, f'{model_name}_checkpoints_{run_id}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    
    checkpoint_callback = ModelCheckpoint(
        monitor=config.callbacks.checkpoint.monitor,
        dirpath=checkpoint_dir,
        filename=config.callbacks.checkpoint.filename,
        save_top_k=config.callbacks.checkpoint.save_top_k,
        mode=config.callbacks.checkpoint.mode,
        every_n_epochs=50 
    )

    # Define the trainer
    trainer = pl.Trainer(
        max_epochs=config.trainer.max_epochs,
        callbacks=[checkpoint_callback],
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        log_every_n_steps=config.trainer.log_every_n_steps,
        logger=wandb_logger,
        precision=32
    )


    # Measure the time taken for training and testing
    start_time = time.time()
    
    # Train the model
    trainer.fit(model, train_loader, test_loader)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total training time: {elapsed_time:.2f} seconds")

    # Log the elapsed time to Wandb
    wandb_logger.log_metrics({'training_time': elapsed_time})
    
    # Clean up to free GPU memory
    del model, train_loader, test_loader
    torch.cuda.empty_cache()
    gc.collect()
    wandb.finish()


def sweep(model_name):
    # Load the configuration file to get sweep parameters
    config_path = f'configs/{model_name}/conf.yaml'
    base_config = OmegaConf.load(config_path)
    sweep_parameters = base_config.sweep_parameters

    # Convert the sweep parameters to a dictionary
    sweep_parameters = OmegaConf.to_container(sweep_parameters, resolve=True)

    # Define sweep configuration
    sweep_configuration = {
        'method': 'grid',
        'name': 'sweep',
        'metric': {'goal': 'minimize', 'name': 'val_loss_full'},
        'parameters': {k: v for k, v in sweep_parameters.items()}
    }

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=model_name)

    def train(config=None):
        with wandb.init(config=config):
            config = wandb.config
            # Nest Wandb config correctly under 'model'
            wandb_config = OmegaConf.create({'model': {k: v for k, v in config.items()}})
            main(model_name, config=wandb_config)

    # Start the sweep
    wandb.agent(sweep_id, function=train)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model with a specified configuration.")
    parser.add_argument('--model', type=str, required=True, help='Name of the model to train.')
    parser.add_argument('--sweep', action='store_true', help='Run hyperparameter sweep.')
    args = parser.parse_args()

    if args.sweep:
        sweep(args.model)
    else:
        main(args.model)
