import os
import torch
from torch.utils import data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from mpl_toolkits.axes_grid1 import make_axes_locatable
from models.deeponet.deeponet import DeepONet
from models.geometric_deeponet.geometric_deeponet import GeometricDeepONet
from data.dataset import LidDrivenDataset3D
from residual import ResidualLoss

torch.cuda.empty_cache()


def plot_total_error_3d(idx, error_x, error_y, error_z, plot_dir, sdf):
    """
    Plot the total error sqrt(error_x**2 + error_y**2 + error_z**2) for 3D data with an SDF mask applied.
    This function plots the central slices in the XY, XZ, and YZ planes.
    """
    # Calculate the total error field
    sdf_numpy = sdf.cpu().numpy() if sdf.is_cuda else sdf.numpy()
    error_x = error_x.cpu().numpy() if error_x.is_cuda else error_x.numpy()
    error_y = error_y.cpu().numpy() if error_y.is_cuda else error_y.numpy()
    error_z = error_z.cpu().numpy() if error_z.is_cuda else error_z.numpy()
    
    
    total_error = np.sqrt(error_x**2 + error_y**2 + error_z**2)**0.5
    epsilon = 1e-9
    total_error_log = np.log10(total_error + epsilon)
    # Convert SDF tensor to numpy and create a mask where SDF >= 0 (inside the fluid domain)
    Nx = Ny = Nz = sdf_numpy.shape[-1]  # Adjust indices based on your tensor dimension order   

    center_x, center_y, center_z = Nx // 2, Ny // 2, Nz // 2
    
    planes = {
        "XY": total_error_log[0, :, :, center_z],
        "XZ": total_error_log[0, :, center_y, :],
        "YZ": total_error_log[0, center_x, :, :]
    }
    sdf_planes = {
        "XY": sdf_numpy[0, 1, :Nx-1, :Ny-1, center_z],
        "XZ": sdf_numpy[0, 1, :Nx-1, center_y, :Nz-1],
        "YZ": sdf_numpy[0, 1, center_x, :Ny-1, :Nz-1]
    }

    for plane, data in planes.items():
        mask = sdf_planes[plane] >= 0  # SDF mask for the current plane
        masked_data = np.ma.masked_where(~mask, data)
        # Plotting the masked total error for each plane
        plt.figure()
        plt.imshow(masked_data, cmap='jet', origin='lower', vmin=-4, vmax=1)  # Adjust color limits if needed
        plt.colorbar()
        plt.savefig(os.path.join(plot_dir, f'deriv_error_{plane}_{idx}.png'))
        plt.close()


def plot_total_deriv_3d(idx, deriv_x, deriv_y, deriv_z, plot_dir, sdf, deriv_type, color_limits):
    """
    Plot the total error sqrt(error_x**2 + error_y**2 + error_z**2) for 3D data with an SDF mask applied.
    This function plots the central slices in the XY, XZ, and YZ planes.
    
    Parameters:
        idx (int): Index of the sample being plotted.
        deriv_x (tensor): X component of the derivative.
        deriv_y (tensor): Y component of the derivative.
        deriv_z (tensor): Z component of the derivative.
        plot_dir (str): Directory to save the plots.
        sdf (tensor): Signed Distance Field tensor.
        deriv_type (str): Specifies whether the derivatives are 'true' or 'pred'.
    """
    # Calculate the total error field
    sdf_numpy = sdf.cpu().numpy() if sdf.is_cuda else sdf.numpy()
    deriv_x = deriv_x.cpu().numpy() if deriv_x.is_cuda else deriv_x.numpy()
    deriv_y = deriv_y.cpu().numpy() if deriv_y.is_cuda else deriv_y.numpy()
    deriv_z = deriv_z.cpu().numpy() if deriv_z.is_cuda else deriv_z.numpy()
    
    total_deriv = np.sqrt(deriv_x**2 + deriv_y**2 + deriv_z**2)
    
    # Adjust indices based on your tensor dimension order
    Nx = Ny = Nz = sdf_numpy.shape[-1]
    center_x, center_y, center_z = Nx // 2, Ny // 2, Nz // 2
    
    planes = {
        "XY": total_deriv[0, :, :, center_z],
        "XZ": total_deriv[0, :, center_y, :],
        "YZ": total_deriv[0, center_x, :, :]
    }
    sdf_planes = {
        "XY": sdf_numpy[0, 1, :Nx-1, :Ny-1, center_z],
        "XZ": sdf_numpy[0, 1, :Nx-1, center_y, :Nz-1],
        "YZ": sdf_numpy[0, 1, center_x, :Ny-1, :Nz-1]
    }

    for plane, data in planes.items():
        mask = sdf_planes[plane] >= 0  # SDF mask for the current plane
        masked_data = np.ma.masked_where(~mask, data)
        vmin, vmax = color_limits[plane]
        # Plotting the masked total error for each plane
        plt.figure()
        plt.imshow(masked_data, cmap='jet', origin='lower', vmin=vmin, vmax=vmax)  # Adjust color limits if needed
        plt.colorbar()
        plt.savefig(os.path.join(plot_dir, f'{deriv_type}_deriv_{plane}_{idx}.png'))
        plt.close()

def get_color_limits(true_x_elm, true_y_elm, true_z_elm, sdf):
    """
    Calculate vmin and vmax for color scaling based on the true derivative tensors.
    
    Parameters:
        true_x_elm, true_y_elm, true_z_elm (torch.Tensor): Tensors of derivative components.
        sdf (torch.Tensor): Signed Distance Field tensor.
    
    Returns:
        color_limits (dict): Dictionary with vmin and vmax for each plane (XY, XZ, YZ).
    """
    # Ensure tensors are on CPU and converted to NumPy arrays
    true_x_elm = true_x_elm.cpu().numpy()
    true_y_elm = true_y_elm.cpu().numpy()
    true_z_elm = true_z_elm.cpu().numpy()
    sdf_numpy = sdf.cpu().numpy()

    Nx, Ny, Nz = true_x_elm.shape[-3], true_x_elm.shape[-2], true_x_elm.shape[-1]
    center_x, center_y, center_z = Nx // 2, Ny // 2, Nz // 2

    color_limits = {}
    planes = {
        "XY": np.sqrt(true_x_elm[0, :, :, center_z]**2 + true_y_elm[0, :, :, center_z]**2 + true_z_elm[0, :, :, center_z]**2),
        "XZ": np.sqrt(true_x_elm[0, :, center_y, :]**2 + true_y_elm[0, :, center_y, :]**2 + true_z_elm[0, :, center_y, :]**2),
        "YZ": np.sqrt(true_x_elm[0, center_x, :, :]**2 + true_y_elm[0, center_x, :, :]**2 + true_z_elm[0, center_x, :, :]**2)
    }

    sdf_planes = {
        "XY": sdf_numpy[0, 1, :Nx, :Ny, center_z],
        "XZ": sdf_numpy[0, 1, :Nx, center_y, :Nz],
        "YZ": sdf_numpy[0, 1, center_x, :Ny, :Nz]
    }
    for plane in planes:
        mask = sdf_planes[plane] >= 0
        masked_data = np.ma.masked_where(~mask, planes[plane])
        color_limits[plane] = (masked_data.min(), masked_data.max())

    return color_limits
           
    
def load_model(model_name, checkpoint_path, config):
    params = {k: v for k, v in config.model.items()}
    model_dict = {
        'deeponet': DeepONet,
        'geometric-deeponet': GeometricDeepONet
    }
    if model_name not in model_dict:
        raise ValueError(f"Unknown model name: {model_name}")
    return model_dict[model_name].load_from_checkpoint(checkpoint_path, strict=False, **params)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def custom_loss(yhat, y, x_full):
    mses = torch.zeros(17).to(yhat.device)  # Updated to hold all required values
    
    sdf = x_full[:, 1, :, :, :].to(yhat.device)  
    condition = (sdf > 0)
    condition_near = (sdf > 0) & (sdf <= 0.2)
    
    y_outside_obj = torch.where(condition.unsqueeze(1).expand_as(y), y, torch.tensor(0.0, device=y.device))
    y_near_obj = torch.where(condition_near.unsqueeze(1).expand_as(y), y, torch.tensor(0.0, device=y.device))
    
    yhat_outside_obj = torch.where(condition.unsqueeze(1).expand_as(yhat), yhat, torch.tensor(0.0, device=yhat.device))
    yhat_near_obj = torch.where(condition_near.unsqueeze(1).expand_as(yhat), yhat, torch.tensor(0.0, device=yhat.device))
    
    error_outside = yhat_outside_obj - y_outside_obj
    error_near = yhat_near_obj - y_near_obj
    
    mses[0] = torch.norm(error_outside, p=2) / torch.norm(y_outside_obj, p=2) if torch.norm(y_outside_obj, p=2) > 0 else torch.tensor(float('nan')).to(y.device)
    mses[1] = torch.norm(error_near, p=2) / torch.norm(y_near_obj, p=2) if torch.norm(y_near_obj, p=2) > 0 else torch.tensor(float('nan')).to(y.device)
    mses[2] = torch.mean(error_outside ** 2)
    mses[3] = torch.mean(error_near ** 2)
    
    for i in range(3):
        norm_y_out = torch.norm(y_outside_obj[:, i, :, :, :], p=2)
        norm_y_near = torch.norm(y_near_obj[:, i, :, :, :], p=2)
        
        mses[4 + i] = torch.norm(error_outside[:, i, :, :, :], p=2) / norm_y_out if norm_y_out > 0 else torch.tensor(float('nan')).to(y.device)
        mses[7 + i] = torch.norm(error_near[:, i, :, :, :], p=2) / norm_y_near if norm_y_near > 0 else torch.tensor(float('nan')).to(y.device)
        mses[10 + i] = torch.mean(error_outside[:, i, :, :, :] ** 2)
        mses[13 + i] = torch.mean(error_near[:, i, :, :, :] ** 2)
    
    return mses

def save_individual_plot(label, field, idx, plot_dir, vmin=None, vmax=None):
    plt.figure()
    plt.imshow(field, cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.savefig(os.path.join(plot_dir, f'{label}_{idx}.png'))
    plt.close()

def save_plane_plots(y, y_hat, sdf, idx, sample_dir, channel_labels=('u', 'v', 'w')):
    individual_plot_dir = os.path.join(sample_dir, 'individual_plots')
    os.makedirs(individual_plot_dir, exist_ok=True)
    #print("sdf.shape: ", sdf.shape)
    center_idx = y.shape[-1] // 2
    epsilon = 1e-9

    for plane, sdf_slice, mask, truth_slices, pred_slices in [
        ("XY", sdf[0, 1, :, :, center_idx].cpu().numpy(),
         sdf[0, 1, :, :, center_idx].cpu().numpy() <= 0,
         [y[0, ch, :, :, center_idx].cpu().numpy() for ch in range(3)],
         [y_hat[0, ch, :, :, center_idx].cpu().numpy() for ch in range(3)]),
        ("XZ", sdf[0, 1, :, center_idx, :].cpu().numpy(),
         sdf[0, 1, :, center_idx, :].cpu().numpy() <= 0,
         [y[0, ch, :, center_idx, :].cpu().numpy() for ch in range(3)],
         [y_hat[0, ch, :, center_idx, :].cpu().numpy() for ch in range(3)]),
        ("YZ", sdf[0, 1, center_idx, :, :].cpu().numpy(),
         sdf[0, 1, center_idx, :, :].cpu().numpy() <= 0,
         [y[0, ch, center_idx, :, :].cpu().numpy() for ch in range(3)],
         [y_hat[0, ch, center_idx, :, :].cpu().numpy() for ch in range(3)]),
    ]:
        # Mask ground truth and predictions
        truth_masked = [np.ma.masked_where(mask, truth) for truth in truth_slices]
        pred_masked = [np.ma.masked_where(mask, pred) for pred in pred_slices]
        error_masked = [np.ma.masked_where(mask, np.log10(np.abs(truth - pred) + epsilon))
                        for truth, pred in zip(truth_masked, pred_masked)]

        # Save individual plots for each channel
        for i, label in enumerate(channel_labels):
            vmin = truth_masked[i].min() #min(truth_masked[i].min(), pred_masked[i].min())
            vmax = truth_masked[i].max() #max(truth_masked[i].max(), pred_masked[i].max())
            save_individual_plot(f"{label}_{plane}_truth", truth_masked[i], idx, individual_plot_dir, vmin, vmax)
            save_individual_plot(f"{label}_{plane}_pred", pred_masked[i], idx, individual_plot_dir, vmin, vmax)
            save_individual_plot(f"{label}_{plane}_log_error", error_masked[i], idx, individual_plot_dir, vmin=-6, vmax=0)

        # Create 3x3 plot for the plane
        fig, axs = plt.subplots(3, 3, figsize=(15, 15))
        rows = ["Ground Truth", "Prediction", "Log(Error)"]
        cols = ["u", "v", "w"]

        for i, (truth, pred, error) in enumerate(zip(truth_masked, pred_masked, error_masked)):
            # Ensure same colorbar limits for ground truth and prediction
            vmin = truth.min() #min(truth.min(), pred.min())
            vmax = truth.max() #max(truth.max(), pred.max())

            axs[0, i].imshow(truth, cmap="jet", origin="lower", vmin=vmin, vmax=vmax)
            axs[0, i].set_title(f"{rows[0]} {cols[i]} ({plane})")
            divider = make_axes_locatable(axs[0, i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(axs[0, i].images[0], cax=cax)

            axs[1, i].imshow(pred, cmap="jet", origin="lower", vmin=vmin, vmax=vmax)
            axs[1, i].set_title(f"{rows[1]} {cols[i]} ({plane})")
            divider = make_axes_locatable(axs[1, i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(axs[1, i].images[0], cax=cax)

            axs[2, i].imshow(error, cmap="jet", origin="lower", vmin=-6, vmax=0)
            axs[2, i].set_title(f"{rows[2]} {cols[i]} ({plane})")
            divider = make_axes_locatable(axs[2, i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(axs[2, i].images[0], cax=cax)

        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, f"{plane}_slices_{idx}.png"))
        plt.close()

def main(model_name, config_path, checkpoint_path):
    config = OmegaConf.load(config_path)

    plot_dir = config.model.plot_path
    os.makedirs(plot_dir, exist_ok=True)

    model = load_model(model_name, checkpoint_path, config)
    model = model.to(device) #model = model.cuda()

    test_dataset_SDF = LidDrivenDataset3D(
        file_path_x=config.data.file_path_test_x,
        file_path_y=config.data.file_path_test_y,
        data_type=config.data.type,
    )

    test_dataset = LidDrivenDataset3D(
        file_path_x=config.data.file_path_test_x,
        file_path_y=config.data.file_path_test_y,
        data_type=config.data.type,
    )
    
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=6
    )
    residual_loss_fn = ResidualLoss(domain_size=128, device=device).to(device)
    model.eval()

    all_losses = []
    total_deriv_errors = []
    total_relative_deriv_errors = []    
    solenoidality_vals = []          
    sample_to_plot = [22, 33, 44, 55, 66, 77]

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            x_batch, y = batch
            x_batch = x_batch.to(device)
            y = y.to(device)

            sdf = test_dataset_SDF.x[idx:idx+1].to(device) #x_batch[:, 1:2, :, :, :]  # Extract SDF
            y_hat = model(x_batch)

            # Compute loss
            losses = custom_loss(y_hat, y[:, :3, :, :, :], sdf)
            all_losses.append(losses.cpu().numpy())
            deriv_error, solenoidality, error_x, error_y, error_z, pred_x_elm, pred_y_elm, pred_z_elm, \
                true_x_elm, true_y_elm, true_z_elm = residual_loss_fn(x_batch, y_hat, y)
            total_deriv_errors.append(deriv_error.item())
            solenoidality_vals.append(solenoidality.item())
            # Save plots for specified samples
            if idx in sample_to_plot:
                sample_dir = os.path.join(plot_dir, f"sample_{idx}")
                os.makedirs(sample_dir, exist_ok=True)
                save_plane_plots(y, y_hat, sdf, idx, sample_dir)
                ind_plots_dir = os.path.join(sample_dir, f"individual_plots")
                plot_total_error_3d(idx, error_x, error_y, error_z, ind_plots_dir, sdf)
                color_limits = get_color_limits(true_x_elm, true_y_elm, true_z_elm, sdf)
                plot_total_deriv_3d(idx, pred_x_elm, pred_y_elm, pred_z_elm, ind_plots_dir, sdf, 'pred', color_limits)
                plot_total_deriv_3d(idx, true_x_elm, true_y_elm, true_z_elm, ind_plots_dir, sdf, 'true', color_limits)
            

    # Save average losses to a text file
    all_losses = np.array(all_losses)
    avg_losses = np.nanmean(all_losses, axis=0)

    with open(os.path.join(plot_dir, "losses.txt"), "w") as f:
        f.write(f"Average Relative L2 (Outside Object): {avg_losses[0]:.6f}\n")
        f.write(f"Average Relative L2 (Near Object): {avg_losses[1]:.6f}\n")
        f.write(f"Average Relative L2 (u, Outside): {avg_losses[4]:.6f}\n")
        f.write(f"Average Relative L2 (v, Outside): {avg_losses[5]:.6f}\n")
        f.write(f"Average Relative L2 (w, Outside): {avg_losses[6]:.6f}\n")
        f.write(f"Average Relative L2 (u, Near): {avg_losses[7]:.6f}\n")
        f.write(f"Average Relative L2 (v, Near): {avg_losses[8]:.6f}\n")
        f.write(f"Average Relative L2 (w, Near): {avg_losses[9]:.6f}\n")
        f.write(f"Average MSE (Outside Object): {avg_losses[2]:.6f}\n")
        f.write(f"Average MSE (Near Object): {avg_losses[3]:.6f}\n")
        f.write(f"Average MSE (u, Outside): {avg_losses[10]:.6f}\n")
        f.write(f"Average MSE (v, Outside): {avg_losses[11]:.6f}\n")
        f.write(f"Average MSE (w, Outside): {avg_losses[12]:.6f}\n")
        f.write(f"Average MSE (u, Near): {avg_losses[13]:.6f}\n")
        f.write(f"Average MSE (v, Near): {avg_losses[14]:.6f}\n")
        f.write(f"Average MSE (w, Near): {avg_losses[15]:.6f}\n")
        f.write(f"Average Derivative Error: {np.mean(total_deriv_errors):.6f}\n")
        f.write(f"Average Continuity: {np.mean(solenoidality_vals):.6f}\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process different models for prediction and plotting.")
    parser.add_argument('--model', type=str, required=True, help='Name of the model to load (fno, cno, deeponet, geometric-deeponet).')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint.')

    args = parser.parse_args()
    main(args.model, args.config, args.checkpoint)
