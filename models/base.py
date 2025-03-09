import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from models.deriv_calc import ResidualLoss


class BaseLightningModule(pl.LightningModule):
    def __init__(self, lr, plot_path, log_file, **kwargs):
        """
        loss_type:
        - "mse": Standard MSE loss
        - "relative_mse": MSE loss with per-plane RMS normalization
        - "derivative_mse": MSE loss including derivatives
        """
        super(BaseLightningModule, self).__init__()
        self.lr = lr
        self.plot_path = plot_path
        self.log_file = log_file
        self.loss_type = kwargs.pop("loss_type", "mse")  # Default to "mse" 
        self.residual_loss = ResidualLoss(domain_size=128, device=self.device)

    def forward(self, x):
        pass  # Forward pass should be implemented in subclasses

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


    def compute_loss(self, x, y, y_hat):
        """
        Computes loss based on the selected loss_type.
        """
        u_gd, v_gd, w_gd, *extra_gd = self.residual_loss(x, y)
        u, v, w, *extra = self.residual_loss(x, y_hat)

        mask = (x[:, 1, :, :, :] > 0.).float().unsqueeze(1)
        mask_sum = mask.sum((1, 2, 3, 4))
        mask = mask.expand(-1, y.shape[1], -1, -1, -1)
        y, y_hat = y * mask, y_hat * mask

        if self.loss_type == "mse":
            # Standard MSE loss
            loss_terms = {
                "loss_u": ((u_gd - u) ** 2).sum() / mask_sum,
                "loss_v": ((v_gd - v) ** 2).sum() / mask_sum,
                "loss_w": ((w_gd - w) ** 2).sum() / mask_sum,
            }

        elif self.loss_type == "relative_mse":
            # Relative MSE loss
            weight_u, weight_v, weight_w = 1, 3, 150
            loss_terms = {
                "loss_u": weight_u*((u_gd - u) ** 2).sum() / mask_sum,
                "loss_v": weight_v*((v_gd - v) ** 2).sum() / mask_sum,
                "loss_w": weight_w*((w_gd - w) ** 2).sum() / mask_sum,
            }

        elif self.loss_type == "derivative_mse":
            # MSE loss including derivatives
            u_x_gd, u_y_gd, u_z_gd, v_x_gd, v_y_gd, v_z_gd, w_x_gd, w_y_gd, w_z_gd = extra_gd
            u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y, w_z = extra

            h = 2.0 / x.shape[-1]  # Grid spacing
            loss_terms = {
                "loss_u": ((u_gd - u) ** 2).sum() / mask_sum,
                "loss_v": ((v_gd - v) ** 2).sum() / mask_sum,
                "loss_w": ((w_gd - w) ** 2).sum() / mask_sum,
                "loss_u_x": h * ((u_x_gd - u_x) ** 2).sum() / mask_sum,
                "loss_u_y": h * ((u_y_gd - u_y) ** 2).sum() / mask_sum,
                "loss_u_z": h * ((u_z_gd - u_z) ** 2).sum() / mask_sum,
                "loss_v_x": h * ((v_x_gd - v_x) ** 2).sum() / mask_sum,
                "loss_v_y": h * ((v_y_gd - v_y) ** 2).sum() / mask_sum,
                "loss_v_z": h * ((v_z_gd - v_z) ** 2).sum() / mask_sum,
                "loss_w_x": h * ((w_x_gd - w_x) ** 2).sum() / mask_sum,
                "loss_w_y": h * ((w_y_gd - w_y) ** 2).sum() / mask_sum,
                "loss_w_z": h * ((w_z_gd - w_z) ** 2).sum() / mask_sum,
            }

        elif self.loss_type == "relative_derivative_mse":
            # MSE loss including derivatives
            u_x_gd, u_y_gd, u_z_gd, v_x_gd, v_y_gd, v_z_gd, w_x_gd, w_y_gd, w_z_gd = extra_gd
            u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y, w_z = extra
            
            weight_u, weight_v, weight_w = 1, 3, 150
            weight_u_x, weight_u_y, weight_u_z = 15, 1, 30
            weight_v_x, weight_v_y, weight_v_z = 50, 30, 5
            weight_w_x, weight_w_y, weight_w_z = 600, 750, 600

            h = 2.0 / x.shape[-1]  # Grid spacing
            loss_terms = {
                "loss_u": weight_u * ((u_gd - u) ** 2).sum() / mask_sum,
                "loss_v": weight_v * ((v_gd - v) ** 2).sum() / mask_sum,
                "loss_w": weight_w * ((w_gd - w) ** 2).sum() / mask_sum,
                "loss_u_x": weight_u_x * h * ((u_x_gd - u_x) ** 2).sum() / mask_sum,
                "loss_u_y": weight_u_y * h * ((u_y_gd - u_y) ** 2).sum() / mask_sum,
                "loss_u_z": weight_u_z * h * ((u_z_gd - u_z) ** 2).sum() / mask_sum,
                "loss_v_x": weight_v_x * h * ((v_x_gd - v_x) ** 2).sum() / mask_sum,
                "loss_v_y": weight_v_y * h * ((v_y_gd - v_y) ** 2).sum() / mask_sum,
                "loss_v_z": weight_v_z * h * ((v_z_gd - v_z) ** 2).sum() / mask_sum,
                "loss_w_x": weight_w_x * h * ((w_x_gd - w_x) ** 2).sum() / mask_sum,
                "loss_w_y": weight_w_y * h * ((w_y_gd - w_y) ** 2).sum() / mask_sum,
                "loss_w_z": weight_w_z * h * ((w_z_gd - w_z) ** 2).sum() / mask_sum,
            }
        elif self.loss_type == "pure_deriv":
            # MSE loss including derivatives
            u_x_gd, u_y_gd, u_z_gd, v_x_gd, v_y_gd, v_z_gd, w_x_gd, w_y_gd, w_z_gd = extra_gd
            u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y, w_z = extra
            weight_u_x, weight_u_y, weight_u_z = 15, 1, 30
            weight_v_x, weight_v_y, weight_v_z = 50, 30, 5
            weight_w_x, weight_w_y, weight_w_z = 600, 750, 600
            weight_boundary = 5
            
            h = 2.0 / x.shape[-1]  # Grid spacing
            loss_terms = {
                "loss_u_x": weight_u_x * h * ((u_x_gd - u_x) ** 2).sum() / mask_sum,
                "loss_u_y": weight_u_y * h * ((u_y_gd - u_y) ** 2).sum() / mask_sum,
                "loss_u_z": weight_u_z * h * ((u_z_gd - u_z) ** 2).sum() / mask_sum,
                "loss_v_x": weight_v_x * h * ((v_x_gd - v_x) ** 2).sum() / mask_sum,
                "loss_v_y": weight_v_y * h * ((v_y_gd - v_y) ** 2).sum() / mask_sum,
                "loss_v_z": weight_v_z * h * ((v_z_gd - v_z) ** 2).sum() / mask_sum,
                "loss_w_x": weight_w_x * h * ((w_x_gd - w_x) ** 2).sum() / mask_sum,
                "loss_w_y": weight_w_y * h * ((w_y_gd - w_y) ** 2).sum() / mask_sum,
                "loss_w_z": weight_w_z * h * ((w_z_gd - w_z) ** 2).sum() / mask_sum,
                "loss_boundary": weight_boundary * ((y[:, :, 0,   :,   :] - y_hat[:, :, 0,   :,   :])**2 + 
                                                    (y[:, :, -1,  :,   :] - y_hat[:, :, -1,  :,   :])**2 +
                                                    (y[:, :, :,   0,   :] - y_hat[:, :, :,   0,   :])**2 + 
                                                    (y[:, :, :,   -1,  :] - y_hat[:, :, :,   -1,  :])**2 +
                                                    (y[:, :, :,   :,   0] - y_hat[:, :, :,   :,   0])**2 + 
                                                    (y[:, :, :,   :,  -1] - y_hat[:, :, :,   :,  -1])**2).sum() / mask_sum
               
                
            }  
        elif self.loss_type == "physics_loss":
            # MSE loss including derivatives
            u_x_gd, u_y_gd, u_z_gd, v_x_gd, v_y_gd, v_z_gd, w_x_gd, w_y_gd, w_z_gd = extra_gd
            u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y, w_z = extra
            
            weight_u_x, weight_u_y, weight_u_z = 15, 1, 30
            weight_v_x, weight_v_y, weight_v_z = 50, 30, 5
            weight_w_x, weight_w_y, weight_w_z = 600, 750, 600
            weight_boundary = 5
            weight_solenoidality = 10
            
            h = 2.0 / x.shape[-1]  # Grid spacing
            loss_terms = {
                "loss_u_x": weight_u_x * h * ((u_x_gd - u_x) ** 2).sum() / mask_sum,
                "loss_u_y": weight_u_y * h * ((u_y_gd - u_y) ** 2).sum() / mask_sum,
                "loss_u_z": weight_u_z * h * ((u_z_gd - u_z) ** 2).sum() / mask_sum,
                "loss_v_x": weight_v_x * h * ((v_x_gd - v_x) ** 2).sum() / mask_sum,
                "loss_v_y": weight_v_y * h * ((v_y_gd - v_y) ** 2).sum() / mask_sum,
                "loss_v_z": weight_v_z * h * ((v_z_gd - v_z) ** 2).sum() / mask_sum,
                "loss_w_x": weight_w_x * h * ((w_x_gd - w_x) ** 2).sum() / mask_sum,
                "loss_w_y": weight_w_y * h * ((w_y_gd - w_y) ** 2).sum() / mask_sum,
                "loss_w_z": weight_w_z * h * ((w_z_gd - w_z) ** 2).sum() / mask_sum,
                "loss_boundary": weight_boundary * ((y[:, :, 0,   :,   :] - y_hat[:, :, 0,   :,   :])**2 + 
                                                    (y[:, :, -1,  :,   :] - y_hat[:, :, -1,  :,   :])**2 +
                                                    (y[:, :, :,   0,   :] - y_hat[:, :, :,   0,   :])**2 + 
                                                    (y[:, :, :,   -1,  :] - y_hat[:, :, :,   -1,  :])**2 +
                                                    (y[:, :, :,   :,   0] - y_hat[:, :, :,   :,   0])**2 + 
                                                    (y[:, :, :,   :,  -1] - y_hat[:, :, :,   :,  -1])**2).sum() / mask_sum,                
                "loss_solenoidality": weight_solenoidality * h * (u_x ** 2 + v_y ** 2 + w_z ** 2).sum() / mask_sum
            }
                                              
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")

        total_loss = sum(loss_terms.values()).sum()

        return total_loss, loss_terms

    def training_step(self, batch):
        x, y = batch
        y_hat = self(x)  # Forward pass

        total_loss, loss_terms = self.compute_loss(x, y, y_hat)

        self.log("train_loss", total_loss, prog_bar=True)
        for name, value in loss_terms.items():
            self.log(f"train_{name}", value.mean())

        return total_loss

    def validation_step(self, batch):
        x, y = batch
        y_hat = self(x)

        total_loss, loss_terms = self.compute_loss(x, y, y_hat)

        self.log("val_loss_full", total_loss, on_epoch=True, prog_bar=True)
        for name, value in loss_terms.items():
            self.log(f"val_{name}", value.mean())

        return total_loss

