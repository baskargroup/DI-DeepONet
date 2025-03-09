import torch
import torch.nn as nn
from models.base import BaseLightningModule
from models.geometric_deeponet.network import GeoDeepONet


class GeometricDeepONet(BaseLightningModule):
    def __init__(self, input_channels_func, input_channels_loc, output_channels, branch_net_layers, trunk_net_layers, modes, loss=nn.MSELoss(), lr=1e-3, plot_path='./plots/', log_file='DeepONet_log.txt'):
        super(GeometricDeepONet, self).__init__(lr=lr, plot_path=plot_path, log_file=log_file)
        
        self.input_channels_func = input_channels_func  # Input function channels, e.g., [Re, SDF, mask]
        self.input_channels_loc = input_channels_loc    # Input location channels, e.g., [x, y, z]
        self.output_channels = output_channels          # Output channels, e.g., [u, v, w, p]
        self.branch_net_layers = branch_net_layers      # Layers for the branch network
        self.trunk_net_layers = trunk_net_layers        # Layers for the trunk network
        self.modes = modes                              # Number of modes in the hidden space
        self.loss = loss                                # Loss function, default is MSE
        self.lr = lr                                    # Learning rate
        self.plot_path = plot_path                      # Path for saving plots
        self.log_file = log_file                        # Log file path

        # Initialize the model
        self.model = GeoDeepONet(
            input_channels_func=self.input_channels_func,
            input_channels_loc=self.input_channels_loc,
            output_channels=self.output_channels,
            branch_net_layers=self.branch_net_layers,
            trunk_net_layers=self.trunk_net_layers,
            modes=self.modes
        )

    def forward(self, x):
        """
        Forward pass for the GeometricDeepONet.
        
        Args:
        - x: Input tensor of shape [batch, channels, depth, height, width].
             Channels include function values (e.g., [Re, SDF, mask, x, y, z]).
        
        Returns:
        - Predicted output tensor from the model.
        """
        # Split inputs into function values (x1) and locations (x2)
        x1 = x[:, :self.input_channels_func, :, :, :]  # Function values [Re, SDF, mask]
        x2 = x[:, -3:, :, :, :]  # Grid points [x, y, z]
        
        # Pull the SDF out of the function values and concatenate it with the location inputs
        sdf = x[:, 1, :, :, :].unsqueeze(1)  # Assuming the second channel is SDF
        x2 = torch.cat((x2, sdf), dim=1)     # Add SDF as an additional location channel
        
        # Pass through the model
        return self.model(x1, x2)
