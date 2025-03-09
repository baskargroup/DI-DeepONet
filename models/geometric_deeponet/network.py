import torch
import torch.nn as nn
from einops import rearrange


class torchSine(nn.Module):
    """Custom sine activation function."""
    def forward(self, x):
        return torch.sin(x)


class LinearMLP(nn.Module):
    """A simple linear MLP with customizable activation functions."""
    def __init__(self, dims, nonlin):
        super(LinearMLP, self).__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation after the last layer
                layers.append(nonlin())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class GeoDeepONet(nn.Module):
    """Geometric DeepONet for 3D data with separate stages for branch and trunk networks."""
    def __init__(self, input_channels_func, input_channels_loc, output_channels, modes, branch_net_layers=None, trunk_net_layers=None):
        super().__init__()

        self.input_channels_func = input_channels_func  # Input function values, e.g., [Re, SDF, Mask]
        self.input_channels_loc = input_channels_loc    # Input location values, e.g., [x, y, z, SDF]
        self.output_channels = output_channels          # Output values, e.g., [u, v, w, p, cd, cl]
        self.modes = modes                              # Number of modes in the hidden space

        # Define branch and trunk networks for both stages
        self.branch_net_layers_stage_1 = [self.input_channels_func] + branch_net_layers + [self.modes]
        self.branch_net_layers_stage_2 = [self.modes] + branch_net_layers + [self.modes * self.output_channels]

        self.trunk_net_layers_stage_1 = [self.input_channels_loc] + trunk_net_layers + [self.modes]
        self.trunk_net_layers_stage_2 = [self.modes] + trunk_net_layers + [self.modes * self.output_channels]

        self.branch_stage_1 = LinearMLP(dims=self.branch_net_layers_stage_1, nonlin=nn.ReLU)
        self.branch_stage_2 = LinearMLP(dims=self.branch_net_layers_stage_2, nonlin=nn.ReLU)

        self.trunk_stage_1 = LinearMLP(dims=self.trunk_net_layers_stage_1, nonlin=nn.ReLU)
        self.trunk_stage_2 = LinearMLP(dims=self.trunk_net_layers_stage_2, nonlin=torchSine)

        self.b = nn.Parameter(torch.tensor(0.0, requires_grad=True))

    def forward(self, x1, x2):
        """
        Forward pass for the GeoDeepONet in 3D.
        
        Args:
        - x1: Input to the branch network, [batch, channels, height, width, depth].
        - x2: Input to the trunk network, [batch, channels, height, width, depth].
        
        Returns:
        - Predicted output, reshaped to [batch, output_channels, height, width, depth].
        """
        # Flatten inputs for MLPs
        x1_flattened = rearrange(x1, 'b c h w d -> b (h w d) c')
        x2_flattened = rearrange(x2, 'b c h w d -> b (h w d) c')

        # Stage 1 outputs
        intermediate_branch = self.branch_stage_1(x1_flattened)  # [batch, num_points, modes]
        intermediate_trunk = self.trunk_stage_1(x2_flattened)    # [batch, num_points, modes]

        # Merge intermediate representations
        merge_intermediate_reps = torch.einsum('bpm, bpm -> bpm', intermediate_branch, intermediate_trunk)

        # Average over points for the branch network
        avg_merged_reps_branch_net = torch.mean(merge_intermediate_reps, dim=1)  # [batch, modes]

        # Stage 2 outputs
        output_branch = self.branch_stage_2(avg_merged_reps_branch_net)  # [batch, modes * output_channels]
        output_trunk = self.trunk_stage_2(merge_intermediate_reps)       # [batch, num_points, modes * output_channels]

        # Reshape outputs
        output_branch = rearrange(output_branch, 'b (m c) -> b m c', m=self.modes, c=self.output_channels)
        output_trunk = rearrange(output_trunk, 'b p (m c) -> b p m c', p=x1_flattened.shape[1], m=self.modes, c=self.output_channels)

        # Dot product and bias addition
        output_solution = torch.einsum('bmc, bpmc -> bpc', output_branch, output_trunk) + self.b

        # Reshape to [batch, output_channels, height, width, depth]
        output_solution = rearrange(output_solution, 'b (h w d) c -> b c h w d', h=x1.shape[-3], w=x1.shape[-2], d=x1.shape[-1])

        return output_solution
