import torch
import torch.nn as nn
from einops import rearrange

class LinearMLP(nn.Module):
    def __init__(self, dims, nonlin):
        super(LinearMLP, self).__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # No activation after the last layer
                layers.append(nonlin())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class DeepONet3D(nn.Module):
    def __init__(self, input_channels_func, input_channels_loc, output_channels, modes, branch_net_layers=None, trunk_net_layers=None):
        super().__init__()
        
        self.input_channels_func = input_channels_func     # [Re, SDF]
        self.input_channels_loc = input_channels_loc       # [x, y, z]
        self.output_channels = output_channels            # [u, v, w]
        self.modes = modes
        
        self.branch_net_layers = [self.input_channels_func] + branch_net_layers + [self.modes * self.output_channels]
        self.branch = LinearMLP(dims=self.branch_net_layers, nonlin=nn.ReLU)
        
        self.trunk_net_layers = [self.input_channels_loc] + trunk_net_layers + [self.modes]
        self.trunk = LinearMLP(dims=self.trunk_net_layers, nonlin=nn.ReLU)
            
        self.b = torch.tensor(0.0, requires_grad=True)

    def forward(self, x1, x2):
        '''
        x1 : input to branch network. 
                [batch_size, num_points, input_channels_func] 
                num_points = (d * h * w) of domain
                input_channels_func is number of input function values. 
                For our case that is (Re, SDF, Mask)
        x2 : input to trunk network.
                [batch_size, num_points, input_channels_loc]
                num_points = (d * h * w) of domain
                input_channels_loc is number of input location values.
                For our case that is (x, y, z).
        
        Reshapes and operations are adapted for 3D domains.
        '''
        
        x1_flattened = rearrange(x1, 'b c d h w -> b (d h w) c')
        x2_flattened = rearrange(x2, 'b c d h w -> b (d h w) c')
        
        output_branch = self.branch(x1_flattened)
        output_trunk = self.trunk(x2_flattened)
        
        # Reshape tensors so branch output is shape [batch, num_pts, modes, output_channels]
        # and trunk output is already shape [batch, num_pts, modes]
        output_branch = rearrange(output_branch, 'b p (m c) -> b p m c', m=self.modes, c=self.output_channels)
        
        # Conduct final dot product over outputs from each network to get predicted solution
        output_solution = torch.einsum('bpmc, bpm -> bpc', output_branch, output_trunk)
        output_solution = output_solution + self.b
        
        # Reshape so output is of shape [batch, c, d, h, w] for loss function in pl.lightning
        output_solution = rearrange(output_solution, 'b (d h w) c -> b c d h w', d=x1.shape[-3], h=x1.shape[-2], w=x1.shape[-1], c=self.output_channels)
        return output_solution
