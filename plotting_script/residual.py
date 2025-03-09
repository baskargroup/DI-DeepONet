import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, List, Tuple


#####################################
#          Boiler plate FEM         #
#####################################

def gauss_pt_eval(tensor: Tensor, N: List[Tensor], nsd: int = 2, stride: int = 1) -> Tensor:
    if nsd == 1:
        conv_gp = F.conv1d
    elif nsd == 2:
        conv_gp = F.conv2d
    elif nsd == 3:
        conv_gp = F.conv3d
    else:
        raise ValueError("nsd must be 1, 2, or 3")

    result_list = []
    for kernel in N:
        result = conv_gp(tensor, kernel, stride=stride)
        result_list.append(result)
    return torch.cat(result_list, dim=1)


class FEMEngine(nn.Module):
    """
    PDE Base Class
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.kwargs = kwargs
        self.nsd = kwargs.get('nsd', 3)

        # For backward compatibility
        self.domain_length = kwargs.get('domain_length', 1.0)
        self.domain_size = kwargs.get('domain_size', 64)
        self.domain_lengths_nd = kwargs.get(
            'domain_lengths',
            (self.domain_length, self.domain_length, self.domain_length)
        )
        self.domain_sizes_nd = kwargs.get(
            'domain_sizes',
            (self.domain_size, self.domain_size, self.domain_size)
        )

        if self.nsd >= 2:
            self.domain_lengthX = self.domain_lengths_nd[0]
            self.domain_lengthY = self.domain_lengths_nd[1]
            self.domain_sizeX = self.domain_sizes_nd[0]
            self.domain_sizeY = self.domain_sizes_nd[1]
            if self.nsd >= 3:
                self.domain_lengthZ = self.domain_lengths_nd[2]
                self.domain_sizeZ = self.domain_sizes_nd[2]
                
        self.ngp_1d = kwargs.get('ngp_1d', 2)
        self.fem_basis_deg = kwargs.get('fem_basis_deg', 1)

        # Gauss quadrature setup
        if self.fem_basis_deg == 1:
            ngp_1d = 2
        elif self.fem_basis_deg in (2, 3):
            ngp_1d = 3
        else:
            raise ValueError("Unsupported fem_basis_deg. Supported degrees: 1, 2, 3.")

        if ngp_1d > self.ngp_1d:
            self.ngp_1d = ngp_1d

        self.ngp_total = self.ngp_1d ** self.nsd
        self.gpx_1d, self.gpw_1d = self.gauss_guadrature_scheme(self.ngp_1d)

        # Element setup
        self.nelemX = (self.domain_sizeX - 1) // self.fem_basis_deg
        self.nelemY = (self.domain_sizeY - 1) // self.fem_basis_deg
        if self.nsd == 3:
            self.nelemZ = (self.domain_sizeZ - 1) // self.fem_basis_deg
        self.nelem = (self.domain_size - 1) // self.fem_basis_deg  # Backward compatibility

        self.hx = self.domain_lengthX / self.nelemX
        self.hy = self.domain_lengthY / self.nelemY
        if self.nsd == 3:
            self.hz = self.domain_lengthZ / self.nelemZ
        self.h = self.domain_length / self.nelem  # Backward compatibility

        # Basis functions setup
        if self.fem_basis_deg == 1:
            # Linear basis functions
            self.nbf_1d = 2
            self.nbf_total = self.nbf_1d ** self.nsd

            self.bf_1d = lambda x: np.array([
                0.5 * (1.0 - x),
                0.5 * (1.0 + x)
            ])
            self.bf_1d_der = lambda x: np.array([
                -0.5,
                0.5
            ])
            self.bf_1d_der2 = lambda x: np.array([
                0.0,
                0.0
            ])

            self.bf_1d_th = lambda x: torch.stack([
                0.5 * (1.0 - x),
                0.5 * (1.0 + x)
            ])
            self.bf_1d_der_th = lambda x: torch.stack([
                -0.5 * torch.ones_like(x),
                0.5 * torch.ones_like(x)
            ])
            self.bf_1d_der2_th = lambda x: torch.stack([
                torch.zeros_like(x),
                torch.zeros_like(x)
            ])

        elif self.fem_basis_deg == 2:
            # Quadratic basis functions
            assert (self.domain_size - 1) % 2 == 0, \
                "For quadratic basis, (domain_size - 1) must be divisible by 2."
            self.nbf_1d = 3
            self.nbf_total = self.nbf_1d ** self.nsd

            self.bf_1d = lambda x: np.array([
                0.5 * x * (x - 1.0),
                1.0 - x ** 2,
                0.5 * x * (x + 1.0)
            ], dtype=float)
            self.bf_1d_der = lambda x: np.array([
                0.5 * (2.0 * x - 1.0),
                -2.0 * x,
                0.5 * (2.0 * x + 1.0)
            ], dtype=float)
            self.bf_1d_der2 = lambda x: np.array([
                1.0,
                -2.0,
                1.0
            ], dtype=float)

            self.bf_1d_th = lambda x: torch.stack([
                0.5 * x * (x - 1.0),
                1.0 - x ** 2,
                0.5 * x * (x + 1.0)
            ])
            self.bf_1d_der_th = lambda x: torch.stack([
                0.5 * (2.0 * x - 1.0),
                -2.0 * x,
                0.5 * (2.0 * x + 1.0)
            ])
            self.bf_1d_der2_th = lambda x: torch.stack([
                torch.ones_like(x),
                -2.0 * torch.ones_like(x),
                torch.ones_like(x)
            ])

        elif self.fem_basis_deg == 3:
            # Cubic basis functions
            assert (self.domain_size - 1) % 3 == 0, \
                "For cubic basis, (domain_size - 1) must be divisible by 3."
            self.nbf_1d = 4
            self.nbf_total = self.nbf_1d ** self.nsd

            self.bf_1d = lambda x: np.array([
                (-9.0 / 16.0) * (x ** 3 - x ** 2 - (1.0 / 9.0) * x + (1.0 / 9.0)),
                (27.0 / 16.0) * (x ** 3 - (1.0 / 3.0) * x ** 2 - x + (1.0 / 3.0)),
                (-27.0 / 16.0) * (x ** 3 + (1.0 / 3.0) * x ** 2 - x - (1.0 / 3.0)),
                (9.0 / 16.0) * (x ** 3 + x ** 2 - (1.0 / 9.0) * x - (1.0 / 9.0))
            ], dtype=float)
            self.bf_1d_der = lambda x: np.array([
                (-9.0 / 16.0) * (3.0 * x ** 2 - 2.0 * x - (1.0 / 9.0)),
                (27.0 / 16.0) * (3.0 * x ** 2 - (2.0 / 3.0) * x - 1.0),
                (-27.0 / 16.0) * (3.0 * x ** 2 + (2.0 / 3.0) * x - 1.0),
                (9.0 / 16.0) * (3.0 * x ** 2 + 2.0 * x - (1.0 / 9.0))
            ], dtype=float)

            self.bf_1d_der2 = lambda x: np.array([
                (-9.0 / 16.0) * (6.0 * x - 2.0),
                (27.0 / 16.0) * (6.0 * x - (2.0 / 3.0)),
                (-27.0 / 16.0) * (6.0 * x + (2.0 / 3.0)),
                (9.0 / 16.0) * (6.0 * x + 2.0)
            ], dtype=float)
                
                
    def gauss_guadrature_scheme(self, ngp_1d: int) -> Tuple[np.ndarray, np.ndarray]:
        if ngp_1d == 1:
            gpx_1d = np.array([0.0])
            gpw_1d = np.array([2.0])
        elif ngp_1d == 2:
            gpx_1d = np.array([-0.5773502691896258, 0.5773502691896258])
            gpw_1d = np.array([1.0, 1.0])
        elif ngp_1d == 3:
            gpx_1d = np.array([-0.774596669, 0.0, 0.774596669])
            gpw_1d = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
        elif ngp_1d == 4:
            gpx_1d = np.array([-0.861136, -0.339981, 0.339981, 0.861136])
            gpw_1d = np.array([0.347855, 0.652145, 0.652145, 0.347855])
        else:
            raise ValueError("Unsupported number of Gauss points per dimension.")
        return gpx_1d, gpw_1d

    def gauss_pt_evaluation(self, tensor: Tensor, stride: int = 1) -> Tensor:
        return gauss_pt_eval(tensor, self.N_gp, nsd=self.nsd, stride=(self.nbf_1d - 1))

    def gauss_pt_evaluation_surf(self, tensor: Tensor, stride: int = 1) -> Tensor:
        return gauss_pt_eval(
            tensor, self.N_gp_surf, nsd=(self.nsd - 1), stride=(self.nbf_1d - 1)
        )

    def gauss_pt_evaluation_der_x(self, tensor: Tensor, stride: int = 1) -> Tensor:
        return gauss_pt_eval(tensor, self.dN_x_gp, nsd=self.nsd, stride=(self.nbf_1d - 1))

    def gauss_pt_evaluation_der_y(self, tensor: Tensor, stride: int = 1) -> Tensor:
        return gauss_pt_eval(tensor, self.dN_y_gp, nsd=self.nsd, stride=(self.nbf_1d - 1))

    def gauss_pt_evaluation_der_z(self, tensor: Tensor, stride: int = 1) -> Tensor:
        return gauss_pt_eval(tensor, self.dN_z_gp, nsd=self.nsd, stride=(self.nbf_1d - 1))

    def gauss_pt_evaluation_der2_x(self, tensor: Tensor, stride: int = 1) -> Tensor:
        return gauss_pt_eval(
            tensor, self.d2N_x_gp, nsd=self.nsd, stride=(self.nbf_1d - 1)
        )

    def gauss_pt_evaluation_der2_y(self, tensor: Tensor, stride: int = 1) -> Tensor:
        return gauss_pt_eval(
            tensor, self.d2N_y_gp, nsd=self.nsd, stride=(self.nbf_1d - 1)
        )

    def gauss_pt_evaluation_der2_z(self, tensor: Tensor, stride: int = 1) -> Tensor:
        return gauss_pt_eval(
            tensor, self.d2N_z_gp, nsd=self.nsd, stride=(self.nbf_1d - 1)
        )

    def gauss_pt_evaluation_der2_xy(self, tensor: Tensor, stride: int = 1) -> Tensor:
        return gauss_pt_eval(
            tensor, self.d2N_xy_gp, nsd=self.nsd, stride=(self.nbf_1d - 1)
        )

    def gauss_pt_evaluation_der2_yz(self, tensor: Tensor, stride: int = 1) -> Tensor:
        return gauss_pt_eval(
            tensor, self.d2N_yz_gp, nsd=self.nsd, stride=(self.nbf_1d - 1)
        )

    def gauss_pt_evaluation_der2_zx(self, tensor: Tensor, stride: int = 1) -> Tensor:
        return gauss_pt_eval(
            tensor, self.d2N_zx_gp, nsd=self.nsd, stride=(self.nbf_1d - 1)
        )       
                
                

    def forward(self, *args: Tensor, **kwargs: Any) -> Tuple[Tensor, ...]:
        """
        Forward pass for FEMEngine.

        This method accepts an arbitrary number of input tensors and returns an arbitrary
        number of output tensors. It serves as a placeholder and should be implemented
        with the specific FEM computations during inference.

        Args:
            *args (Tensor): Variable length input tensor list representing the state or parameters.
            **kwargs (Any): Arbitrary keyworded input kwargs.

        Returns:
            Tuple[Tensor, ...]: A tuple of output tensors after FEM computations.
        """
        pass
    
    


class FEM3D(FEMEngine):
    """3D Finite Element Method Engine with Differentiable Components"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        assert self.nsd == 3, "FEM3D is designed for 3D problems only."
        self.gpw = torch.zeros(self.ngp_total)
        self.N_gp = nn.ParameterList() 
        self.dN_x_gp = nn.ParameterList()
        self.dN_y_gp = nn.ParameterList()
        self.dN_z_gp = nn.ParameterList()
        self.d2N_x_gp = nn.ParameterList()
        self.d2N_y_gp = nn.ParameterList() 
        self.d2N_z_gp = nn.ParameterList() 
        self.d2N_xy_gp = nn.ParameterList()
        self.d2N_yz_gp = nn.ParameterList()
        self.d2N_zx_gp = nn.ParameterList()
        self.Nvalues = torch.ones((1,self.nbf_total,self.ngp_total,1,1,1))
        self.dN_x_values = torch.ones((1,self.nbf_total,self.ngp_total,1,1,1))
        self.dN_y_values = torch.ones((1,self.nbf_total,self.ngp_total,1,1,1))
        self.dN_z_values = torch.ones((1,self.nbf_total,self.ngp_total,1,1,1))
        self.d2N_x_values = torch.ones((1,self.nbf_total,self.ngp_total,1,1,1))
        self.d2N_y_values = torch.ones((1,self.nbf_total,self.ngp_total,1,1,1))
        self.d2N_z_values = torch.ones((1,self.nbf_total,self.ngp_total,1,1,1))
        for kgp in range(self.ngp_1d):
            for jgp in range(self.ngp_1d):
                for igp in range(self.ngp_1d):
                    N_gp = torch.zeros((self.nbf_1d, self.nbf_1d, self.nbf_1d))
                    dN_x_gp = torch.zeros((self.nbf_1d, self.nbf_1d, self.nbf_1d))
                    dN_y_gp = torch.zeros((self.nbf_1d, self.nbf_1d, self.nbf_1d))
                    dN_z_gp = torch.zeros((self.nbf_1d, self.nbf_1d, self.nbf_1d))
                    d2N_x_gp = torch.zeros((self.nbf_1d, self.nbf_1d, self.nbf_1d))
                    d2N_y_gp = torch.zeros((self.nbf_1d, self.nbf_1d, self.nbf_1d))
                    d2N_z_gp = torch.zeros((self.nbf_1d, self.nbf_1d, self.nbf_1d))
                    d2N_xy_gp = torch.zeros((self.nbf_1d, self.nbf_1d, self.nbf_1d))
                    d2N_yz_gp = torch.zeros((self.nbf_1d, self.nbf_1d, self.nbf_1d))
                    d2N_zx_gp = torch.zeros((self.nbf_1d, self.nbf_1d, self.nbf_1d))

                    IGP = kgp * self.ngp_1d**2 + jgp * self.ngp_1d + igp # tensor product id or the linear id of the gauss point
                    self.gpw[IGP] = self.gpw_1d[igp] * self.gpw_1d[jgp] * self.gpw_1d[kgp]

                    for kbf in range(self.nbf_1d):
                        for jbf in range(self.nbf_1d):
                            for ibf in range(self.nbf_1d):
                                IBF = kbf * self.nbf_1d**2 + jbf * self.nbf_1d + ibf
                                N_gp[kbf,jbf,ibf] = self.bf_1d(self.gpx_1d[igp])[ibf] * self.bf_1d(self.gpx_1d[jgp])[jbf] * self.bf_1d(self.gpx_1d[kgp])[kbf]
                                dN_x_gp[kbf,jbf,ibf] = self.bf_1d_der(self.gpx_1d[igp])[ibf] * self.bf_1d(self.gpx_1d[jgp])[jbf] * self.bf_1d(self.gpx_1d[kgp])[kbf] * (2 / self.hx)
                                dN_y_gp[kbf,jbf,ibf] = self.bf_1d(self.gpx_1d[igp])[ibf] * self.bf_1d_der(self.gpx_1d[jgp])[jbf] * self.bf_1d(self.gpx_1d[kgp])[kbf] * (2 / self.hy)
                                dN_z_gp[kbf,jbf,ibf] = self.bf_1d(self.gpx_1d[igp])[ibf] * self.bf_1d(self.gpx_1d[jgp])[jbf] * self.bf_1d_der(self.gpx_1d[kgp])[kbf] * (2 / self.hz)
                                d2N_x_gp[ibf,jbf,kbf] = self.bf_1d_der2(self.gpx_1d[igp])[ibf] * self.bf_1d(self.gpx_1d[jgp])[jbf] * self.bf_1d(self.gpx_1d[kgp])[kbf] * (2 / self.hx)**2
                                d2N_y_gp[ibf,jbf,kbf] = self.bf_1d(self.gpx_1d[igp])[ibf] * self.bf_1d_der2(self.gpx_1d[jgp])[jbf] * self.bf_1d(self.gpx_1d[kgp])[kbf] * (2 / self.hy)**2
                                d2N_z_gp[ibf,jbf,kbf] = self.bf_1d(self.gpx_1d[igp])[ibf] * self.bf_1d(self.gpx_1d[jgp])[jbf] * self.bf_1d_der2(self.gpx_1d[kgp])[kbf] * (2 / self.hz)**2
                                d2N_xy_gp[ibf,jbf,kbf] = self.bf_1d_der(self.gpx_1d[igp])[ibf] * self.bf_1d_der(self.gpx_1d[jgp])[jbf] * self.bf_1d(self.gpx_1d[kgp])[kbf] * (2 / self.hx) * (2 / self.hy)
                                d2N_yz_gp[ibf,jbf,kbf] = self.bf_1d(self.gpx_1d[igp])[ibf] * self.bf_1d_der(self.gpx_1d[jgp])[jbf] * self.bf_1d_der(self.gpx_1d[kgp])[kbf] * (2 / self.hy) * (2 / self.hz)
                                d2N_zx_gp[ibf,jbf,kbf] = self.bf_1d_der(self.gpx_1d[igp])[ibf] * self.bf_1d(self.gpx_1d[jgp])[jbf] * self.bf_1d_der(self.gpx_1d[kgp])[kbf] * (2 / self.hz) * (2 / self.hx)
                                self.Nvalues[0,IBF,IGP,:,:,:] = N_gp[kbf,jbf,ibf]
                                self.dN_x_values[0,IBF,IGP,:,:,:] = dN_x_gp[kbf,jbf,ibf]
                                self.dN_y_values[0,IBF,IGP,:,:,:] = dN_y_gp[kbf,jbf,ibf]
                                self.dN_z_values[0,IBF,IGP,:,:,:] = dN_z_gp[kbf,jbf,ibf]
                                self.d2N_x_values[0,IBF,IGP,:,:,:] = d2N_x_gp[kbf,jbf,ibf]
                                self.d2N_y_values[0,IBF,IGP,:,:,:] = d2N_y_gp[kbf,jbf,ibf]
                                self.d2N_z_values[0,IBF,IGP,:,:,:] = d2N_z_gp[kbf,jbf,ibf]

                    self.N_gp.append(nn.Parameter(N_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                    self.dN_x_gp.append(nn.Parameter(dN_x_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                    self.dN_y_gp.append(nn.Parameter(dN_y_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                    self.dN_z_gp.append(nn.Parameter(dN_z_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                    self.d2N_x_gp.append(nn.Parameter(d2N_x_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                    self.d2N_y_gp.append(nn.Parameter(d2N_y_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                    self.d2N_z_gp.append(nn.Parameter(d2N_x_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                    self.d2N_xy_gp.append(nn.Parameter(d2N_xy_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                    self.d2N_yz_gp.append(nn.Parameter(d2N_yz_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                    self.d2N_zx_gp.append(nn.Parameter(d2N_zx_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))

        x = np.linspace(0,self.domain_lengthX,self.domain_sizeX)
        y = np.linspace(0,self.domain_lengthY,self.domain_sizeY)
        z = np.linspace(0,self.domain_lengthZ,self.domain_sizeZ)
        M = x.shape[0]
        N = y.shape[0]
        P = z.shape[0]
        x_2d, y_2d = np.meshgrid(x, y)
        xx = np.tile(x_2d, (P,1,1))
        yy = np.tile(y_2d, (P,1,1))
        zz = np.reshape(np.repeat(z, (N*M)), (P,N,M))
        self.xx = torch.FloatTensor(xx)
        self.yy = torch.FloatTensor(yy)
        self.zz = torch.FloatTensor(zz)
        self.xgp = self.gauss_pt_evaluation(self.xx.unsqueeze(0).unsqueeze(0))
        self.ygp = self.gauss_pt_evaluation(self.yy.unsqueeze(0).unsqueeze(0))
        self.zgp = self.gauss_pt_evaluation(self.zz.unsqueeze(0).unsqueeze(0))





#####################################
#          LDC NS Residual          #
#####################################


class ResidualLoss(FEM3D):
    '''
    Class for computing the residual of LDC-NS.
    '''
    def __init__(self, domain_size, device, **kwargs):
        super().__init__(**kwargs)
        self.domain_size = domain_size
        self.hx = 2 / self.domain_size
        self.hy = 2 / self.domain_size
        self.hz = 2 / self.domain_size
        self.device = device
        
    def forward(self, dataX, dataY, Y_true):
        '''
        args:
            dataX: SDF or Mask and Re
            dataY: Field solution (u, v, p)
        '''
        r = dataX[:,0:1,...]        
        s = dataX[:,1:2,...]  # SDF (Signed Distance Function)
        u = dataY[:,0:1,...]
        v = dataY[:,1:2,...]
        w = dataY[:,2:3,...]
        utrue = Y_true[:,0:1,...]
        vtrue = Y_true[:,1:2,...]
        wtrue = Y_true[:,2:3,...]

        # Apply mask where SDF < 0 (inside the geometry)
        mask = (s > 0).float()  # Assuming s is SDF, mask outside geometry

        # Apply mask to predicted fields
        u = u * mask
        v = v * mask
        w = w * mask
        utrue = utrue * mask
        vtrue = vtrue * mask
        wtrue = wtrue * mask
        
        gpw = self.gpw
        
        trnsfrm_jac = (0.5*self.hx)*(0.5*self.hy)*(0.5*self.hz)
        JxW = (gpw*trnsfrm_jac).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(self.device)
        JxW = JxW.unsqueeze(-1)

        trnsfrm_jac2 = (0.5)*(0.5)*(0.5)
        JxW2 = (gpw*trnsfrm_jac2).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(self.device)
        JxW2 = JxW2.unsqueeze(-1)
                
        u_x_gp = self.gauss_pt_evaluation_der_x(u)
        u_y_gp = self.gauss_pt_evaluation_der_y(u)
        u_z_gp = self.gauss_pt_evaluation_der_z(u)
        v_x_gp = self.gauss_pt_evaluation_der_x(v)
        v_y_gp = self.gauss_pt_evaluation_der_y(v)
        v_z_gp = self.gauss_pt_evaluation_der_z(v)
        w_x_gp = self.gauss_pt_evaluation_der_x(w)
        w_y_gp = self.gauss_pt_evaluation_der_y(w)
        w_z_gp = self.gauss_pt_evaluation_der_z(w)
        utrue_x_gp = self.gauss_pt_evaluation_der_x(utrue)
        utrue_y_gp = self.gauss_pt_evaluation_der_y(utrue)
        utrue_z_gp = self.gauss_pt_evaluation_der_z(utrue)
        vtrue_x_gp = self.gauss_pt_evaluation_der_x(vtrue)
        vtrue_y_gp = self.gauss_pt_evaluation_der_y(vtrue)
        vtrue_z_gp = self.gauss_pt_evaluation_der_z(vtrue)
        wtrue_x_gp = self.gauss_pt_evaluation_der_x(wtrue)
        wtrue_y_gp = self.gauss_pt_evaluation_der_y(wtrue)
        wtrue_z_gp = self.gauss_pt_evaluation_der_z(wtrue)

        pred_x_gp = u_x_gp**2 + v_x_gp**2 + w_x_gp**2
        pred_y_gp = u_y_gp**2 + v_y_gp**2 + w_y_gp**2
        pred_z_gp = u_z_gp**2 + v_z_gp**2 + w_z_gp**2
        true_x_gp = utrue_x_gp**2 + vtrue_x_gp**2 + wtrue_x_gp**2
        true_y_gp = utrue_y_gp**2 + vtrue_y_gp**2 + wtrue_y_gp**2
        true_z_gp = utrue_z_gp**2 + vtrue_z_gp**2 + wtrue_z_gp**2
        error_x_gp = (u_x_gp - utrue_x_gp)**2 + (v_x_gp - vtrue_x_gp)**2 + (w_x_gp - wtrue_x_gp)**2
        error_y_gp = (u_y_gp - utrue_y_gp)**2 + (v_y_gp - vtrue_y_gp)**2 + (w_y_gp - wtrue_y_gp)**2
        error_z_gp = (u_z_gp - utrue_z_gp)**2 + (v_z_gp - vtrue_z_gp)**2 + (w_z_gp - wtrue_z_gp)**2
        pred_x_elm_gp = pred_x_gp * JxW2
        pred_y_elm_gp = pred_y_gp * JxW2
        pred_z_elm_gp = pred_z_gp * JxW2
        true_x_elm_gp = true_x_gp * JxW2
        true_y_elm_gp = true_y_gp * JxW2
        true_z_elm_gp = true_z_gp * JxW2
        error_x_elm_gp = error_x_gp * JxW2
        error_y_elm_gp = error_y_gp * JxW2
        error_z_elm_gp = error_z_gp * JxW2
        pred_x_elm = torch.sum(pred_x_elm_gp, 1)
        pred_y_elm = torch.sum(pred_y_elm_gp, 1)
        pred_z_elm = torch.sum(pred_z_elm_gp, 1)
        true_x_elm = torch.sum(true_x_elm_gp, 1)
        true_y_elm = torch.sum(true_y_elm_gp, 1)
        true_z_elm = torch.sum(true_z_elm_gp, 1)
        error_x_elm = torch.sum(error_x_elm_gp, 1)
        error_y_elm = torch.sum(error_y_elm_gp, 1)
        error_z_elm = torch.sum(error_z_elm_gp, 1)
        
        u_x_elm_gp_integral = u_x_gp * JxW
        v_y_elm_gp_integral = v_y_gp * JxW
        w_z_elm_gp_integral = w_z_gp * JxW

        u_x_elm_integral = torch.sum(u_x_elm_gp_integral, 1)
        v_y_elm_integral = torch.sum(v_y_elm_gp_integral, 1)
        w_z_elm_integral = torch.sum(w_z_elm_gp_integral, 1)
        solenoidality_elm_integral =  u_x_elm_integral + v_y_elm_integral + w_z_elm_integral
        solenoidality_Total = ((torch.sum(torch.sum(torch.sum(solenoidality_elm_integral**2, -1), -1),-1))**0.5).mean()

        error_u_x_total_squared_integral_sqrt = (torch.sum(torch.sum(torch.sum((u_x_gp - utrue_x_gp)**2, -1), -1),-1)/
                                                 torch.sum(torch.sum(torch.sum(utrue_x_gp**2, -1), -1),-1))**0.5
        
        error_u_y_total_squared_integral_sqrt = (torch.sum(torch.sum(torch.sum((u_y_gp - utrue_y_gp)**2, -1), -1),-1)/
                                                 torch.sum(torch.sum(torch.sum(utrue_y_gp**2, -1), -1),-1))**0.5
                
        error_u_z_total_squared_integral_sqrt = (torch.sum(torch.sum(torch.sum((u_z_gp - utrue_z_gp)**2, -1), -1),-1)/
                                                 torch.sum(torch.sum(torch.sum(utrue_z_gp**2, -1), -1),-1))**0.5

        error_v_x_total_squared_integral_sqrt = (torch.sum(torch.sum(torch.sum((v_x_gp - vtrue_x_gp)**2, -1), -1),-1)/
                                                 torch.sum(torch.sum(torch.sum(vtrue_x_gp**2, -1), -1),-1))**0.5
        
        error_v_y_total_squared_integral_sqrt = (torch.sum(torch.sum(torch.sum((v_y_gp - vtrue_y_gp)**2, -1), -1),-1)/
                                                 torch.sum(torch.sum(torch.sum(vtrue_y_gp**2, -1), -1),-1))**0.5
                
        error_v_z_total_squared_integral_sqrt = (torch.sum(torch.sum(torch.sum((v_z_gp - vtrue_z_gp)**2, -1), -1),-1)/
                                                 torch.sum(torch.sum(torch.sum(vtrue_z_gp**2, -1), -1),-1))**0.5        

        error_w_x_total_squared_integral_sqrt = (torch.sum(torch.sum(torch.sum((w_x_gp - wtrue_x_gp)**2, -1), -1),-1)/
                                                 torch.sum(torch.sum(torch.sum(wtrue_x_gp**2, -1), -1),-1))**0.5
        
        error_w_y_total_squared_integral_sqrt = (torch.sum(torch.sum(torch.sum((w_y_gp - wtrue_y_gp)**2, -1), -1),-1)/
                                                 torch.sum(torch.sum(torch.sum(wtrue_y_gp**2, -1), -1),-1))**0.5
                
        error_w_z_total_squared_integral_sqrt = (torch.sum(torch.sum(torch.sum((w_z_gp - wtrue_z_gp)**2, -1), -1),-1)/
                                                 torch.sum(torch.sum(torch.sum(wtrue_z_gp**2, -1), -1),-1))**0.5
        
        deriv_error_Total = ((1.0/9.0)*(error_u_x_total_squared_integral_sqrt + error_u_y_total_squared_integral_sqrt + error_u_z_total_squared_integral_sqrt
                              + error_v_x_total_squared_integral_sqrt + error_v_y_total_squared_integral_sqrt + error_v_z_total_squared_integral_sqrt
                              + error_w_x_total_squared_integral_sqrt + error_w_y_total_squared_integral_sqrt + error_w_z_total_squared_integral_sqrt)).mean() 
 
        
        return deriv_error_Total, solenoidality_Total, error_x_elm, error_y_elm, error_z_elm, pred_x_elm, pred_y_elm, pred_z_elm, true_x_elm, true_y_elm, true_z_elm
