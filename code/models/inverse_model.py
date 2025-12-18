import torch
import torch.nn as nn
from tomosipo.torch_support import to_autograd

class ReconNet(nn.Module):
    def __init__(self, A, H, W, n_iters=100, alpha=1e-6):
        """
        A: ts.operator(volume_geom, proj_geom)
        H, W: tamaño de la imagen (coincide con vg.shape[1:], p.e. 64x64)
        n_iters: número de iteraciones SIRT
        alpha: paso (relajación)
        """
        super().__init__()

        self.H = H
        self.W = W
        self.n_iters = n_iters

        # Hacemos A y A^T diferenciables para PyTorch
        # is_2d=True porque trabajas en 2D, num_extra_dims=2 (batch, canales)
        self.A_torch  = to_autograd(A,   is_2d=True, num_extra_dims=2)
        self.AT_torch = to_autograd(A.T, is_2d=True, num_extra_dims=2)
        
        # Parámetros aprendibles: un alpha por iteración
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

    def forward(self, y):
        """
        y: sinograma, tensor [B, 1, num_angles, num_detectors]
           (la forma exacta debe ser consistente con cómo usas ts.operator)
        Devuelve: x_recon [B, 1, H, W]
        """
        B = y.shape[0] #batch
        device = y.device
        x = torch.zeros(B, 1, self.H, self.W, device=device)

        for k in range(self.n_iters):
            Ax = self.A_torch(x)
            r = y - Ax
            AT_r = self.AT_torch(r)
            x = x + self.alpha * AT_r

        return x