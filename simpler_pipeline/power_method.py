import torch 
import tomosipo as ts
import numpy as np

####################################### Lipschitz constant of A ###################################################
### data
seed = 29072000
H = 100
W = 100
num_angles = 60
i_0 = 10**5
sigma_data = 0.1

# define CT geometry
vg = ts.volume(shape=(1,H,W))                                 # Volume
angles = np.linspace(0, np.pi, num_angles, endpoint=True)     # Angles
pg = ts.parallel(angles=angles, shape=(1, H))                 # Parallel beam structure
A = ts.operator(vg,pg)




def power_method(
    op ,
    maxiter: int = 100,
    tol: float = 1e-6,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Estimate operator norm by power iteration using torch.

    Parameters
    ----------
    op : Operator
        The operator for which to estimate the norm.
    maxiter : int
        Number of power iterations.
    tol : float
        Absolute tolerance for convergence. (TODO: consider adding relative tolerance?)
    device : torch.device | str | None
        Device for computations. If None, use operator's device.

    Returns
    -------
    sigma : torch.Tensor
        Estimated operator norm.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    arr_old = torch.randn(*op.domain_shape, device=device)
    error = tol + 1
    i = 0
    while error >= tol:

        # very verbose and inefficient for now
        omega = op(arr_old)
        alpha = torch.linalg.norm(omega)
        u = (1.0 / alpha) * omega
        z = op.T(u)
        beta = torch.linalg.norm(z)
        arr = (1.0 / beta) * z
        error = torch.linalg.norm(op(arr) - beta * u)
        sigma = beta
        arr_old = arr
        i += 1
        if i >= maxiter:
            return sigma

    return sigma

sigma = power_method(A)
print(f"Lipschitz constant: {sigma}")
#Lipschitz constant: 53.595951080322266


# seed = 29072000
# H = 28
# W = 28
# num_angles = 60
# i_0 = 10**5
# sigma_data = 0.5
#Lipschitz constant: 40.08146667480469


# seed = 29072000
# H = 100
# W = 100
# num_angles = 60
# i_0 = 10**5
# sigma_data = 0.5 or 0.1
#Lipschitz constant: 75.77664947509766

