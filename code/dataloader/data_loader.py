#from torchvision.datasets import FashionMNIST
from torchvision.datasets import MNIST
from torch.utils.data import Dataset
from torchvision import transforms
from code.ct_recon.SIRT import normalization_matrices, sirt_method
import numpy as np
import tomosipo as ts
import random
import torch

"""
FashionMNIST CT Dataset — Parallel-Beam Tomography
------------------------------------------

This dataset generates simple 2D CT phantoms from FashionMNIST images and simulates
their parallel-beam projections with realistic noise:

1. Loads FashionMNIST images (train or test split).
2. Embeds each digit in a larger zero-padded canvas to form a 2D phantom.
3. Applies random translations so clothes appear at different positions.
   - Optionally allows the cloth to be partially outside the canvas (`cut=True`).
4. Uses an external tomosipo operator `A` to compute parallel-beam sinograms.
5. Adds Poisson (photon counting) + Gaussian (detector) noise to the sinograms.
6. Returns a dictionary with:
       - noisy_sinogram: noisy projection data
       - ground_truth_image: clean padded MNIST phantom
       - label: original FashionMNIST class (0–9)
       - center: FashionMNIST center coordinates in the canvas

"""

class FashionMnistDataset(Dataset):

    def __init__(self, training, testing, A, dimension=(50, 50), num_angles=60, sigma = 0.1, i_0 = 10**5, constant = 10**2, noisy_images=False,
                sigma_image=0.1, sirt_images = False, sirt_iterations = 30, sirt_alpha = 1/(53.595951080322266)**2,cut = False,  seed = 29072000, max_len = None,  device="cuda"):
        """
        Initialize the FashionMNIST CT dataset.

        Parameters
        ----------
        training : bool
            If True, use the FashionMNIST training split.
        testing : bool
            If True, use the FashionMNIST test split.
            Exactly one of (training, testing) must be True.
        A : tomosipo.Operator
            Tomosipo forward operator implementing the ray transform
            (e.g. parallel-beam projections) that maps images -> sinograms.
            It is assumed to be compatible with tensors living on `device`.
        dimension : tuple[int, int], optional
            Spatial size (H, W) of the padded phantom image.
            The FashionMNIST digit is embedded into a canvas of this size.
        num_angles : int, optional
            Number of projection angles used by the CT geometry.
            (Stored for bookkeeping; the geometry is encoded in `A`.)
        sigma : float, optional
            Standard deviation of the additive Gaussian detector noise
            applied on top of the Poisson log-domain sinogram.
        standard_dose : bool, optional
            If True, use a standard incident photon count I0 = 1e5.
            If False, reduce dose (currently I0 / 4).
        cut : bool, optional
            Controls whether the FashionMNIST digit is always fully contained
            in the canvas or allowed to be partially cut off:
            - False: top-left corner is restricted so that the whole digit
              lies inside the (H, W) canvas (no cropping).
            - True: the top-left corner is sampled anywhere in the canvas,
              so parts of the digit may fall outside and be cropped.
        seed : int, optional
            Base random seed (currently not used inside the class; global
            seeding is expected to be handled outside for reproducibility).
        max_len : int or None, optional
            If not None, cap the dataset length to `max_len` samples.
        device : str or torch.device, optional
            Target device for tensors produced by the dataset
            (e.g. "cuda", "cuda:0", or "cpu").
        """
        self.training = training
        self.testing = testing

        if self.training and self.testing:
            raise ValueError(
                "You cannot be training and testing with the same dataset. "
                "Set exactly one of (training, testing) to True."
            )
        if self.training == self.testing:
            raise ValueError(
                "Exactly one of training and testing must be True."
            )
        
        self.dimension = dimension
        self.H, self.W = self.dimension
        self.num_angles = num_angles
        self.sigma = sigma
        self.i_0 = i_0
        self.constant = constant

        # for models training (noisy or sirt images)
        self.noisy_images = noisy_images
        self.sigma_image = sigma_image

        #checking sirt_iterations is a valid value
        if isinstance(sirt_iterations, int):
            if sirt_iterations < 1:
                raise ValueError("sirt_iterations must be positive")

        elif isinstance(sirt_iterations, list):
            if not sirt_iterations or any(
                not isinstance(t, int) or t < 1 for t in sirt_iterations
            ):
                raise ValueError("The list should contain positive intergers")

        else:
            raise TypeError("Use a interger or a list")

        self.sirt_images = sirt_images
        self.sirt_iterations = sirt_iterations
        self.sirt_alpha = sirt_alpha

        

        self.cut = cut
        self.seed = seed
        self.max_len = max_len
        self.device = torch.device(device)

        #load mnist dataset 
        transform = transforms.ToTensor()

        if self.training:
            self.data = MNIST(root="data", train=True, download=True, transform=transform)
            #self.data = FashionMNIST(root="data", train=True, download=True, transform=transform)
        elif self.testing:
            self.data = MNIST(root="data", train=False, download=True, transform=transform)
            #self.data = FashionMNIST(root="data", train=False, download=True, transform=transform)

        # Create tomosipo volume and projection geometry
        self.A = A   
        self.AT = A.T

        #normalization matrices
        self.M ,self.D = normalization_matrices(self.A, self.AT, 1, self.H, self.W, self.num_angles, self.device)

    
    def __len__(self):
        """
        Returns the number of samples in the dataset.

        If max_len is set, it caps the length to that value.
        """
        if self.max_len is not None:
            return min(self.max_len, len(self.data))
        return len(self.data)


    def __repr__(self):
        """
        Returns a string summary of the dataset configuration.

        Returns:
            str: 
                Dataset description.
        """

        return (f"MnistDataset(num_samples={self.max_len}, "
                f"Training={self.training}, "
                f"noise=Poisson+Gaussian(σ={self.sigma}), "
                f"i_0={self.i_0})")


    def max_normalize(self, tensor):
        """
        Scales a tensor to the range [0, 1] using max normalization.

        Args:
            tensor (torch.Tensor): 
                Input tensor.

        Returns:
            tuple: 
                (normalized tensor, original max value)
        """
        # calculatin max from tensor
        max_val = tensor.max()

        #Normalising
        norm = tensor  / max_val 
        return norm, max_val
    


    def max_denormalize(self, norm_tensor, max_val):
        """
        Reverses max normalization to restore original scale.

        Args:
            norm_tensor (torch.Tensor): 
                Normalized tensor.
            max_val (float): 
                Max value used in normalization.

        Returns:
            torch.Tensor: 
                Restored tensor.
        """
        return norm_tensor * max_val  
    
    
    def noise(self, sinogram, g):
        """
        Add realistic Poisson + Gaussian noise to a sinogram.

        Parameters:
            sinogram (torch.Tensor):
                Clean input sinogram before noise.
            idx (int):
                Index of the sample, used to derive a reproducible seed.

        Returns:
            torch.Tensor:
                Noisy sinogram tensor with the same shape as the input.

        Notes:
            - Poisson noise simulates X-ray photon counting statistics.
            - Gaussian noise models additional detector/system imperfections.
        """
        # Nomalised sinogram between [0,1] to add noise
        norm_sino,  max_val =  self.max_normalize(sinogram)

       
        # Simulate measured photons using Poisson noise (counting error)
        measured_photons = torch.poisson(self.i_0 * torch.exp(-norm_sino))

        # Avoid log(0) by setting the minimum to 1
        measured_photons = torch.clamp(measured_photons, min=1.0)

        # Convert back to log domain (using Beer–Lambert Law)
        noisy_sinogram = -torch.log(measured_photons / self.i_0)

        # Adding gaussian noise (detector's imperfections)
        gaussian_noise = torch.randn(sinogram.shape, device=sinogram.device, dtype=sinogram.dtype, generator=g) * self.sigma

        # Final result 
        noisy_sinogram += gaussian_noise

        # reaply the normalization
        noisy_sinogram = self.max_denormalize(noisy_sinogram, max_val)

        # no negative 
        noisy_sinogram = torch.clamp(noisy_sinogram, min=0)

        return noisy_sinogram
    
    def padding_and_translation(self,img, g):
        """
        Embed a FashionMNIST digit into a larger zero-padded canvas with translation.

        Parameters:
            img (torch.Tensor):
                FashionMNIST image tensor of shape (1, h, w).
            idx (int):
                Sample index, used to generate a reproducible translation.

        Returns:
            tuple:
                center (tuple[int, int]):
                    (x_center, y_center) of the digit in the canvas.
                im_prim (torch.Tensor):
                    Padded image tensor of shape (1, H, W) with the digit placed.
        """
        #original image dimension
        _, h, w = img.shape

        #creating 0 image 
        im_prim = torch.zeros((1, self.H, self.W), device=self.device)

        ###### ESTO ESTA SIN IMPLEMENTAR #####
        if self.cut:
            # poistion of the top-left corner of mnist
            x = torch.randint(0, self.H, (1,), generator=g, device=self.device).item()
            y = torch.randint(0, self.W, (1,), generator=g, device=self.device).item()


            #place the mnist number on the new image
            im_prim[:, x:x+h, y:y+w] = img

            #calculate the coordinates of the mnist image on the new canvas
            center = (x+round(h/2),y+round(w/2))

        else: 
            # poistion of the top-left corner of mnist
            x = torch.randint(0, self.H - h + 1, (1,), generator=g, device=self.device).item()
            y = torch.randint(0, self.W - w + 1, (1,), generator=g, device=self.device).item()


            #place the mnist number on the new image
            im_prim[:, x:x+h, y:y+w] = img

            #calculate the coordinates of the mnist image on the new canvas
            center = (x+round(h/2),y+round(w/2))

        return center,im_prim
    
    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.

        Parameters:
            idx (int):
                Index of the sample to retrieve.

        Returns:
            tuple:
                noisy_sino (torch.Tensor):
                    Noisy sinogram corresponding to the padded phantom.
                im_prim (torch.Tensor):
                    Ground-truth padded image (phantom) of shape (1, H, W).
                label (int):
                    Original FashionMNIST class label (digit identity).
                center (tuple[int, int]):
                    Center coordinates of the digit in the padded canvas.
        """
        g = torch.Generator(device=self.device)
        g.manual_seed(self.seed + int(idx))

        #get image and label
        img, label = self.data[idx]
        img = img.to(self.device)

        #padding and translation
        center, im_prim = self.padding_and_translation(img, g)

        #sinogram
        sinogram = self.A(im_prim)

        # adding noise
        noisy_sino = self.noise(sinogram, g) 

        # for training models, noisy images
        ground_truth_small = img.clone()


        
        # true noisy image
        if self.noisy_images:
            #if sigma_images is a list, we randomdly select a noise level 
            if isinstance(self.sigma_image, (tuple, list)):
                sigma = torch.empty(1).uniform_(
                    self.sigma_image[0],
                    self.sigma_image[1]
                ).item()
            else:
                # if sigma is a number all images have the same gaussian noise
                sigma = float(self.sigma_image)

            # adding gaussian noise
            noisy_ground_truth_small = (ground_truth_small + torch.randn_like(ground_truth_small) * sigma).clamp(0, 1)
        
        else:
            noisy_ground_truth_small = None

        # sirt image
        if self.sirt_images:
            #if sirt_iterations is a tuple/list we should return the iteration and  
            x = torch.zeros(1, 1, self.H, self.W, device=self.device)
            sirt_recon = []
            if isinstance(self.sirt_iterations, int):
                iters = self.sirt_iterations
                save_iterations = []

            elif isinstance(self.sirt_iterations, list):
                iters = max(self.sirt_iterations)
                save_iterations = list(self.sirt_iterations)

            for i in range(iters):
                x, _ = sirt_method(self.A, self.AT, x, noisy_sino, self.D, self.M, self.sirt_alpha)
                if i in save_iterations or i == iters-1:
                    sirt_recon.append(x.clone())

        else:
            sirt_recon = None
        




        return {
            'mnist_image': img,
            'noisy_ground_truth_image': noisy_ground_truth_small,
            "sirt_reconstructions": sirt_recon,
            'ground_truth_small': ground_truth_small,
            'noisy_sinogram': noisy_sino,
            'ground_truth_image': im_prim,
            'label': label,
            'centre': torch.tensor(center, dtype=torch.float32)
        }

        