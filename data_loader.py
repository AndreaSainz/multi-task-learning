from torchvision.datasets import MNIST
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import tomosipo as ts
import random
import torch

"""
MNIST CT Dataset — Parallel-Beam Tomography
------------------------------------------

This dataset generates simple CT phantoms using MNIST digits:

1. Loads MNIST digits (train or test).
2. Embeds each digit inside a larger blank canvas to produce a 2D phantom.
3. Applies random translations so digits appear in different positions.
4. Builds a parallel-beam CT geometry using tomosipo:
       - Volume geometry (vg)
       - Parallel-beam projection geometry (pg)
       - Ray transform operator A
5. Computes sinograms by applying A(im_prim).
6. Supports Poisson + Gaussian realistic CT noise.
7. Returns:
       (noisy sinogram, ground-truth image, MNIST label, digit center)

"""

class MnistDataset(Dataset):

    def __init__(self, training=True, testing=False, dimension=(50, 50), num_angles=60, sigma = 0.1, standard_dose = True, seed = 29072000, max_len = None,  device="cuda"):
        """
        Initialize the MNIST CT dataset, load data, and build CT geometry.

        Parameters:
            training (bool):
                If True, use the MNIST training split.
            testing (bool):
                If True, use the MNIST test split.
            dimension (tuple[int, int]):
                Size (H, W) of the padded image / volume grid.
            num_angles (int):
                Number of projection angles for CT acquisition.
            sigma (float):
                Standard deviation of the Gaussian detector noise.
            standard_dose (bool):
                If True, uses standard dose I0 = 1e5; otherwise reduced dose.
            seed (int):
                Base random seed used for reproducible noise and translations.
            max_len (int or None):
                Optional cap on the number of samples exposed by the dataset.
            device (str):
                Device string (e.g. "cpu" or "cuda") used for generators/tensors.
        """
        self.training = training
        self.testing = testing
        self.dimension = dimension
        self.H, self.W = self.dimension
        self.num_angles = num_angles
        self.sigma = sigma
        
        if standard_dose:
            self.i_0 = 10**5
        else:
            self.i_0 = 10**5/4
        self.seed = seed
        self.max_len = max_len
        self.device = device

        #load mnist dataset 
        transform = transforms.ToTensor()

        if self.training:
            self.data = MNIST(root="data", train=True, download=True, transform=transform)
        elif self.testing:
            self.data = MNIST(root="data", train=False, download=True, transform=transform)


        # Create tomosipo volume and projection geometry
        self.vg = ts.volume(shape=(1,self.H,self.W))                                                       # Volumen
        self.angles = np.linspace(0, np.pi, self.num_angles, endpoint=True)                                # Angles
        self.pg = ts.parallel(angles=self.angles, shape=(1, self.H))                                       # Parallel beam structure
        self.A = ts.operator(self.vg,self.pg)     

    
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
    
    
    def noise(self, sinogram, idx):
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

        #Initilizing seed generator for reproducibility porpuses (this is for having more than one coworkers)
        generator = torch.Generator(device=sinogram.device).manual_seed(self.seed + idx)

        # Simulate measured photons using Poisson noise (counting error)
        measured_photons = torch.poisson(self.i_0 * torch.exp(-norm_sino), generator=generator)

        # Avoid log(0) by setting the minimum to 1
        measured_photons = torch.clamp(measured_photons, min=1.0)

        # Convert back to log domain (using Beer–Lambert Law)
        noisy_sinogram = -torch.log(measured_photons / self.i_0)

        # Adding gaussian noise (detector's imperfections)
        gaussian_noise = torch.normal(mean=0, std=self.sigma, size=sinogram.shape, dtype=sinogram.dtype, device=sinogram.device) 
        #other functions do not allow for device

        # Final result 
        noisy_sinogram += gaussian_noise

        # reaply the normalization
        noisy_sinogram = self.max_denormalize(noisy_sinogram, max_val)

        # no negative 
        noisy_sinogram = torch.clamp(noisy_sinogram, min=0)

        return noisy_sinogram
    
    def padding_and_translation(self,img, idx):
        """
        Embed a MNIST digit into a larger zero-padded canvas with translation.

        Parameters:
            img (torch.Tensor):
                MNIST image tensor of shape (1, h, w).
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
        im_prim = torch.zeros((1, self.H, self.W))

        # make translation reproducible by using idx
        g = torch.Generator(device=self.device)
        g.manual_seed(self.seed + idx)

        # poistion of the top-left corner of mnist
        x = torch.randint(0, self.H - h + 1, (1,), generator=g).item()
        y = torch.randint(0, self.W - w + 1, (1,), generator=g).item()


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
                    Original MNIST class label (digit identity).
                center (tuple[int, int]):
                    Center coordinates of the digit in the padded canvas.
        """
        
        #get image and label
        img, label = self.data[idx]

        #padding and translation
        center, im_prim = self.padding_and_translation(img, idx)

        #sinogram
        sinogram = self.A(im_prim)

        # adding noise
        noisy_sino = self.noise(sinogram, idx)

        return noisy_sino, im_prim, label, center

