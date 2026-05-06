import numpy as np
from tomosipo.torch_support import to_autograd
import tomosipo as ts
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision import transforms
import random
import torch
import torchvision as tv
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import struct

class mnist_ct(Dataset):

    def __init__(self, data, A, A_T, T = 30, alpha= 1/(40.08146667480469)**2, num_angles=60, sigma = 0.2, i_0 = 10**5, constant = 10**2,  seed = 29072000, device="cuda"):
       
        self.data = data
        self.T = T
        self.alpha = alpha
        self.num_angles = num_angles
        self.sigma = sigma
        self.i_0 = i_0
        self.constant = constant
        self.seed = seed
        self.device = torch.device(device)

        # Create tomosipo volume and projection geometry
        self.A = A   
        self.A_T = A_T

    
    def __len__(self):
    
        return len(self.data)


    def __repr__(self):
        """
        Returns a string summary of the dataset configuration.

        Returns:
            str: 
                Dataset description.
        """

        return (f"MnistDataset(num_samples={len(self.data)}, "
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
        g = torch.Generator(device=self.device)
        g.manual_seed(self.seed + int(idx))

        #get image and label
        img, label = self.data[idx]
        img = img.to(self.device)
        img = img.unsqueeze(0) 
        #sinogram
        sinogram = self.A(img)

        # adding noise
        noisy_sino = self.noise(sinogram, g) 

        
        x = torch.zeros(1, 1, 28, 28, device=self.device)

        for t in range(self.T):
            r = noisy_sino - self.A(x)
            x = x + self.alpha * self.A_T(r)
        x = x.clamp(0, 1)

        if idx == 0:
            plt.figure()
            plt.imshow(x[0, 0].detach().cpu().numpy(), cmap="gray")  # (28,28)
            plt.axis("off")
            plt.savefig("simpler_pipeline/results/MNIST_CT_EXAMPLE.png", dpi=300, bbox_inches="tight")
            plt.close()

        return {'mnist_ct': x.squeeze(0),
            'noisy_sinogram': noisy_sino,  
            'ground_truth_image':  img.squeeze(0),
            'label': label
            }


def generate_full_dataset(loader, img_key="mnist_ct", label_key="label"):
    """
    Convierte un DataLoader (que devuelve dicts) en arrays numpy:
      images: (N, 1, 28, 28)
      labels: (N,)
    """

    images = []
    labels = []

    for batch in tqdm(loader):
        # Caso 1: batch es dict (lo que devuelves tú en mnist_ct)
        if isinstance(batch, dict):
            x = batch[img_key]       # (B, 1, 28, 28) o (B,1,28,28)
            y = batch[label_key]     # (B,)

        # Caso 2: batch es tuple/list (img, label)
        elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            raise TypeError(f"No entiendo el formato del batch: {type(batch)}")

        # Asegura tensores en CPU para numpy
        x = x.detach().cpu()
        y = y.detach().cpu()

        images.append(x.numpy())
        labels.append(y.numpy())

    images = np.concatenate(images, axis=0)  # (N, 1, 28, 28)
    labels = np.concatenate(labels, axis=0)  # (N,)

    # Por si labels viene como float tensor
    labels = labels.astype(np.int64)

    return images, labels



def to_uint8(images):
    images = images.squeeze(1)       # (N,28,28)
    images = np.clip(images, 0, 1)
    images = (images * 255).astype(np.uint8)
    return images


def save_images_idx(filename, images):
    with open(filename, 'wb') as f:
        f.write(struct.pack('>IIII',
                            2051,                # magic number
                            images.shape[0],     # N
                            images.shape[1],     # rows
                            images.shape[2]))    # cols
        f.write(images.tobytes())


def save_labels_idx(filename, labels):
    with open(filename, 'wb') as f:
        f.write(struct.pack('>II',
                            2049,               # magic number
                            labels.shape[0]))   # N
        f.write(labels.astype(np.uint8).tobytes())


# define CT geometry
vg = ts.volume(shape=(1,28,28))                                 # Volume
angles = np.linspace(0, np.pi, 60, endpoint=True)     # Angles
pg = ts.parallel(angles=angles, shape=(1, 28))                 # Parallel beam structure
A = ts.operator(vg,pg)


A_torch  = to_autograd(A,   is_2d=True, num_extra_dims=2)
AT_torch = to_autograd(A.T, is_2d=True, num_extra_dims=2)

batch_size = 128

train_data = tv.datasets.MNIST('datasets/', train=True,
                                 download=True, transform=tv.transforms.ToTensor())


test_data = tv.datasets.MNIST('datasets/', train=False,
                                download=True, transform=tv.transforms.ToTensor()) 


train = mnist_ct(train_data, A_torch, AT_torch)
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
test = mnist_ct(test_data, A_torch, AT_torch)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)


train_images, train_labels = generate_full_dataset(train_loader)


train_images = to_uint8(train_images)

save_images_idx("data/MNIST_CT/train-images-ct-idx3-ubyte", train_images)
save_labels_idx("data/MNIST_CT/train-labels-ct-idx1-ubyte", train_labels)


test_images, test_labels = generate_full_dataset(test_loader)
test_images = to_uint8(test_images)

save_images_idx("data/MNIST_CT/test-images-ct-idx3-ubyte", test_images)
save_labels_idx("data/MNIST_CT/test-labels-ct-idx1-ubyte", test_labels)