import torch
from tqdm import tqdm
from torch.utils.data import Dataset

def save_sirt_dataset(dataloader, save_path):
    all_images = []
    all_iterations = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Generating SIRT dataset"):

        # This could be an image or a list of images
        sirt_images = batch["sirt_reconstructions"]

        # labels
        labels = batch["label"]

        #for noise aware classifiers
        iterations = batch["iteration_normalized"]

        # If there is only one reconstruction, wrap it in a list
        if not isinstance(sirt_images, list):
            sirt_images = [sirt_images]


        batch_size = labels.shape[0]

        for sample_idx in range(batch_size):

            sample_images = [images[sample_idx].cpu() for images in sirt_images]

            all_images.append(sample_images)
            all_labels.append(labels[sample_idx].cpu())
            all_iterations.append(iterations)

    all_labels = torch.stack(all_labels)

    torch.save( { "images": all_images, "labels": all_labels, "iterations": all_iterations}, save_path)






class SIRTDataset(Dataset):

    def __init__(self, dataset_path, noise_aware=False):

        data = torch.load(dataset_path, map_location="cpu", weights_only=False)

        self.images = data["images"]
        self.labels = data["labels"]
        self.iterations = data["iterations"]

        self.noise_aware = noise_aware

        # If noise_aware=False, keep only the last reconstruction
        if not self.noise_aware:

            self.images = [images[-1]for images in self.images]

            self.iterations = [iterations[-1]for iterations in self.iterations]

        # If noise_aware=True, flatten the reconstruction lists
        else:

            all_images = []
            all_labels = []
            all_iterations = []

            for images, label, iterations in zip(self.images, self.labels, self.iterations):

                for image, iteration in zip(images, iterations):

                    all_images.append(image)
                    all_labels.append(label)
                    all_iterations.append(iteration)

            self.images = all_images
            self.labels = torch.stack(all_labels)
            self.iterations = all_iterations

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        image = self.images[idx].float()

        # Ensure image shape is (H, W)
        if image.ndim == 3 and image.shape[0] == 1:
            image = image.squeeze(0)

        if self.noise_aware:

            return {
                "sirt_reconstructions": image,
                "label": self.labels[idx],
                "iteration_normalized": self.iterations[idx]
            }

        else:

            return {
                "sirt_reconstructions": image,
                "label": self.labels[idx]
            }