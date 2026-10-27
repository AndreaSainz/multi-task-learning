import torch
from tqdm import tqdm

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