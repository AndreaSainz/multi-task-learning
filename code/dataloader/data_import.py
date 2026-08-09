
def import_data(training, validation, length_train, length_validation, A, num_angles, sigma_data, i_0, seed, logger, device):
    """
    Create stratified PyTorch DataLoaders for the MNIST CT dataset.

    The function loads either the MNIST training or test split depending on
    training, and creates a stratified subset containing
    length_train samples. The stratified split preserves the distribution
    of the ten MNIST digit classes.

    If both training and validation are True, an additional
    stratified subset of length_validation samples is selected from the
    samples not used for training, and a second DataLoader is returned.

    Parameters
    ----------
        training : bool
            Determines which MNIST split is used.

            - True : use the MNIST training split.
            - False : use the MNIST test split.

            Therefore, when training=False, the returned DataLoader contains
            test samples even though it is named train_dataloader.

        validation : bool
            Determines whether a validation subset is created.

            - True : create and return both training and validation
            DataLoaders. This is only applied when training=True.
            - False : return only the main DataLoader.

        length_train : int
            Number of samples selected for the main training/test subset.

            The samples are selected using a stratified split based on their
            MNIST digit labels.

        length_validation : int
            Number of samples selected for the validation subset when
            training=True and validation=True.

            Validation samples are selected from the portion of the dataset not
            used for the main training subset, while preserving the class
            distribution.

        A : tomosipo.Operator
            TomoSipo forward operator used by ``MnistDataset`` to generate the
            CT sinograms.

        num_angles : int
            Number of projection angles used by the CT acquisition geometry.
            The geometry itself is encoded in A.

        sigma_data : float
            Standard deviation of the additive Gaussian detector noise used in
            the simulated sinograms.

        i_0 : int or float
            Incident photon count used for the Poisson noise model.

        seed : int
            Random seed used for reproducible dataset splitting and DataLoader
            shuffling.

        logger : logging.Logger
            Logger used to report the number of samples belonging to each MNIST
            digit in the selected main subset.

        device : str or torch.device
            Device used by the MNIST CT dataset to generate tensors, e.g.
            "cuda", "cuda:0" or "cpu".

    Returns
    -------
    train_dataloader : torch.utils.data.DataLoader
        DataLoader containing the selected training samples when
        training=True.

        When training=False, this DataLoader instead contains the
        selected samples from the MNIST test split.

    val_dataloader : torch.utils.data.DataLoader
        DataLoader containing the validation samples. Returned only when
        training=True and validation=True.

    Notes
    -----
    The complete MNIST split is loaded before selecting the required number
    of samples. This ensures that the selected subset is stratified across
    all ten digit classes instead of simply taking the first
    length_train samples.

    MNIST labels are accessed directly from the underlying torchvision
    dataset to avoid unnecessarily generating CT projections and noisy
    sinograms when determining the class distribution.

    The same seed produces reproducible dataset splits and DataLoader
    shuffling.
    """


    # Import data with curstom dataloader 
    data = MnistDataset(training = training,
                    testing= not training, 
                    A = A, 
                    dimension = (H,W), 
                    num_angles = num_angles, 
                    sigma = sigma_data,
                    i_0 = i_0,
                    cut = False,
                    seed = seed,
                    max_len = None,
                    device = device)

    # get the labels from MNIST dataset directly
    labels = np.array(data.data.targets)

    # Suflling and splitting for reproducibility
    g_split = torch.Generator()
    g_split.manual_seed(seed)


    # stratified split (same numer of digits in samples)
    train_idx, val_idx = train_test_split(
        np.arange(len(data)),
        train_size=length_train,
        stratify=labels,
        random_state=seed
    )

    train_data = Subset(data, train_idx)

    # create pytorch dataloader
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        generator=g_split)
    
    #Check if the dataset is balanced
    labels_bal = torch.tensor(labels[train_idx]) #labels of the trainign dataset
    counts = torch.bincount(labels_bal, minlength=10)

    if accelerator.is_main_process:
        logger.info("Balanced training dataset label counts:")
        for d in range(10):
            logger.info(f"Digit {d}: {counts[d].item()}")


    # In case we need training and validation dataset
    if training and validation:
        data_new = Subset(data, val_idx)
        val_idx_selected, _ = train_test_split(
            np.arange(len(data_new)),
            train_size=length_validation,
            stratify=labels[val_idx],
            random_state=seed)

        val_data = Subset(data_new, val_idx_selected)

        # create pytorch dataloader
        val_dataloader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False)
        
        del data, data_new, val_data, train_data
        return train_dataloader, val_dataloader

    del data, train_data

    return train_dataloader
