import numpy as np
import normflows as nf
import torch



def normalizing_flow_model(L= 2,K= 32,input_shape= (1, 28, 28), channels=1, hidden_channels= 256, split_mode= 'channel', 
                            scale= "True", num_classes=10):
    """
    Build and initialize a class-conditional multiscale normalizing flow model.

    The model is constructed using Glow blocks arranged in multiple levels.
    Each level contains ``K`` Glow blocks followed by a squeeze operation.
    Merge operations are added between levels to construct the multiscale flow.
    A class-conditional diagonal Gaussian distribution is used as the base
    distribution at each level.

    Parameters
    ----------
        L : int, optional
            Number of levels in the multiscale flow.

        K : int, optional
            Number of Glow blocks used at each level.

        input_shape : tuple of int, optional
            Shape of the input images in ``(channels, height, width)`` format.

        channels : int, optional
            Number of input image channels.

        hidden_channels : int, optional
            Number of hidden channels used in the Glow blocks.

        split_mode : str, optional
            Splitting strategy used by the Glow blocks. Default is ``"channel"``.

        scale : str or bool, optional
            Whether scaling is used in the Glow blocks.

        num_classes : int, optional
            Number of classes used for the class-conditional base distributions.

    Returns
    -------
    nf.MultiscaleFlow
        Initialized class-conditional multiscale normalizing flow model.
    """

    torch.manual_seed(0)
    n_dims = np.prod(input_shape)

    # Set up flows, distributions and merge operations
    q0 = []
    merges = []
    flows = []
    for i in range(L):
        flows_ = []
        for j in range(K):
            flows_ += [nf.flows.GlowBlock(channels * 2 ** (L + 1 - i), hidden_channels,
                                        split_mode=split_mode, scale=scale)]
        flows_ += [nf.flows.Squeeze()]
        flows += [flows_]
        if i > 0:
            merges += [nf.flows.Merge()]
            latent_shape = (input_shape[0] * 2 ** (L - i), input_shape[1] // 2 ** (L - i), 
                            input_shape[2] // 2 ** (L - i))
        else:
            latent_shape = (input_shape[0] * 2 ** (L + 1), input_shape[1] // 2 ** L, 
                            input_shape[2] // 2 ** L)
        q0 += [nf.distributions.ClassCondDiagGaussian(latent_shape, num_classes)]

    model = nf.MultiscaleFlow(q0, flows, merges)
    model = model.to(device)

    return model