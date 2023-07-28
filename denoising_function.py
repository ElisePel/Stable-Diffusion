import numpy as np
from UNet-DiffusionModel import alpha, alpha_bar, beta

def ddpm(x_t, pred_noise, t):
    """
    Performs denoising using the DDPM (Denoising Diffusion Probabilistic Model) algorithm.

    Args:
        x_t (array): Input image at time step t.
        pred_noise (array): Predicted noise at time step t.
        t (int): Time step.

    Returns:
        array: Denoised image.
    """
    alpha_t = np.take(alpha, t)
    alpha_t_bar = np.take(alpha_bar, t)

    eps_coef = (1 - alpha_t) / (1 - alpha_t_bar) ** .5
    mean = (1 / (alpha_t ** .5)) * (x_t - eps_coef * pred_noise)

    var = np.take(beta, t)
    z = np.random.normal(size=x_t.shape)

    return mean + (var ** .5) * z


def ddim(x_t, pred_noise, t, sigma_t):
    """
    Performs denoising using the DDIM (Denoising Diffusion Implicit Model) algorithm.

    Args:
        x_t (array): Input image at time step t.
        pred_noise (array): Predicted noise at time step t.
        t (int): Time step.
        sigma_t (float): Standard deviation at time step t.

    Returns:
        array: Denoised image.
    """
    alpha_t_bar = np.take(alpha_bar, t)
    alpha_t_minus_one = np.take(alpha, t-1)

    pred = (x_t - ((1 - alpha_t_bar) ** 0.5) * pred_noise)/ (alpha_t_bar ** 0.5)
    pred = (alpha_t_minus_one ** 0.5) * pred

    pred = pred + ((1 - alpha_t_minus_one - (sigma_t ** 2)) ** 0.5) * pred_noise
    eps_t = np.random.normal(size=x_t.shape)
    pred = pred+(sigma_t * eps_t)
    return pred