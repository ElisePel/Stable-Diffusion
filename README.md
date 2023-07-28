<div align="center">
  <h2>Generation of new shape with Stable Diffusion</h2>
</div>

## 1. Data
cube data set : https://ranahanocka.github.io/MeshCNN/

## 2. Train
In the file [stable-diffusion.py](stable-diffusion.py), load the data and modify the parameters:
- epochs
- batch size
- learning rate  

The number of steps, iterations, and image dimensions should be indicated in [UNet_StableDiffusion.py](UNet_StableDiffusion.py).
In a terminal, activate the environment and run ```python stable-diffusion.py```. This file trains the diffusion model and saves the model weights and loss.
It is recommended to use a GPU and avoid training in a notebook, as it may cause the kernel to crash.

## 3. Generate
In the notebook [2.Stable Diffusion.ipynb](2.Stable Diffusion.ipynb), load the weights and loss. Then, denoising method with DDPM.

Afterwards, results can be saved.

## Results
![image](https://github.com/ElisePel/Stable-Diffusion/assets/98736513/abb8192c-08f5-42c4-85e1-f09fcac0d152)


## References
Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Bjorn Ommer, «High-Resolution Image Synthesis with Latent Diffusion Models,» 2022.
Jonathan Ho, Ajay Jn, Pieter Abbeel, «Denoising Diffusion Probabilistic Models,» 2020.
