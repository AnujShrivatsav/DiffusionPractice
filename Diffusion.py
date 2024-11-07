import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms # or from torchvision import transforms
import urllib
import PIL
import matplotlib.pyplot as plt
import numpy as np
from Model import UNet
import os

device = 'cuda'

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(), # torch converts to 0 to 1 range inherently, so we need to scale it from -1 to 1 as our noise is isotropic gaussian so lies between -1 to 1
    transforms.Lambda(lambda t: (t*2)-1), # scales b/w [-1, 1]
])

reverse_transform = transforms.Compose([
    transforms.Lambda(lambda t: (t + 1)/2), # scale is back to 0-1
    transforms.Lambda(lambda t: t.squeeze().permute(1, 2, 0)), #CHW to HCW ie torch to numpy
    transforms.Lambda(lambda t: t * 255.), # scale data between 0-255
    transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)), # convert to uint8 numpy array
    transforms.ToPILImage(), # Convert back to PIL Image
])

class DiffusionModel:
    '''
    betas - amount of noise being added in each step of diffusion
    alphas - 1-betas - amount of original image info being preserved after a diffusion process
    '''

    def __init__(self, start_schedule=0.0001, end_schedule=0.02, timesteps=300):
        self.start_schedule = start_schedule
        self.end_schedule = end_schedule
        self.timesteps = timesteps
        self.betas = torch.linspace(start_schedule, end_schedule, timesteps)
        self.alphas = 1 - self.betas
        # contains values for each time step in diffusion
        self.alpha_hat = torch.cumprod(self.alphas, axis=0)

    def get_sample_image(self)-> PIL.Image.Image:
        url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTZmJy3aSZ1Ix573d2MlJXQowLCLQyIUsPdniOJ7rBsgG4XJb04g9ZFA9MhxYvckeKkVmo&usqp=CAU'
        filename = 'racoon.jpg'
        urllib.request.urlretrieve(url, filename)
        return PIL.Image.open(filename)

    def forward_diffusion(self, x0, t, device):
        '''
        x0 - (B, C, H, W)
        t - (B,)
        mean = sqrt(alpha_hat)*x_o
        variance = sqrt(1-alpha_hat)*random noise
        '''
        # getting the alpha hat for time step 1 and 3 in diffusion and reshaping to match dimensions of input image
        # for multiplication later
        sqrt_alpha_hats = self.get_index_from_list(self.alpha_hat.sqrt(), t, x0.shape)
        sqrt_info_alpha_hats = self.get_index_from_list(torch.sqrt(1 - self.alphas), t, x0.shape)
        # mean and variance of the noisy image using the image provided or x0
        noise = torch.rand_like(x0)
        mean = sqrt_alpha_hats.to(device) * x0.to(device)
        variance = sqrt_info_alpha_hats.to(device) * noise.to(device)
        return mean + variance, noise.to(device)

    @torch.no_grad()
    def backward_diffusion(self, x, t, model):
        '''
        calls the model to predict the noise added at that time step.
        slowly obtains the denoised image from true data distribution
        Applied noise to the image if not in last step
        x_t-1 = 1/sqrt(alpha_t) (x_t - beta_t*model(x_t, t)/sqrt(1-alpha_hat_t)) + sigma_t*z
        where z is sampled from a zero mean unit var gaussian if t >= 1 that is before returning the original image step
        '''
        betas_t = self.get_index_from_list(self.betas, t, x.shape)
        sqrt_recip_alphas = self.get_index_from_list(torch.sqrt(1.0/self.alpha_hat), t, x.shape)
        sqrt_one_minus_alpha_hats = self.get_index_from_list(torch.sqrt(1 - self.alpha_hat), t, x.shape)
        mean = sqrt_one_minus_alpha_hats * (x - betas_t*model(x, t)/sqrt_one_minus_alpha_hats)
        posterior_variance = betas_t

        output = []
        # handle the case where a set of time steps can be zero
        # zero_values = t[mask = (t == 0)]
        # for values in zero_values:
        #     output.append(mean)
        # for values in t and not in zero_values:
        #     output.append(mean)
        if t == 0:
            return mean
        else:
            z = torch.rand_like(x)
            variance = torch.sqrt(posterior_variance)*z
            return mean + variance

    @staticmethod
    def get_index_from_list(values, t, x0_shape):
        bs = t.shape[0]
        result = torch.gather(values, -1, t.cpu())
        '''
        reshape to (bs, 1, 1, 1) dims so len(x0_shape)
        '''
        return result.view(bs, *((1,) * (len(x0_shape)-1))).to(t.device)

if __name__== "__main__":
    diffusion_model = DiffusionModel()
    batch_size = 1
    lr = 1e-3
    no_epochs = 10000
    print_freq = 500
    image = diffusion_model.get_sample_image()
    height, width = image.size
    x0 = transform(image)
    # creating batch of the same images
    x0 = torch.stack([x0]*batch_size)
    unet = UNet(x0.shape[1]).to(device)
    optimizer = torch.optim.Adam(unet.parameters(), lr)
    save_dir = "Denoised_Images/"
    for epoch in range(no_epochs):
        epoch_losses = []

        t = torch.randint(0, diffusion_model.timesteps, (batch_size, )).long()
        noisy_img, gt_noise = diffusion_model.forward_diffusion(x0, t, device)
        preds = unet(noisy_img, t.to(device))

        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(gt_noise, preds)

        epoch_losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if epoch % print_freq == 0:
            print('---')
            print(f'Epoch: {epoch} | Train loss: {np.mean(epoch_losses)}')
            plt.figure(figsize=(5, 5))
            denoised_img = diffusion_model.backward_diffusion(noisy_img, t.to(device), unet)
            resize_transform = transforms.Resize((height, width))
            denoised_img = resize_transform(denoised_img)
            denoised_img = reverse_transform(denoised_img)
            plt.imshow(denoised_img, cmap='gray')
            plt.axis('off')
            plt.title(f'Epoch {epoch}')
            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            plt.savefig(os.path.join(save_dir, f'prediction_epoch_{epoch}.png'))
            plt.close()



    