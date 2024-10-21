import torch
import urllib
import PIL
from PIL import Image
import torchvision.transforms as transforms

transform = transforms.ToTensor()
transform_pil = transforms.ToPILImage()


'''
mean = sqrt(alpha_hat)*x_o
variance = sqrt(1-alpha_hat)*random noise
betas - amount of noise being added in each step of diffusion
alphas - 1-betas - amount of original image info being preserved after a diffusion process
'''

def forward_diffusion(x0, t, betas = torch.linspace(0.0, 1.0, 5)):
    alphas = 1 - betas
    # contains values for each time step in diffusion
    alpha_hat = torch.cumprod(alphas, axis=0)
    # getting the alpha hat for time step 1 and 3 in diffusion and reshaping to match dimensions of input image
    # for multiplication later
    result = alpha_hat.gather(-1, t).reshape(-1, 1, 1, 1)
    # mean and variance of the noisy image using the image provided or x0
    noise = torch.rand_like(x0)
    mean = result.sqrt() * x0
    variance = torch.sqrt(1 - result) * noise
    return mean + variance, noise

def get_sample_image()-> PIL.Image.Image:
    url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTZmJy3aSZ1Ix573d2MlJXQowLCLQyIUsPdniOJ7rBsgG4XJb04g9ZFA9MhxYvckeKkVmo&usqp=CAU'
    filename = 'racoon.jpg'
    urllib.request.urlretrieve(url, filename)
    return PIL.Image.open(filename)


if __name__== "__main__":
    # batch_size = 1
    # x0 = torch.randn(batch_size, 3, 32, 32)
    t = torch.tensor([4])
    image = get_sample_image()
    x0 = transform(image)
    x0 = x0.unsqueeze(0)
    x_t, noise = forward_diffusion(x0, t)
    x_t = x_t.squeeze(0)
    x_t = transform_pil(x_t)
    x_t.save('noisy_racoon.jpg')



    