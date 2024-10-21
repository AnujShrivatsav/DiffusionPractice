This is a repository to understand the working of Diffusion models on CIFAR-10 dataset.
A basic implementation of U-Net based diffusion in pytorch and visualizations to see the prediction of noise at certain epochs in the training and image denoising

## TODO

- [x] Task 1: Test out sampling an image and adding gaussian noise
- [x] Task 2: Write a class to implement the above forward_diffusion process
- Task 3: Implement a basic Unet to predict the noise at each denoising step
- Task 4: Implement the reverse diffusion function inside the class based on the sampling algo in the paper
- Task 5: Train and plot loss curves on CIFAR-10 dataset
- Task 6: Helper functions to see the predicted and ground truth noise distribution and images generated
