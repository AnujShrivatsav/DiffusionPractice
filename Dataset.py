import os
import tqdm
import PIL.Image as Image
import torchvision.transforms as transforms # or from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(), # torch converts to 0 to 1 range inherently, so we need to scale it from -1 to 1 as our noise is isotropic gaussian so lies between -1 to 1
    transforms.Lambda(lambda t: (t*2)-1), # scales b/w [-1, 1]
])

class MNIST_Dataset(Dataset):
    '''
    Simple class for mnist images, can be later used for any standard dataset
    '''
    def __init__(self, split, im_path, im_ext='png'):
        r"""
        ;param split: train/test to locate image files
        ;param im_path: root folder of images
        ;param im_ext: image extension
        """
        self.split = split
        self.im_ext = im_ext
        self.images, self.labels = self.load_images(im_path)

    def load_images(self, im_path):
        '''
        gets all images from path specified
        '''
        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)
        ims = []
        labels = []
        for d_name in tqdm(os.listdir(im_path)):
            for fname in glob.glob(os.path.join(im_path, d_name, '*.{}'.format(self.im_ext))):
                ims.append(fname)
                labels.append(int(d_name))
        print('Found {} images for split {}'.format(len(ims), self.split))
        return ims, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        im = Image.open(self.images[index])
        im_tensor = transform(im)

        return im_tensor

