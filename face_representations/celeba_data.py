import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import kornia
import kornia.augmentation as K
import os

class AugmentedCelebA(Dataset):
    def __init__(self, celeba_folder, mask_folder,
                 mode="train", use_transforms=False):
        super().__init__()
        self.images = []
        self.labels = []
        self.mode = mode
        self.mask_folder = mask_folder

        self.crop_aug = K.RandomCrop((218,178), pad_if_needed=True)
        self.flip_aug = K.RandomHorizontalFlip()

        self.id_map = {}
        with open(celeba_folder + "/identity_CelebA.txt") as id_f:
            for line in id_f:
                im, id = line.split()
                self.id_map[im[:-4]] = int(id)

        with open(celeba_folder + "/labels/list_attr_celeba.txt") as label_f:
            id_count = label_f.readline().strip("\n")
            self.attributes = np.array(label_f.readline().strip("\n").split())

            for i, line in enumerate(label_f.read().split("\n")[:-1]):
                image_data = line.split()
                image = image_data[0]

                if not os.path.exists(f"{mask_folder}"
                                      f"/id-{self.id_map[image[:-4]]}"):
                    continue

                image_labels = [int(label) for label in image_data[1:]]

                image_n = int(image[:-4])
                train_cond = mode == "train" and image_n < 162771
                val_cond = mode == "val" and image_n>=162771 and image_n<182638
                test_cond = mode == "test" and image_n >= 182638
                full_cond = mode == "full"
                if train_cond or val_cond or test_cond or full_cond:
                    self.images.append(celeba_folder + "/images/" + image)
                    self.labels.append(image_labels)

        self.alt_background = TF.to_tensor(Image.open("unicorn.jpg"))

        self.length = len(self.images)

    def load_mask(self, folder, image_name):
        mask = np.zeros((18,218,178))
        for i in range(18):
            channel = Image.open(
                f"{folder}/id-{self.id_map[image_name]}/{image_name}_{i}.jpg")
            mask[i] = np.array(channel)/255

        return torch.tensor(mask)

    # delete background (everything that isn't face)
    def background_removal(self, image, mask):
        no_background = image*mask[0].repeat((3,1,1))
        return no_background.type(torch.float32)

    # remove smallest eye from image
    def eye_removal(self, image, mask):
        l_eye_area = torch.sum(mask[2] > .5)
        r_eye_area = torch.sum(mask[3] > .5)
        if l_eye_area == r_eye_area and l_eye_area == 0:
            pass
        elif l_eye_area > r_eye_area:
            image[mask[3].repeat((3,1,1)) > .5] = 0
        else:
            image[mask[2].repeat((3,1,1)) > .5] = 0
        return no_eye.type(torch.float32)

    # apply random color distortion, with the hue shift
    # substantially reduced for the face region
    def selective_color_distort(self, image, mask):
        # make skin mask using face, neck, and ear segments
        skin = mask[0] + mask[16] + mask[6] + mask[7]
        # ensure there's no clipping
        skin[skin > 1] = 1
        skin = skin.repeat((3,1,1))

        # torch transforms only work on PIL images, so first
        # the image tensors must be converted to PIL images
        background = TF.to_pil_image(image)
        face = TF.to_pil_image(image)

        # generate parameters for color jitter
        # taken from https://arxiv.org/pdf/2002.05709.pdf page 12
        jitter_params = torch.rand(4)*torch.tensor([.8, .8, .8, .2])
        if torch.rand(1).item() > .2:
            # apply transforms to background
            background = TF.adjust_brightness(background, .2 + jitter_params[0]*2)
            background = TF.adjust_contrast(background, .2 + jitter_params[1]*2)
            background = TF.adjust_saturation(background, .2 + jitter_params[2]*2)
            background = TF.adjust_hue(background, jitter_params[3]*2 - .2)

            # apply identical brightness/contrast/saturation transform to face
            face = TF.adjust_brightness(face, .2 + jitter_params[0]*2)
            face = TF.adjust_saturation(face, .2 + jitter_params[2]*2)

            # only apply 50% of the contrast transform to the face
            face = TF.adjust_contrast(face, .4 + jitter_params[1])
            # only apply 25% of the hue transform to the face
            face = TF.adjust_hue(face, jitter_params[3]*.5 - .05)

        # note: the possibility of neither color jitter or grayscale or both
        #       color jitter and grayscale is intended behavior. the random
        #       draws are meant to be different for these two if statements
        if torch.rand(1).item() > .8:
            background = TF.to_grayscale(background, num_output_channels=3)
            face = TF.to_grayscale(face, num_output_channels=3)

        # convert PIL images back to tensors
        background = TF.to_tensor(background)
        face = TF.to_tensor(face)

        # construct final image as convex combination of face pixels
        # and mask pixels. where the mask has higher confidence (closer
        # to 1), the face image will be used
        distorted = (1 - skin)*background + skin*face

        return distorted.type(torch.float32)

    # replaces background (anything not included in a mask)
    # with a provided image
    def background_replacement(self, image, new_background, mask):
        mask_sum = torch.sum(mask, axis=0)
        mask_sum[mask_sum > 1] = 1

        replaced_background = (1 - mask_sum)*new_background + mask_sum*image

        return replaced_background.type(torch.float32)

    def augment(self, image, mask):
        # scale between .85x - 1.4x
        rand_scale = torch.rand(1).item()*.55 + .85
        new_dimensions = (int(218*rand_scale),int(178*rand_scale))

        augmented = kornia.resize(image.unsqueeze(0), new_dimensions)
        aug_mask = kornia.resize(mask.unsqueeze(0), new_dimensions)

        aug_mask = aug_mask.type(torch.float32)
        concat = self.crop_aug(torch.cat((augmented,aug_mask),
                                         axis=1)).squeeze(0)
        augmented = concat[:3]
        aug_mask = concat[3:]

        rand_background = TF.to_tensor(Image.open(self.images[
            torch.randint(len(self.images),(1,)).item()]))
        augmented = self.background_replacement(augmented, rand_background,
                                                aug_mask)
        augmented = self.selective_color_distort(augmented, aug_mask)
        augmented = self.flip_aug(augmented.unsqueeze(0)).squeeze(0)

        return augmented

    def __getitem__(self, index):
        image = TF.to_tensor(Image.open(self.images[index]))
        image_n = self.images[index].split("/")[-1][:-4]
        mask = self.load_mask(self.mask_folder, image_n)

        augmented_1 = self.augment(image, mask)
        augmented_2 = self.augment(image, mask)

        return augmented_1, augmented_2

    def __len__(self):
        return self.length

if __name__ == "__main__":
    dataset = AugmentedCelebA("../../../CelebA", "../../celeba-segmented")
    for i, (im1, im2) in enumerate(dataset):
        plt.imshow(np.moveaxis(np.array(im1),0,-1))
        plt.show()

        plt.imshow(np.moveaxis(np.array(im2),0,-1))
        plt.show()
