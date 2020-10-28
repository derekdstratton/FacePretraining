import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import kornia
import kornia.augmentation as K

# load segmentation masks
def load_mask(folder, image_name):
    mask = np.zeros((18,218,178))
    for i in range(18):
        channel = Image.open(f"{folder}/{image_name}_{i}.jpg")
        mask[i] = np.array(channel)/255

    return mask

# delete background (everything that isn't face)
def background_removal(image, mask):
    no_background = image.clone()
    no_background *= mask[0].repeat((3,1,1))
    return no_background.type(torch.float32)

# remove smallest eye from image
def eye_removal(image, mask):
    no_eye = image.clone()

    l_eye_area = torch.sum(mask[2] > .5)
    r_eye_area = torch.sum(mask[3] > .5)
    if l_eye_area == r_eye_area and l_eye_area == 0:
        pass
    elif l_eye_area > r_eye_area:
        no_eye[mask[3].repeat((3,1,1)) > .5] = 0
    else:
        no_eye[mask[2].repeat((3,1,1)) > .5] = 0
    return no_eye.type(torch.float32)

# apply random color distortion, with the hue shift
# substantially reduced for the face region
def selective_color_distort(image, mask):
    # make skin mask using face, neck, and ear segments
    skin = mask[0] + mask[16] + mask[6] + mask[7]
    # ensure there's no clipping
    skin[skin > 1] = 1
    skin = skin.repeat((3,1,1))

    # torch transforms only work on PIL images, so first
    # the image tensors must be converted to PIL images
    face = image.clone()
    background = TF.to_pil_image(image)
    face = TF.to_pil_image(face)

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
def background_replacement(image, new_background, mask):
    mask_sum = torch.sum(mask, axis=0)
    mask_sum[mask_sum > 1] = 1

    replaced_background = (1 - mask_sum)*new_background + mask_sum*image

    return replaced_background.type(torch.float32)

# takes an image tensor as input, augments it using
# several different augmentations, then returns the result
def augmentation_pipeline(image, mask):
    alt_background = TF.to_tensor(Image.open("unicorn.jpg"))

    # doing random crop/pad before anything else means that background
    # replacement will never be cropped/padded. this prevents any 0
    # padding appearing when the image is downscaled then padded back
    # to 218x178
    crop_aug = K.RandomCrop((218,178), pad_if_needed=True)
    rand_scale = torch.rand(1).item()*.4 + .8 # scale between .8x - 1.2x
    new_dimensions = (int(218*rand_scale),int(178*rand_scale))

    # resize first
    augmented = kornia.resize(image.unsqueeze(0), new_dimensions)
    aug_mask = kornia.resize(mask.unsqueeze(0), new_dimensions)

    # crop back to 218x178
    # for some reason the generate_parameters() function for K.RandomCrop
    # always generates the same crop box so I have to concatenate the
    # image and mask so I can crop them both with the same random params
    aug_mask = aug_mask.type(torch.float32)
    concat = crop_aug(torch.cat((augmented,aug_mask), axis=1)).squeeze(0)
    augmented = concat[:3]
    aug_mask = concat[3:]

    # augmentation pipeline:
    #augmented = background_removal(image_tensor, mask)
    #augmented = eye_removal(image_tensor, mask)
    augmented = background_replacement(augmented, alt_background, aug_mask)
    augmented = selective_color_distort(augmented, aug_mask)

    return augmented

if __name__ == "__main__":
    image = Image.open("nate.jpg")
    image_tensor = TF.to_tensor(image)
    mask = torch.tensor(load_mask("output", "nate"))

    augmented = augmentation_pipeline(image_tensor, mask)

    plt.imshow(np.moveaxis(augmented.numpy(),0,-1))
    plt.show()
