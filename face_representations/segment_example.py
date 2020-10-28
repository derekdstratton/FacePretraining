from modeling.deeplab import DeepLab
import kornia
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import numpy as np

# this example only uses 1 image, so cpu is fine
device = torch.device("cpu")

# load pre-trained weights, set network to inference mode
network = DeepLab(num_classes=18)
network.load_state_dict(torch.load("segmentation-model/epoch-14", map_location="cpu"))
network.eval()
network.to(device)

# load example image. the image is resized because DeepLab uses
# a lot of dilated convolutions and doesn't work very well for
# low resolution images.
image = Image.open("nate.jpg")
scaled_image = image.resize((418,512),resample=Image.LANCZOS)
image_tensor = TF.to_tensor(scaled_image)

# send the input through the network. unsqueeze is used to
# add a batch dimension, because torch always expects a batch
# but in this case it's just one image
# I then use Kornia to resize the mask back to 218x178 then
# squeeze to remove the batch channel again (kornia also
# always expects a batch dimension)
with torch.no_grad():
    mask_large = network(image_tensor.unsqueeze(0).to(device))
mask = kornia.resize(mask_large, (218,178)).squeeze(0)

# this function saves the mask as 18 grayscale JPG images
# (one for each channel). It takes the torch tensor as input
def save_mask(mask, image_name, folder):
    mask = np.uint8(np.array(torch.sigmoid(mask))*255)
    for i, channel in enumerate(mask):
        channel_im = Image.fromarray(channel, mode='L')
        channel_im.save(f"{folder}/{image_name}_{i}.jpg")

save_mask(mask, "nate", "output")

# finally, I have a function for plotting the mask as a single
# colorful image for better vizualization
def make_mask_image(mask):
    # sigmoid is needed because DeepLab does not have a sigmoid at
    # the end but I train using BCEWithLogitsLoss
    mask = torch.sigmoid(mask)

    # tile each segment to 3 x height x width
    face_scale = np.tile(mask[0].detach().numpy(), (3,1,1))
    nose_scale = np.tile(mask[1].detach().numpy(), (3,1,1))
    eyes_scale = np.tile(mask[2].detach().numpy(), (3,1,1))
    eyes_scale += np.tile(mask[3].detach().numpy(), (3,1,1))
    brows_scale = np.tile(mask[4].detach().numpy(), (3,1,1))
    brows_scale += np.tile(mask[5].detach().numpy(), (3,1,1))
    ears_scale = np.tile(mask[6].detach().numpy(), (3,1,1))
    ears_scale += np.tile(mask[7].detach().numpy(), (3,1,1))
    mouth_scale = np.tile(mask[8].detach().numpy(), (3,1,1))
    lips_scale = np.tile(mask[9].detach().numpy(), (3,1,1))
    lips_scale += np.tile(mask[10].detach().numpy(), (3,1,1))
    hair_scale = np.tile(mask[11].detach().numpy(), (3,1,1))
    hat_scale = np.tile(mask[12].detach().numpy(), (3,1,1))
    neck_scale = np.tile(mask[16].detach().numpy(), (3,1,1))
    cloth_scale = np.tile(mask[17].detach().numpy(), (3,1,1))

    # multiply segment by color, add to mask image
    mask_im = np.zeros(face_scale.shape, dtype=np.float64)
    mask_im += np.array([.75,0,0]).reshape(3,1,1)*face_scale
    mask_im += np.array([.2,.9,.2]).reshape(3,1,1)*nose_scale
    mask_im += np.array([0,0,1]).reshape(3,1,1)*eyes_scale
    mask_im += np.array([0,1,1]).reshape(3,1,1)*brows_scale
    mask_im += np.array([0,1,0]).reshape(3,1,1)*ears_scale
    mask_im += np.array([1,0,0]).reshape(3,1,1)*mouth_scale
    mask_im += np.array([0,0,.75]).reshape(3,1,1)*lips_scale
    mask_im += np.array([0,0,1]).reshape(3,1,1)*hair_scale
    mask_im += np.array([1,.5,0]).reshape(3,1,1)*hat_scale
    mask_im += np.array([1,.4,.7]).reshape(3,1,1)*neck_scale
    mask_im += np.array([0,.5,0]).reshape(3,1,1)*cloth_scale

    # adjust data from [0,1] floats to [0,255] ints
    mask_im = np.moveaxis(mask_im,0,-1)*255
    mask_im[mask_im > 255] = 255
    mask_im = Image.fromarray(np.uint8(mask_im))

    return mask_im

color_mask_image = make_mask_image(mask)
color_mask_image.save(f"output/nate_segmented.jpg", quality=100)
