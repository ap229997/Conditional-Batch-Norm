# Source : https://github.com/GuessWhatGame/generic/tree/master/data_provider

from PIL import Image
import numpy as np

def resize_image(img, width, height):
    return img.resize((width, height), resample=Image.BILINEAR)

def get_spatial_feat(bbox, im_width, im_height):
    # Rescale features fom -1 to 1

    x_left = (1.*bbox.x_left / im_width) * 2 - 1
    x_right = (1.*bbox.x_right / im_width) * 2 - 1
    x_center = (1.*bbox.x_center / im_width) * 2 - 1

    y_lower = (1.*bbox.y_lower / im_height) * 2 - 1
    y_upper = (1.*bbox.y_upper / im_height) * 2 - 1
    y_center = (1.*bbox.y_center / im_height) * 2 - 1

    x_width = (1.*bbox.x_width / im_width) * 2
    y_height = (1.*bbox.y_height / im_height) * 2

    # Concatenate features
    feat = [x_left, y_lower, x_right, y_upper, x_center, y_center, x_width, y_height]
    feat = np.array(feat)

    return feat


def scaled_crop_and_pad(bbox, raw_img, scale=1.0):

    im_width, im_height = raw_img.size

    # Need to use integer only
    x_left, x_right, y_lower, y_upper = bbox.x_left, \
                                        bbox.x_right, \
                                        bbox.y_lower, \
                                        bbox.y_upper

    x_left = int((2.0 - scale) * x_left )
    x_right = int(min(scale * x_right, im_width))

    y_lower = int((2.0 - scale) * y_lower)
    y_upper = int(min(scale * y_upper, im_height))


    # Create crop with tuple defining by left, upper, right, lower (Beware -> y descending!)
    crop = raw_img.crop(box=(x_left, im_height-y_upper, x_right,im_height-y_lower))
    crop_w, crop_h = crop.size

    # rescaling the crop
    max_side = max(crop_w, crop_h)
    background = Image.new('RGB', (max_side, max_side), (0, 0, 0))
    background.paste(im=crop, box=(((max_side - crop_w) // 2), (max_side - crop_h) // 2))
    return background


