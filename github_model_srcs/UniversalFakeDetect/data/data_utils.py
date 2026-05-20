import numpy as np
from PIL import Image
from random import random, choice
from scipy.ndimage.filters import gaussian_filter
from io import BytesIO
import cv2

########################
# Following are image augmentation and transformation functions preset by UniversalFakeDetect paper
########################
def data_augment(img, args):
    if args.is_train:
        img = np.array(img)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)

        if random() < args.blur_prob:
            sig = sample_continuous(args.blur_sig)
            gaussian_blur(img, sig)

        if random() < args.jpg_prob:
            method = sample_discrete(args.jpg_method)
            qual = sample_discrete(args.jpg_qual)
            img = jpeg_from_key(img, qual, method)

        return Image.fromarray(img)
    else:
        return img
    
def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img

jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)

def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


