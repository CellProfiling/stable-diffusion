import numpy as np
from PIL import Image
from skimage import exposure
import torch


def is_float(x):
    if isinstance(x, np.ndarray):
        return np.issubdtype(x.dtype, np.floating)
    elif torch.is_tensor(x):
        return torch.is_floating_point(x)
    else:
        raise ValueError('The input is not a numpy array or a torch tensor.')


def is_integer(x):
    if isinstance(x, np.ndarray):
        return np.issubdtype(x.dtype, np.integer)
    elif torch.is_tensor(x):
        return not torch.is_floating_point(x) and not torch.is_complex(x)
    else:
        raise ValueError('The input is not a numpy array or a torch tensor.')


def is_between_minus1_1(x):
    correct_range = x.min() >= -1 and x.min() < 0 and x.max() <= 1
    if correct_range:
        assert is_float(x)
        return True
    else:
        return False


def is_between_0_1(x):
    correct_range = x.min() >= 0 and x.max() <= 1
    if correct_range:
        assert is_float(x)
        return True
    else:
        return False


def is_between_0_255(x):
    correct_range = x.min() >= 0 and x.max() <= 255
    if correct_range:
        assert is_integer(x)
        return True
    else:
        return False


def convert_to_uint8(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.uint8)
    elif torch.is_tensor(x):
        return x.to(torch.uint8)
    else:
        raise ValueError('The input is not a numpy array or a torch tensor.')


def convert_to_float32(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    elif torch.is_tensor(x):
        return x.to(torch.float32)
    else:
        raise ValueError('The input is not a numpy array or a torch tensor.')


def convert_to_0_1(x):
    if is_between_minus1_1(x):
        y = (x + 1) / 2
    elif is_between_0_1(x):
        y = x
    elif is_between_0_255(x):
        y = x / 255
        return convert_to_float32(x)
    else:
        raise ValueError('The input is not in the range of [0, 1] or [-1, 1] or [0, 255].')
    return convert_to_float32(y)


def convert_to_0_255(x):
    if is_between_minus1_1(x):
        y = ((x + 1) / 2 * 255)
    elif is_between_0_1(x):
        y = x * 255
    elif is_between_0_255(x):
        y = x
    else:
        raise ValueError('The input is not in the range of [0, 1] or [-1, 1] or [0, 255].')
    return convert_to_uint8(y)


def convert_to_minus1_1(x):
    if is_between_minus1_1(x):
        y = x
    elif is_between_0_1(x):
        y = x * 2 - 1
    elif is_between_0_255(x):
        y = x / 255 * 2 - 1
    else:
        raise ValueError('The input is not in the range of [0, 1] or [-1, 1] or [0, 255].')
    return convert_to_float32(y)


def get_bbox_from_mask(mask, bbox_label):
    # Identify the positions of the object in the mask
    positions = np.where(mask == bbox_label)

    # Make sure the object is not in the mask, return None or handle as needed
    assert len(positions[0]) > 0 and len(positions[1]) > 0

    # Calculate bounding box
    top = np.min(positions[0])
    left = np.min(positions[1])
    bottom = np.max(positions[0])
    right = np.max(positions[1])
    return top, left, bottom, right


def crop_around_object(img, mask, bbox_label, size):
    """
    Crop the image around the object specified by bbox_label in the mask.

    Parameters:
    - img: numpy array of the image (height, width, 3).
    - mask: numpy array of the mask (same size as img), integers.
    - bbox_label: Integer, label in the mask for the object of interest.
    - size: Integer, side length of the square crop.

    Returns:
    - Cropped (and possibly padded) image around the object.
    """
    
    top, left, bottom, right = get_bbox_from_mask(mask, bbox_label)

    if size > 0:
        # Calculate center of the bounding box
        center_y = (top + bottom) // 2
        center_x = (left + right) // 2

        h, w, c = img.shape
        half_size = size // 2
        
        top = max(0, center_y - half_size)
        left = max(0, center_x - half_size)
        bottom = min(h, center_y + half_size)
        right = min(w, center_x + half_size)

    # Cropping
    cropped_img = img[top:bottom, left:right]
    cropped_mask = mask[top:bottom, left:right]

    if size > 0:
        # Padding
        pad_top = abs(min(0, center_y - half_size))
        pad_bottom = abs(h - max(h, center_y + half_size))
        pad_left = abs(min(0, center_x - half_size))
        pad_right = abs(w - max(w, center_x + half_size))
        cropped_img = np.pad(cropped_img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 'constant', constant_values=0)
        cropped_mask = np.pad(cropped_mask, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant', constant_values=0)

        assert cropped_img.shape == (size, size, c)
        assert cropped_mask.shape == (size, size)

    return cropped_img, cropped_mask


def load_raw_image(plate_id, image_id):
    data_dir = "/scratch/groups/emmalu/HPA_temp"
    blue = Image.open(f'{data_dir}/{plate_id}/{image_id}_blue.png')
    bluearray = np.array(blue)
    yellow = Image.open(f'{data_dir}/{plate_id}/{image_id}_yellow.png')
    try:
        yellowarray = np.array(yellow)
    except OSError:
        print(f"The ER channel of image {image_id} is corrupted.")
        yellowarray = np.zeros_like(bluearray)
    red = Image.open(f'{data_dir}/{plate_id}/{image_id}_red.png')
    try:
        redarray = np.array(red)
    except SyntaxError:
        print(f"The DAPI channel of image {image_id} is corrupted.")
        redarray = np.zeros_like(bluearray)
    if len(redarray.shape) == 3:
        redarray = redarray[:, :, 0]
        assert redarray.sum() > 0, f"The DAPI channel of image {image_id} is all zeros."
    green = Image.open(f'{data_dir}/{plate_id}/{image_id}_green.png')
    greenarray = np.array(green)
    full_res_image = np.stack([redarray, greenarray, bluearray, yellowarray], axis=-1)
    p0, p99 = np.percentile(full_res_image, (0, 99))
    full_res_image = exposure.rescale_intensity(full_res_image, in_range=(p0, p99), out_range=(0, 255)).astype(np.uint8)
    assert is_between_0_255(full_res_image)
    return full_res_image


def load_intensity_rescaled_image(image_id):
    data_dir = "/scratch/groups/emmalu/HPA_rescaled"
    full_res_image = np.array(Image.open(f'{data_dir}/{image_id}.tif'))
    assert is_between_0_255(full_res_image)
    return full_res_image


def load_image(datasource, image_id, channels):
    if datasource == "jump":
        data_dir = "/scratch/groups/emmalu/JUMP/processed"
    elif datasource == "hpa":
        data_dir = "/scratch/groups/emmalu/HPA_rescaled"
    else:
        raise NotImplementedError
    

    if datasource == "jump":
        imarrays = []
        for i in range(3):
            if i >= len(channels):
                #if only two reference channels than last channel will just be array of 0s
                #Will call this channel "0" in .yaml fil
                #Note that channel "0" must come last
                imarrays.append([np.zeros(imarrays[0][0].shape, dtype=np.uint8)])
            else:
                image_path = image_id + f"p01-ch{channels[i]}sk1fk1fl1.png"
                imarrays.append([np.array(Image.open(f'{data_dir}/{image_path}'))])

        #combine channels into single multichannel image
        image = np.concatenate(imarrays, axis=0)
        image = np.transpose(image, (1, 2, 0)) #need num channels to be last dimension
    
    elif datasource == "hpa":
        image_path = image_id + ".tif"
        imarray = np.array(Image.open(f'{data_dir}/{image_path}'))
        image = imarray[:, :, channels]

        for i in range(3-image.shape[2]):
            z = np.array([np.zeros(image.shape[:2])]).transpose((1,2,0)).astype(np.uint8)
            image = np.append(image, z, axis=2)
    
    assert image.ndim == 3
    assert image.shape[2] == 3
    assert is_between_0_255(image)

    return image