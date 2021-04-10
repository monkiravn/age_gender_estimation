import torch
from skimage.transform import AffineTransform, warp
import numpy as np
from PIL import Image

class Rotate_Image(object):
    def __call__(self, sample):
        image, label1, label2 = sample['image'], sample['label_age'], sample['label_gender']
        min_scale = 0.8
        max_scale = 1.2
        sx = np.random.uniform(min_scale, max_scale)
        sy = np.random.uniform(min_scale, max_scale)
        # --- rotation ---
        max_rot_angle = 7
        rot_angle = np.random.uniform(-max_rot_angle, max_rot_angle) * np.pi / 180.
        # --- shear ---
        max_shear_angle = 10
        shear_angle = np.random.uniform(-max_shear_angle, max_shear_angle) * np.pi / 180.
        # --- translation ---
        max_translation = 4
        tx = np.random.randint(-max_translation, max_translation)
        ty = np.random.randint(-max_translation, max_translation)
        tform = AffineTransform(scale=(sx, sy), rotation=rot_angle, shear=shear_angle,
                                translation=(tx, ty))
        transformed_image = warp(image, tform.inverse)
        img1 = Image.fromarray(np.uint8(transformed_image * 255))
        sample = {'image': np.array(img1), 'label_age': label1, 'label_gender': label2}

        return sample


class RGB_ToTensor(object):
    def __call__(self, sample):
        image, label1, label2= sample['image'], sample['label_age'], sample['label_gender']
        image = torch.from_numpy(image)
        image = image.permute(2,0,1)
        label1 = torch.from_numpy(label1)
        label2 = torch.from_numpy(label2)

        return {'image': image,
                'label_age': label1,
                'label_gender': label2}



class Normalization(object):
    def __init__(self, mean, std):
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def __call__(self, sample):
        image, label1, label2 = sample['image'], sample['label_age'], sample['label_gender']

        return {'image': image,
                'label_age': label1,
                'label_gender': label2}

if __name__ == '__main__':
    x = np.ones((2,224,224,3))
    y1 = np.ones((2,1))
    y2 = np.ones((2,1))
    sample = {'image': x,
              'label_age': y1,
              'label_gender': y2}
    sample_tensor = RGB_ToTensor()(sample)
    print(sample_tensor['image'].size())
    print(sample_tensor['label_age'].size())