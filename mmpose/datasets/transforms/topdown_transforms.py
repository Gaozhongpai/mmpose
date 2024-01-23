# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmengine import is_seq_of

from mmpose.registry import TRANSFORMS
from mmpose.structures.bbox import get_udp_warp_matrix, get_warp_matrix
from random import randrange
# from kornia.augmentation import RandomBrightness

@TRANSFORMS.register_module()
class TopdownAffine(BaseTransform):
    """Get the bbox image as the model input by affine transform.

    Required Keys:

        - img
        - bbox_center
        - bbox_scale
        - bbox_rotation (optional)
        - keypoints (optional)

    Modified Keys:

        - img
        - bbox_scale

    Added Keys:

        - input_size
        - transformed_keypoints

    Args:
        input_size (Tuple[int, int]): The input image size of the model in
            [w, h]. The bbox region will be cropped and resize to `input_size`
        use_udp (bool): Whether use unbiased data processing. See
            `UDP (CVPR 2020)`_ for details. Defaults to ``False``

    .. _`UDP (CVPR 2020)`: https://arxiv.org/abs/1911.07524
    """

    def __init__(self,
                 input_size: Tuple[int, int],
                 use_udp: bool = False) -> None:
        super().__init__()

        assert is_seq_of(input_size, int) and len(input_size) == 2, (
            f'Invalid input_size {input_size}')

        self.input_size = input_size
        self.use_udp = use_udp

    @staticmethod
    def _fix_aspect_ratio(bbox_scale: np.ndarray, aspect_ratio: float):
        """Reshape the bbox to a fixed aspect ratio.

        Args:
            bbox_scale (np.ndarray): The bbox scales (w, h) in shape (n, 2)
            aspect_ratio (float): The ratio of ``w/h``

        Returns:
            np.darray: The reshaped bbox scales in (n, 2)
        """

        w, h = np.hsplit(bbox_scale, [1])
        bbox_scale = np.where(w > h * aspect_ratio,
                              np.hstack([w, w / aspect_ratio]),
                              np.hstack([h * aspect_ratio, h]))
        return bbox_scale

    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`TopdownAffine`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """

        w, h = self.input_size
        warp_size = (int(w), int(h))

        # reshape bbox to fixed aspect ratio
        results['bbox_scale'] = self._fix_aspect_ratio(
            results['bbox_scale'], aspect_ratio=w / h)

        # TODO: support multi-instance
        assert results['bbox_center'].shape[0] == 1, (
            'Top-down heatmap only supports single instance. Got invalid '
            f'shape of bbox_center {results["bbox_center"].shape}.')

        center = results['bbox_center'][0]
        scale = results['bbox_scale'][0]
        if 'bbox_rotation' in results:
            rot = results['bbox_rotation'][0]
        else:
            rot = 0.

        if self.use_udp:
            warp_mat = get_udp_warp_matrix(
                center, scale, rot, output_size=(w, h))
        else:
            warp_mat = get_warp_matrix(center, scale, rot, output_size=(w, h))

        if isinstance(results['img'], list):
            results['img'] = [
                cv2.warpAffine(
                    img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)
                for img in results['img']
            ]
        else:
            results['img'] = cv2.warpAffine(
                results['img'], warp_mat, warp_size, flags=cv2.INTER_LINEAR)

        if results.get('keypoints', None) is not None:
            if results.get('transformed_keypoints', None) is not None:
                transformed_keypoints = results['transformed_keypoints'].copy()
            else:
                transformed_keypoints = results['keypoints'].copy()
            # Only transform (x, y) coordinates
            transformed_keypoints[..., :2] = cv2.transform(
                results['keypoints'][..., :2], warp_mat)
            results['transformed_keypoints'] = transformed_keypoints

        results['input_size'] = (w, h)
        results['input_center'] = center
        results['input_scale'] = scale

        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(input_size={self.input_size}, '
        repr_str += f'use_udp={self.use_udp})'
        return repr_str


@TRANSFORMS.register_module()
class TopDownRandomLowRes(BaseTransform):
    """Data augmentation with random image flip.

    Required key: 'img', 'joints_3d', 'joints_3d_visible', 'center' and
    'ann_info'.

    Modifies key: 'img', 'joints_3d', 'joints_3d_visible', 'center' and
    'flipped'.

    Args:
        flip (bool): Option to perform random flip.
        flip_prob (float): Probability of flip.
    """

    def __init__(self, low_res_prob=0.5):
        super().__init__()
        self.low_res_prob = low_res_prob

    def transform(self, results: Dict) -> Optional[dict]:
        """Perform data augmentation with random image flip."""
        img = results['img']
        if np.random.rand() <= self.low_res_prob:
            down_scale = 2**(randrange(3) + 1)
            img = cv2.resize(img, [256//down_scale, 256//down_scale])
            img = img + (np.random.random([256//down_scale, 256//down_scale, 3]) - 0.5) * 16
            img = np.clip(img, 0, 255).astype(np.uint8)
            img = cv2.resize(img, [256, 256])
        results['img'] = img
        return results


@TRANSFORMS.register_module()
class NightAugNumpy(BaseTransform):
    def __init__(self):
        self.kernel_size = 11
        self.sigma_min, self.sigma_max = 0.1, 2.0

    def mask_img(self, img, cln_img):
        while np.random.random() > 0.4:
            x1, x2 = np.random.randint(0, img.shape[1], size=2)
            y1, y2 = np.random.randint(0, img.shape[2], size=2)
            img[:, x1:x2, y1:y2] = cln_img[:, x1:x2, y1:y2]
        return img

    def gaussian_heatmap(self, x):
        sig = np.random.randint(1, 32)
        image_size = x.shape[:2]  # assuming x.shape is [height, width, channels]
        center = (np.random.randint(image_size[0]), np.random.randint(image_size[1]))
        x_axis = np.linspace(0, image_size[0] - 1, image_size[0]) - center[0]
        y_axis = np.linspace(0, image_size[1] - 1, image_size[1]) - center[1]
        xx, yy = np.meshgrid(x_axis, y_axis)
        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

        # Add a new axis to kernel to make it [height, width, 1] so it broadcasts correctly over x's [height, width, channels]
        kernel = kernel[:, :, np.newaxis]
        new_img = np.clip(x * (1 - kernel) + 100 * kernel, 0, 255).astype(np.uint8)
        return new_img


    def transform(self, results: Dict) -> Optional[dict]:
        g_b_flag = True

        img = results['img']
        # Gaussian Blur
        if np.random.random() > 0.5:
            img = cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), np.random.uniform(self.sigma_min, self.sigma_max))

        cln_img_zero = img.copy()

        # Gamma
        if np.random.random() > 0.5:
            cln_img = img.copy()
            val = 1 / (np.random.random() * 0.6 + 0.4)
            img = np.clip(((img / 255) ** val) * 255, 0, 255).astype(np.uint8)
            img = self.mask_img(img, cln_img)
            g_b_flag = False

        # Brightness
        if np.random.random() > 0.5 or g_b_flag:
            cln_img = img.copy()
            val = np.random.random() * 0.6 + 0.4
            img = np.clip(img * val, 0, 255).astype(np.uint8)
            img = self.mask_img(img, cln_img)

        # Contrast
        if np.random.random() > 0.5:
            cln_img = img.copy()
            val = np.random.random() * 0.6 + 0.4
            img = np.clip(127.5 + val * (img - 127.5), 0, 255).astype(np.uint8)
            img = self.mask_img(img, cln_img)
        img = self.mask_img(img, cln_img_zero)

        prob = 0.5
        while np.random.random() > prob:
            img = self.gaussian_heatmap(img)
            prob += 0.1

        # Noise
        if np.random.random() > 0.5:
            n = np.clip(np.random.normal(0, np.random.randint(20), img.shape), 0, 255)
            img = np.clip(n + img, 0, 255).astype(np.uint8)
        
        results['img'] = img
        return results


# @TRANSFORMS.register_module()
# class TopDownRandomLowLight:
#     """Data augmentation with random image flip.

#     Required key: 'img', 'joints_3d', 'joints_3d_visible', 'center' and
#     'ann_info'.

#     Modifies key: 'img', 'joints_3d', 'joints_3d_visible', 'center' and
#     'flipped'.

#     Args:
#         flip (bool): Option to perform random flip.
#         flip_prob (float): Probability of flip.
#     """

#     def __init__(self, low_light_prob=0.5):
#         self.low_light_prob = low_light_prob
#         self.aug = RandomBrightness(brightness=(0.75, 1.25), p=low_light_prob, keepdim=True)

#     def transform(self, results: Dict) -> Optional[dict]:
#         """Perform data augmentation with random image flip."""
#         img = results['img']
#         # if np.random.rand() <= self.low_light_prob:
#         #     img = img * (np.random.rand() * 0.2 + 0.75)
#         img = self.aug(img)
#         results['img'] = img
#         return results