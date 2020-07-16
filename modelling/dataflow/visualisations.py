from typing import Union, Callable, Optional

import numpy as np
from PIL import Image

import torch



def tensor_to_rgb(t: torch.Tensor) -> np.ndarray:
    img = t.cpu().numpy().transpose((1, 2, 0))
    return img


# def make_grid(
#         batch_img: torch.Tensor,
#         batch_mask: torch.Tensor,
#         img_denormalize_fn: Callable,
#         batch_gt_mask: Optional[torch.Tensor] = None,
# ):
#     """Create a grid from batch image and mask as
#         img1  | img2  | img3  | img4  | ...
#         i+m1  | i+m2  | i+m3  | i+m4  | ...
#         mask1 | mask2 | mask3 | mask4 | ...
#         i+M1  | i+M2  | i+M3  | i+M4  | ...
#         Mask1 | Mask2 | Mask3 | Mask4 | ...
#         i+m = image + mask blended with alpha=0.4
#         - maskN is predicted mask
#         - MaskN is ground-truth mask if given
#     Args:
#         batch_img (torch.Tensor) batch of images of any type
#         batch_mask (torch.Tensor) batch of masks
#         img_denormalize_fn (Callable): function to denormalize batch of images
#         batch_gt_mask (torch.Tensor, optional): batch of ground truth masks.
#     """
#     assert isinstance(batch_img, torch.Tensor) and isinstance(batch_mask, torch.Tensor)
#     assert len(batch_img) == len(batch_mask)
#
#     if batch_gt_mask is not None:
#         assert isinstance(batch_gt_mask, torch.Tensor)
#         assert len(batch_mask) == len(batch_gt_mask)
#
#     b = batch_img.shape[0]
#     h, w = batch_img.shape[2:]
#
#     le = 3 if batch_gt_mask is None else 3 + 2
#     out_image = np.zeros((h * le, w * b, 3), dtype="uint8")
#
#     for i in range(b):
#         img = batch_img[i]
#         mask = batch_mask[i]
#
#         img = img_denormalize_fn(img)
#         img = tensor_to_rgb(img)
#         mask = mask.cpu().numpy()
#         mask = render_mask(mask)
#
#         out_image[0:h, i * w: (i + 1) * w, :] = img
#         out_image[1 * h: 2 * h, i * w: (i + 1) * w, :] = render_datapoint(img, mask, blend_alpha=0.4)
#         out_image[2 * h: 3 * h, i * w: (i + 1) * w, :] = mask
#
#         if batch_gt_mask is not None:
#             gt_mask = batch_gt_mask[i]
#             gt_mask = gt_mask.cpu().numpy()
#             gt_mask = render_mask(gt_mask)
#             out_image[3 * h: 4 * h, i * w: (i + 1) * w, :] = render_datapoint(img, gt_mask, blend_alpha=0.4)
#             out_image[4 * h: 5 * h, i * w: (i + 1) * w, :] = gt_mask
#
#     return out_image
