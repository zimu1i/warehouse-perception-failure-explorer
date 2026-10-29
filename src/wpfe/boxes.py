"""Box geometry shared by training augmentation and flip-instability inference.

Convention (same as the manifests): continuous xyxy in absolute pixels, origin
top-left, x to the right. A box covers [x_min, x_max) x [y_min, y_max).
"""

from __future__ import annotations

import torch


def hflip_boxes(boxes: torch.Tensor, width: float) -> torch.Tensor:
    """Mirror xyxy boxes left-right inside an image of the given width.

    boxes: float tensor [N, 4] as (x_min, y_min, x_max, y_max). N may be 0.
    Returns a new [N, 4] tensor that is still valid xyxy (x_min < x_max).

    Used twice: to flip training targets together with the image, and to map
    predictions made on a flipped image back to original coordinates. Applying
    it twice must return the original boxes.
    """
    out = boxes.clone()
    out[:,0] = width - boxes[:, 2] # flips 
    out[:,2] = width - boxes[:, 0]
    return out
