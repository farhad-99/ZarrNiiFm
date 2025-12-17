"""
Vessel FM thresholding segmentation plugin.

This module implements Otsu's automatic threshold selection method for binary
image segmentation, applied locally to each processing block.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from skimage.filters import threshold_otsu

from .base import SegmentationPlugin, hookimpl

import logging
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
import hydra
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from monai.inferers import SlidingWindowInfererAdapt
from skimage.morphology import remove_small_objects
from skimage.exposure import equalize_hist

from vesselfm.seg.utils.data import generate_transforms
from vesselfm.seg.utils.io import determine_reader_writer
from vesselfm.seg.utils.evaluation import Evaluator, calculate_mean_metrics


warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

class VesselFM(SegmentationPlugin):
    """
    Vessel FM thresholding segmentation plugin.

    This plugin uses Otsu's method to automatically determine an optimal threshold
    for binary image segmentation. The method assumes a bimodal histogram and
    finds the threshold that minimizes intra-class variance. The threshold is
    computed locally for each processing block, making it suitable for images
    with varying illumination or contrast.

    Parameters:
        nbins: Number of bins for histogram computation (default: 256)
    """

    def __init__(self,cfg=None, nbins: int = 256, **kwargs):
        """
        Initialize Vessel FM segmentation plugin.

        Args:
            nbins: Number of bins for histogram computation
            **kwargs: Additional parameters passed to parent class
        """
        super().__init__(nbins=nbins, **kwargs)
        if cfg is not None:
            self.cfg = cfg
            device = self.cfg.device
            logger.info(f"Loading model from {cfg.ckpt_path}.")
            ckpt = torch.load(Path(cfg.ckpt_path), map_location=device, weights_only=True)

            model = hydra.utils.instantiate(cfg.model)
            model.load_state_dict(ckpt)
            self.model = model.to(device)

            transforms = generate_transforms(cfg.transforms_config)
            self.transforms = transforms

            logger.debug(f"Sliding window patch size: {cfg.patch_size}")
            logger.debug(f"Sliding window batch size: {cfg.batch_size}.")
            logger.debug(f"Sliding window overlap: {cfg.overlap}.")
            inferer = SlidingWindowInfererAdapt(
                roi_size=cfg.patch_size, sw_batch_size=cfg.batch_size, overlap=cfg.overlap, 
                mode=cfg.mode, sigma_scale=cfg.sigma_scale, padding_mode=cfg.padding_mode
            )
            self.inferer = inferer

        self.nbins = nbins


        

    @hookimpl
    def segment(
        self, image: np.ndarray, metadata: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Segment image using Vessel FM thresholding.

        Args:
            image: Input image as numpy array
            metadata: Optional metadata (unused in Otsu method)

        Returns:
            Binary segmentation mask as numpy array with same shape as input.
            Values are 0 (background) and 1 (foreground).

        Raises:
            ValueError: If input image is empty or has invalid dimensions
        """
        #print("size:",image.size)
        #print("shape:",image.shape)
        #try:
        #    print("type:", image)
        #except:
        #    print('no type')
        device = self.cfg.device
        '''if image.size == 0:
            raise ValueError("Input image is empty")

        if image.ndim < 2:
            raise ValueError("Input image must be at least 2D")

        # Store original shape for output
        original_shape = image.shape

        # Ensure image is in a suitable format for Otsu thresholding
        if image.dtype == bool:
            # Already binary, return as-is
            return image.astype(np.uint8)
'''
        # Handle multi-dimensional images

        '''work_image = image

        # For images with more than 3 dimensions, flatten extra dimensions
        #print('dim',work_image.ndim)
        if work_image.ndim > 3:
            # Reshape to 3D by flattening leading dimensions
            leading_dims = work_image.shape[:-2]  # All dimensions except last 2
          #  print('before', work_image.shape)
            work_image = work_image.reshape(-1, *work_image.shape[-2:])
          #  print('>3',work_image.shape)
           # print(type(work_image))

        if work_image.ndim == 3:
            # For 3D images, process the first slice/channel
            if work_image.shape[0] <= 4:  # Likely channels-first format
                work_image = work_image[0]
            #    print("=3",work_image.shape)
            #   print(type(work_image))
            else:
                # If first dimension is large, likely depth dimension, take middle slice
                mid_slice = work_image.shape[0] // 2
                work_image = work_image[mid_slice]
            #    print('no',work_image.shape)
            #    print(type(work_image))
        print('image shape = ', image.shape)
        print('image.type',type(image))
        # Now work_image should be 2D
        try:
            threshold = threshold_otsu(work_image, nbins=self.nbins)
        except ValueError as e:
            # Handle edge cases where Otsu fails (e.g., constant image)
            if "all values are identical" in str(e).lower() or np.all(
                work_image == work_image.flat[0]
            ):
                # Return all zeros for constant images with original shape
                return np.zeros(original_shape, dtype=np.uint8)
            else:
                raise e'''
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed_all(self.cfg.seed)

        # set device


        # load model and ckpt
        model = self.model
        model.eval()
        transforms = generate_transforms(self.cfg.transforms_config)
        print(transforms)
        with torch.no_grad():
        
            preds = []
            image = image.astype(np.float32)
            image = self.transforms(image).to(self.cfg.device)
            #original_shape = image.shape
            print('shape new', image.shape)
            logits = self.inferer(image, model)
            preds.append(logits.cpu().squeeze(1))
            pred = torch.stack(preds).mean(dim=0).sigmoid()
            pred_thresh = (pred > self.cfg.merging.threshold).numpy()  
            print('pred_thresh done', pred_thresh.shape)
 
            if self.cfg.post.apply:
                pred_thresh = remove_small_objects(
                    pred_thresh, min_size=self.cfg.post.small_objects_min_size, connectivity=self.cfg.post.small_objects_connectivity
                )   

        # Apply threshold to original image
        #binary_mask = image > threshold
        
        return pred_thresh.astype(np.uint8)

    @hookimpl
    def segmentation_plugin_name(self) -> str:
        """Return the name of the segmentation algorithm."""
        return "Vessel FM"

    @hookimpl
    def segmentation_plugin_description(self) -> str:
        """Return a description of the segmentation algorithm."""
        return (
            "Vessel FM's automatic threshold selection method for binary segmentation. "
            "Finds the threshold that minimizes intra-class variance assuming a "
            "bimodal intensity distribution. Threshold is computed locally for each "
            "processing block, suitable for images with varying illumination."
        )

    @property
    def name(self) -> str:
        """Return the name of the segmentation algorithm."""
        return self.segmentation_plugin_name()

    @property
    def description(self) -> str:
        """Return a description of the segmentation algorithm."""
        return self.segmentation_plugin_description()

    def get_threshold(self, image: np.ndarray) -> float:
        """
        Get the Otsu threshold value without applying segmentation.

        Args:
            image: Input image as numpy array

        Returns:
            Computed Otsu threshold value
        """
        if image.size == 0:
            raise ValueError("Input image is empty")

        # Handle multi-channel images by taking the first channel if needed
        if image.ndim > 3:
            while image.ndim > 3:
                image = image[0]

        if image.ndim == 3 and image.shape[0] <= 4:
            image = image[0]

        return threshold_otsu(image, nbins=self.nbins)
