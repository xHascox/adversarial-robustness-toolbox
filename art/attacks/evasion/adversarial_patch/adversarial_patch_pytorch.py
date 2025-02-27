# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2021
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements the adversarial patch attack `AdversarialPatch`. This attack generates an adversarial patch that
can be printed into the physical world with a common printer. The patch can be used to fool image and video estimators.

| Paper link: https://arxiv.org/abs/1712.09665
"""
from __future__ import absolute_import, division, print_function, unicode_literals, annotations

import logging
import math
from packaging.version import parse
from typing import Any, TYPE_CHECKING
import torchvision

import numpy as np
from tqdm.auto import trange
from tqdm import tqdm
import torchtnt

from art.attacks.attack import EvasionAttack
from art.attacks.evasion.adversarial_patch.utils import insert_transformed_patch
from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin
from art.utils import check_and_transform_label_format, is_probability, to_categorical
from art.summary_writer import SummaryWriter

if TYPE_CHECKING:

    import torch

    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE

logger = logging.getLogger(__name__)

interpolation = torchvision.transforms.InterpolationMode.BILINEAR


class AdversarialPatchPyTorch(EvasionAttack):
    """
    Implementation of the adversarial patch attack for square and rectangular images and videos in PyTorch.

    | Paper link: https://arxiv.org/abs/1712.09665
    """

    attack_params = EvasionAttack.attack_params + [
        "rotation_max",
        "scale_min",
        "scale_max",
        "distortion_scale_max",
        "learning_rate",
        "max_epochs",
        "batch_size",
        "patch_shape",
        "optimizer",
        "targeted",
        "summary_writer",
        "verbose",
        "pretrained_patch",
        "disguise",
        "disguise_distance_factor",
        "split",
        "gap_size",
    ]

    _estimator_requirements = (BaseEstimator, NeuralNetworkMixin)

    def __init__(
        self,
        estimator: "CLASSIFIER_NEURALNETWORK_TYPE",
        rotation_max: float = 22.5,
        scale_min: float = 0.1,
        scale_max: float = 1.0,
        distortion_scale_max: float = 0.0,
        learning_rate: float = 5.0,
        max_epochs: int = 5,
        max_steps: int = 0,
        batch_size: int = 16,
        patch_shape: tuple[int, int, int] = (3, 224, 224),
        patch_location: tuple[int, int] | None = None,
        patch_type: str = "circle",
        optimizer: str = "Adam",
        targeted: bool = True,
        summary_writer: str | bool | SummaryWriter = False,
        verbose: bool = True,
        pretrained_patch: np.ndarray | None = None,
        contrast_min: int = 1,
        contrast_max: int = 1,
        disguise: np.ndarray | None = None,
        disguise_distance_factor: float = 1,
        split: bool = False,
        gap_size: int = 0,
    ):
        """
        Create an instance of the :class:`.AdversarialPatchPyTorch`.

        :param estimator: A trained estimator.
        :param rotation_max: The maximum rotation applied to random patches. The value is expected to be in the
               range `[0, 180]`.
        :param scale_min: The minimum scaling applied to random patches. The value should be in the range `[0, 1]`,
               but less than `scale_max`.
        :param scale_max: The maximum scaling applied to random patches. The value should be in the range `[0, 1]`, but
               larger than `scale_min`.
        :param distortion_scale_max: The maximum distortion scale for perspective transformation in range `[0, 1]`. If
               distortion_scale_max=0.0 the perspective transformation sampling will be disabled.
        :param learning_rate: The learning rate of the optimization. For `optimizer="pgd"` the learning rate gets
                              multiplied with the sign of the loss gradients.
        :param max_epochs: The max number of optimization epochs.
        :param max_steps: The max number of optimization steps. Set to zero for no limit.
        :param batch_size: The size of the training batch.
        :param patch_shape: The shape of the adversarial patch as a tuple of shape CHW (nb_channels, height, width).
        :param patch_location: The location of the adversarial patch as a tuple of shape (upper left x, upper left y).
        :param patch_type: The patch type, either circle or square.
        :param optimizer: The optimization algorithm. Supported values: "Adam", and "pgd". "pgd" corresponds to
                          projected gradient descent in L-Inf norm.
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False).
        :param summary_writer: Activate summary writer for TensorBoard.
                               Default is `False` and deactivated summary writer.
                               If `True` save runs/CURRENT_DATETIME_HOSTNAME in current directory.
                               If of type `str` save in path.
                               If of type `SummaryWriter` apply provided custom summary writer.
                               Use hierarchical folder structure to compare between runs easily. e.g. pass in
                               ‘runs/exp1’, ‘runs/exp2’, etc. for each new experiment to compare across them.
        :param verbose: Show progress bars.
        :pretrained_patch: A pretrained patch to continue training, or a target image to apply disturbance to
        :contrast_min: between 0 and contrast_max, 0 reduces all contrast of the patch
        :contrast_max: should probably be 1
        :disguise: What the patch should look like
        :disguise_distance_factor: factor/weight of the disguise distance
        :split: Collusion attack, splits the patch into two
        :gap_size: The gap size when using a collusion attack (2 patchces)
        """
        import torch
        import torchvision

        torch_version = list(parse(torch.__version__.lower()).release)
        torchvision_version = list(
            parse(torchvision.__version__.lower()).release)
        assert (
            torch_version[0] >= 1 and torch_version[1] >= 7 or (
                torch_version[0] >= 2)
        ), "AdversarialPatchPyTorch requires torch>=1.7.0"
        assert (
            torchvision_version[0] >= 0 and torchvision_version[1] >= 8
        ), "AdversarialPatchPyTorch requires torchvision>=0.8.0"

        super().__init__(estimator=estimator, summary_writer=summary_writer)
        self.rotation_max = rotation_max
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.distortion_scale_max = distortion_scale_max
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.patch_shape = patch_shape
        self.patch_location = patch_location
        self.patch_type = patch_type

        self.image_shape = estimator.input_shape
        self.targeted = targeted
        self.verbose = verbose
        self.pretrained_patch = pretrained_patch
        self.contrast_min = contrast_min
        self.contrast_max = contrast_max
        if type(disguise) != type(None):
            self.disguise = torch.tensor(disguise.astype(
                np.float32), requires_grad=True).to(self.estimator.device)
        else:
            self.disguise = None
        self.disguise_distance_factor = disguise_distance_factor
        self.detailed_loss_history = {"classification":[], "disguise":[]}
        self.split = split
        self.gap_size = gap_size
        self._check_params()

        self.i_h_patch = 1
        self.i_w_patch = 2

        self.input_shape = self.estimator.input_shape

        self.nb_dims = len(self.image_shape)
        if self.nb_dims == 3:
            self.i_h = 1
            self.i_w = 2
        elif self.nb_dims == 4:
            self.i_h = 2
            self.i_w = 3

        if self.patch_shape[1] != self.patch_shape[2]:  # pragma: no cover
            raise ValueError("Patch height and width need to be the same.")
            pass

        if not (  # pragma: no cover
            self.estimator.postprocessing_defences is None or self.estimator.postprocessing_defences == []
        ):
            raise ValueError(
                "Framework-specific implementation of Adversarial Patch attack does not yet support "
                + "postprocessing defences."
            )

        mean_value = (self.estimator.clip_values[1] - self.estimator.clip_values[0]) / 2.0 + self.estimator.clip_values[
            0
        ]
        self._initial_value = np.ones(self.patch_shape) * mean_value
        if self.pretrained_patch is not None:
            self._patch = torch.tensor(
                self.pretrained_patch, requires_grad=True, dtype=torch.float32, device=self.estimator.device)
        else:
            self._patch = torch.tensor(
                self._initial_value, requires_grad=True, device=self.estimator.device)

        self._optimizer_string = optimizer
        if self._optimizer_string == "Adam":
            self._optimizer = torch.optim.Adam(
                [self._patch], lr=self.learning_rate)

    def _train_step(
        self, images: "torch.Tensor", target: "torch.Tensor", mask: "torch.Tensor" | None = None
    ) -> "torch.Tensor":
        import torch

        self.estimator.model.zero_grad()
        loss = self._loss(images, target, mask)
        loss.backward(retain_graph=True)

        if self._optimizer_string == "pgd":
            if self._patch.grad is not None:
                gradients = self._patch.grad.sign() * self.learning_rate
            else:
                raise ValueError("Gradient term in PyTorch model is `None`.")

            with torch.no_grad():
                self._patch[:] = torch.clamp(
                    self._patch + gradients, min=self.estimator.clip_values[0], max=self.estimator.clip_values[1]
                )
        else:
            self._optimizer.step()

            with torch.no_grad():
                self._patch[:] = torch.clamp(
                    self._patch, min=self.estimator.clip_values[0], max=self.estimator.clip_values[1]
                )

        return loss

    def _predictions(
        self, images: "torch.Tensor", mask: "torch.Tensor" | None, target: "torch.Tensor"
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        import torch

        patched_input = self._random_overlay(images, self._patch, mask=mask)
        patched_input = torch.clamp(
            patched_input,
            min=self.estimator.clip_values[0],
            max=self.estimator.clip_values[1],
        )

        predictions, target = self.estimator._predict_framework(
            patched_input, target)

        return predictions, target

    def _loss(self, images: "torch.Tensor", target: "torch.Tensor", mask: "torch.Tensor" | None) -> "torch.Tensor":
        import torch

        if (not self.targeted and self._optimizer_string != "pgd") or self.targeted and self._optimizer_string == "pgd":
            change_sign = True
        else:
            change_sign = False

        if isinstance(target, torch.Tensor):

            predictions, target = self._predictions(images, mask, target)

            if self.use_logits:
                loss = torch.nn.functional.cross_entropy(
                    input=predictions, target=torch.argmax(target, dim=1), reduction="mean"
                )
            else:
                loss = torch.nn.functional.nll_loss(
                    input=predictions, target=torch.argmax(target, dim=1), reduction="mean"
                )
            self.detailed_loss_history["classification"] += [loss.item()]

        else:
            patched_input, patch_location_list, _ = self._random_overlay_get_patch_location_split(
                images, self._patch, mask=mask)  # TODO create new targets
            patched_input = torch.clamp(
                patched_input,
                min=self.estimator.clip_values[0],
                max=self.estimator.clip_values[1],
            )
            # targets must be a list of dictionaries, each dictionary element as tensor the following:
            # - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
            # synthetic_y = {'boxes': np.array([[     patch_location[0],      patch_location[1],      patch_location[0]+patch_shape[1],      patch_location[1]+patch_shape[2]]], dtype=np.float32),
            #                'scores': np.array([    1], dtype=np.float32),
            #                'labels': np.array([0])}
            batch_size = patched_input.shape[0]
            syn_scores = 1
            # TODO THIS ONLY WORKS FOR TARGETED ATTACK
            syn_labels = target[0].get("labels")[0].cpu()
            # print("label target:", syn_labels)
            syn_targets = []
            for i in range(batch_size):
                boxes = []
                scores = []
                labels = []
                if type(patch_location_list[i]) != list:
                    patch_location_i = [patch_location_list[i]]
                else:
                    patch_location_i = patch_location_list[i]

                for patch_location in patch_location_i:
                    boxes.append([patch_location[0], patch_location[1],
                                 patch_location[2], patch_location[3]])
                    scores.append(syn_scores)
                    labels.append(syn_labels)
                synthetic_y = {'boxes': np.array(boxes, dtype=np.float32),
                               'scores': np.array(scores, dtype=np.float32),
                               'labels': np.array(labels)}
                syn_targets.append(synthetic_y)

                """patch_location = patch_location_list[i]
                # print(patch_location)
                synthetic_y = {'boxes': np.array([[patch_location[0], patch_location[1], patch_location[2], patch_location[3]]], dtype=np.float32),
                               'scores': np.array([syn_scores], dtype=np.float32),
                               'labels': np.array([syn_labels])}
                syn_targets.append(synthetic_y)"""

            # print("----------------------")
            # print(patched_input.shape)
            # print(type(target))
            # print(target)

            # print("########")
            # for t in syn_targets:
            #    print(t)
            # import matplotlib.pyplot as plt

            # plt.imshow(
            #    (patched_input[0].cpu().detach().numpy().transpose(1, 2, 0).astype(np.uint8)))

            loss = self.estimator.compute_loss(x=patched_input, y=syn_targets)
            # loss = self.estimator.compute_loss(x=patched_input, y=target)
            self.detailed_loss_history["classification"] += [loss.item()]
            
            if type(self.disguise) != type(None):
                disguise_loss = torch.dist(self._patch, self.disguise)
                self.detailed_loss_history["disguise"] += [disguise_loss.item()]
                loss += self.disguise_distance_factor * disguise_loss

        if change_sign:
            loss = -loss

        return loss

    def _get_circular_patch_mask(self, nb_samples: int, sharpness: int = 40) -> "torch.Tensor":
        """
        Return a circular patch mask.
        """
        import torch

        diameter = np.minimum(
            self.patch_shape[self.i_h_patch], self.patch_shape[self.i_w_patch])

        if self.patch_type == "circle":
            x = np.linspace(-1, 1, diameter)
            y = np.linspace(-1, 1, diameter)
            x_grid, y_grid = np.meshgrid(x, y, sparse=True)
            z_grid = (x_grid**2 + y_grid**2) ** sharpness
            image_mask: int | np.ndarray[Any,
                                         np.dtype[Any]] = 1 - np.clip(z_grid, -1, 1)
        elif self.patch_type == "square":
            image_mask = np.ones((diameter, diameter))

        image_mask = np.expand_dims(image_mask, axis=0)
        image_mask = np.broadcast_to(image_mask, self.patch_shape)
        image_mask_tensor = torch.Tensor(
            np.array(image_mask)).to(self.estimator.device)
        image_mask_tensor = torch.stack(
            [image_mask_tensor] * nb_samples, dim=0)
        return image_mask_tensor

    def _random_overlay_get_patch_location(
        self,
        images: "torch.Tensor",
        patch: "torch.Tensor",
        scale: float | None = None,
        mask: "torch.Tensor" | None = None,
        prev_patches: list = [],
        leave_margin_right: bool = False,
        gap_size: int = 0,
        prev_shift_list: list = [],
    ) -> "torch.Tensor":
        """
        Apply the patch but also return its location.
        """
        import torch
        import torchvision

        # Ensure channels-first
        if not self.estimator.channels_first:
            images = torch.permute(images, (0, 3, 1, 2))

        nb_samples = images.shape[0]

        image_mask = self._get_circular_patch_mask(nb_samples=nb_samples)
        image_mask = image_mask.float()

        self.image_shape = images.shape[1:]

        smallest_image_edge = np.minimum(
            self.image_shape[self.i_h], self.image_shape[self.i_w])

        image_mask = torchvision.transforms.functional.resize(
            img=image_mask,
            size=(smallest_image_edge, smallest_image_edge),
            interpolation=2,
        )
        # print("mask1", image_mask.shape)
        # print("mask", self.image_shape[self.i_h],
        #      image_mask.shape[self.i_h_patch + 1])
        pad_h_before = int(
            (self.image_shape[self.i_h] - image_mask.shape[self.i_h_patch + 1]) / 2)
        pad_h_after = int(
            self.image_shape[self.i_h] - pad_h_before - image_mask.shape[self.i_h_patch + 1])

        pad_w_before = int(
            (self.image_shape[self.i_w] - image_mask.shape[self.i_w_patch + 1]) / 2)
        pad_w_after = int(
            self.image_shape[self.i_w] - pad_w_before - image_mask.shape[self.i_w_patch + 1])
        # print("pad", [pad_w_before, pad_h_before, pad_w_after, pad_h_after])
        # print(self.image_shape[self.i_h], self.image_shape[self.i_w])
        # print(image_mask.shape[self.i_h_patch + 1], image_mask.shape[self.i_w_patch + 1])
        image_mask = torchvision.transforms.functional.pad(
            img=image_mask,
            padding=[pad_w_before, pad_h_before, pad_w_after, pad_h_after],
            fill=0,
            padding_mode="constant",
        )
        # print("mask2", image_mask.shape)
        if self.nb_dims == 4:
            image_mask = torch.unsqueeze(image_mask, dim=1)
            image_mask = torch.repeat_interleave(
                image_mask, dim=1, repeats=self.input_shape[0])

        image_mask = image_mask.float()

        # Apply contrast adjustment:
        contrast_alpha = np.random.uniform(
            self.contrast_min, self.contrast_max)
        mid_gray_value = 127
        patch = (patch * contrast_alpha) + \
            ((1 - contrast_alpha) * mid_gray_value)

        patch = patch.float()
        padded_patch = torch.stack([patch] * nb_samples)

        # print("2-", patch.shape)

        padded_patch = torchvision.transforms.functional.resize(
            img=padded_patch,
            # try to keep the aspect ratio
            size=(smallest_image_edge, smallest_image_edge),
            interpolation=2,
        )
        # print("3-", padded_patch.shape)
        # print("pad", pad_w_before, pad_h_before, pad_w_after, pad_h_after)
        padded_patch = torchvision.transforms.functional.pad(
            img=padded_patch,
            padding=[pad_w_before, pad_h_before, pad_w_after, pad_h_after],
            fill=0,
            padding_mode="constant",
        )
        # print("4-", padded_patch.shape)

        if self.nb_dims == 4:
            padded_patch = torch.unsqueeze(padded_patch, dim=1)
            padded_patch = torch.repeat_interleave(
                padded_patch, dim=1, repeats=self.input_shape[0])

        padded_patch = padded_patch.float()

        image_mask_list = []
        padded_patch_list = []
        patch_location_list = []
        shift_list = []

        for i_sample in range(nb_samples):
            if self.patch_location is None:
                if scale is None:
                    im_scale = np.random.uniform(
                        low=self.scale_min, high=self.scale_max)
                else:
                    im_scale = scale
            else:
                im_scale = self.patch_shape[self.i_h] / smallest_image_edge

            if mask is None:
                if self.patch_location is None and not prev_patches:
                    if leave_margin_right:
                        padding_after_scaling_h = (
                            self.image_shape[self.i_h] - im_scale *
                            padded_patch.shape[self.i_h + 1]
                        ) / 2.0
                        padding_after_scaling_w = (
                            self.image_shape[self.i_w] - 2 * im_scale *
                            padded_patch.shape[self.i_w + 1] -
                            gap_size
                        ) / 2.0
                    else:
                        padding_after_scaling_h = (
                            self.image_shape[self.i_h] - im_scale *
                            padded_patch.shape[self.i_h + 1]
                        ) / 2.0
                        padding_after_scaling_w = (
                            self.image_shape[self.i_w] - im_scale *
                            padded_patch.shape[self.i_w + 1]
                        ) / 2.0
                    x_shift = np.random.uniform(-padding_after_scaling_w,
                                                padding_after_scaling_w)
                    y_shift = np.random.uniform(-padding_after_scaling_h,
                                                padding_after_scaling_h)
                elif self.patch_location is None and prev_patches:
                    padding_h = int(math.floor(
                        self.image_shape[self.i_h] - self.patch_shape[self.i_h]) / 2.0)
                    padding_w = int(math.floor(
                        self.image_shape[self.i_w] - self.patch_shape[self.i_w]) / 2.0)
                    padding_after_scaling_h = (
                        self.image_shape[self.i_h] - im_scale *
                        padded_patch.shape[self.i_h + 1]
                    ) / 2.0
                    padding_after_scaling_w = (
                        self.image_shape[self.i_w] - im_scale *
                        padded_patch.shape[self.i_w + 1]
                    ) / 2.0
                    x_shift = -padding_w + prev_patches[i_sample][2]
                    y_shift = -padding_h + prev_patches[i_sample][1]
                    x_shift = prev_shift_list[i_sample][0] + gap_size + (im_scale *
                                                                         padded_patch.shape[self.i_h + 1])  # TODO
                    y_shift = prev_shift_list[i_sample][1]

                else:
                    padding_h = int(math.floor(
                        self.image_shape[self.i_h] - self.patch_shape[self.i_h]) / 2.0)
                    padding_w = int(math.floor(
                        self.image_shape[self.i_w] - self.patch_shape[self.i_w]) / 2.0)
                    x_shift = -padding_w + self.patch_location[0]
                    y_shift = -padding_h + self.patch_location[1]

            else:
                mask_2d = mask[i_sample, :, :]

                edge_x_0 = int(
                    im_scale * padded_patch.shape[self.i_w + 1]) // 2
                edge_x_1 = int(
                    im_scale * padded_patch.shape[self.i_w + 1]) - edge_x_0
                edge_y_0 = int(
                    im_scale * padded_patch.shape[self.i_h + 1]) // 2
                edge_y_1 = int(
                    im_scale * padded_patch.shape[self.i_h + 1]) - edge_y_0

                mask_2d[0:edge_x_0, :] = False
                if edge_x_1 > 0:
                    mask_2d[-edge_x_1:, :] = False
                mask_2d[:, 0:edge_y_0] = False
                if edge_y_1 > 0:
                    mask_2d[:, -edge_y_1:] = False

                num_pos = np.argwhere(mask_2d).shape[0]
                pos_id = np.random.choice(num_pos, size=1)
                pos = np.argwhere(mask_2d)[pos_id[0]]
                x_shift = pos[1] - self.image_shape[self.i_w] // 2
                y_shift = pos[0] - self.image_shape[self.i_h] // 2

            phi_rotate = float(
                np.random.uniform(-self.rotation_max, self.rotation_max))

            image_mask_i = image_mask[i_sample]

            height = padded_patch.shape[self.i_h + 1]
            width = padded_patch.shape[self.i_w + 1]

            half_height = height // 2
            half_width = width // 2
            topleft = [
                int(torch.randint(0, int(self.distortion_scale_max *
                    half_width) + 1, size=(1,)).item()),
                int(torch.randint(0, int(self.distortion_scale_max *
                    half_height) + 1, size=(1,)).item()),
            ]
            topright = [
                int(torch.randint(width - int(self.distortion_scale_max *
                    half_width) - 1, width, size=(1,)).item()),
                int(torch.randint(0, int(self.distortion_scale_max *
                    half_height) + 1, size=(1,)).item()),
            ]
            botright = [
                int(torch.randint(width - int(self.distortion_scale_max *
                    half_width) - 1, width, size=(1,)).item()),
                int(torch.randint(height - int(self.distortion_scale_max *
                    half_height) - 1, height, size=(1,)).item()),
            ]
            botleft = [
                int(torch.randint(0, int(self.distortion_scale_max *
                    half_width) + 1, size=(1,)).item()),
                int(torch.randint(height - int(self.distortion_scale_max *
                    half_height) - 1, height, size=(1,)).item()),
            ]
            startpoints = [[0, 0], [width - 1, 0],
                           [width - 1, height - 1], [0, height - 1]]
            endpoints = [topleft, topright, botright, botleft]

            image_mask_i = torchvision.transforms.functional.perspective(
                img=image_mask_i, startpoints=startpoints, endpoints=endpoints, interpolation=2, fill=None
            )

            image_mask_i = torchvision.transforms.functional.affine(
                img=image_mask_i,
                angle=phi_rotate,
                translate=[x_shift, y_shift],
                scale=im_scale,
                shear=[0, 0],
                interpolation=interpolation,
                fill=None,
            )

            image_mask_list.append(image_mask_i)

            padded_patch_i = padded_patch[i_sample]

            padded_patch_i = torchvision.transforms.functional.perspective(
                img=padded_patch_i, startpoints=startpoints, endpoints=endpoints, interpolation=2, fill=None
            )

            padded_patch_i = torchvision.transforms.functional.affine(
                img=padded_patch_i,
                angle=phi_rotate,
                translate=[x_shift, y_shift],
                scale=im_scale,
                shear=[0, 0],
                interpolation=interpolation,  # NEAREST or BILINEAR
                fill=None,
            )

            # calculate the patch location:
            # Calculate the top-left corner of the patch (approximated)
            center_x = images.shape[3] // 2  # Image width / 2
            center_y = images.shape[2] // 2  # Image height / 2

            # Consider scaling for bottom-right calculation:
            scaled_patch_width = padded_patch.shape[3] * im_scale
            scaled_patch_height = padded_patch.shape[2] * im_scale

            patch_top_left_x = center_x + x_shift - \
                (scaled_patch_width // 2)  # Adjust for patch size
            patch_top_left_y = center_y + y_shift - (scaled_patch_height // 2)
            bottom_right_x = center_x + x_shift + (scaled_patch_width // 2)
            bottom_right_y = center_y + y_shift + (scaled_patch_height // 2)
            # print(patch_top_left_x, patch_top_left_y, bottom_right_x, bottom_right_y)

            shift_list.append((x_shift, y_shift))
            patch_location_list.append(
                (patch_top_left_x, patch_top_left_y, bottom_right_x, bottom_right_y))
            padded_patch_list.append(padded_patch_i)

        image_mask = torch.stack(image_mask_list, dim=0)
        padded_patch = torch.stack(padded_patch_list, dim=0)
        inverted_mask = (
            torch.from_numpy(np.ones(shape=image_mask.shape, dtype=np.float32)).to(
                self.estimator.device) - image_mask
        )

        # print("-", padded_patch.shape, image_mask.shape)
        # import matplotlib.pyplot as plt
        # plt.imshow(image_mask[0].cpu().permute(1, 2, 0))
        # plt.show()
        # print(torch.sum(image_mask != 0.).item())
        #######
        # print("--->", images.shape)
        # print("--->", inverted_mask.shape)
        # print("--->", padded_patch.shape)
        # print("--->", image_mask.shape)
        # print("-->", (images * inverted_mask).shape,
        #      (padded_patch * image_mask).shape)
        patched_images = images * inverted_mask + padded_patch * image_mask

        if not self.estimator.channels_first:
            patched_images = torch.permute(patched_images, (0, 2, 3, 1))

        # print("shift", x_shift, y_shift)

        return patched_images, patch_location_list, shift_list

    def _random_overlay_get_patch_location_split(
        self,
        images: "torch.Tensor",
        patch: "torch.Tensor",
        scale: float | None = None,
        mask: "torch.Tensor" | None = None,
        split_keep_both: bool = True,
        split: bool = False,
        half_to_keep: str = "right",
    ) -> "torch.Tensor":
        """
        Split the patch into two parts, then apply them and also return their location.
        returns patched_images, combined_patch_locations, shift_list
        """
        if not self.split and not split:
            return self._random_overlay_get_patch_location(images, patch, scale, mask)

        # Split the patch into left and right halves
        # print(patch.shape)
        width_split = patch.shape[self.i_w] // 2

        left_half = patch[:, :, :width_split]
        right_half = patch[:, :, width_split:]
        # print(left_half.shape, right_half.shape)

        # import matplotlib.pyplot as plt
        # plt.imshow(images[0].cpu().detach().permute(1, 2, 0)/255)
        # plt.show()
        if not split_keep_both and half_to_keep == "right":
            left_half = right_half
        patched_images, patch_location_list_left, shift_list = self._random_overlay_get_patch_location(
            images, left_half, scale, mask, leave_margin_right=True, gap_size=self.gap_size)
        # print("left", patch_location_list_left)

        # import matplotlib.pyplot as plt
        # plt.imshow(patched_images[0].cpu().detach().permute(1, 2, 0)/255)
        # plt.show()
        if split_keep_both:
            patched_images, patch_location_list_right, shift_list = self._random_overlay_get_patch_location(
                patched_images, right_half, scale, mask, prev_patches=patch_location_list_left, prev_shift_list=shift_list, gap_size=self.gap_size)
            # print("right", patch_location_list_right)

            combined_patch_locations = [[left, right] for left, right in zip(
                patch_location_list_left, patch_location_list_right)]  # This returns two boxes, one for each patch

            combined_patch_locations = [(left[0], right[1], right[2], right[3]) for left, right in zip(
                patch_location_list_left, patch_location_list_right)]  # This returns one box, over both patches and the gap between them
        else:
            combined_patch_locations = patch_location_list_left

        import matplotlib.pyplot as plt
        # plt.imshow(patched_images[0].cpu().detach().permute(1, 2, 0)/255)
        # plt.show()
        # print("combined", combined_patch_locations)
        # print("=====")
        return patched_images, combined_patch_locations, shift_list

    def _random_overlay(
        self,
        images: "torch.Tensor",
        patch: "torch.Tensor",
        scale: float | None = None,
        mask: "torch.Tensor" | None = None,
    ) -> "torch.Tensor":
        import torch
        import torchvision

        # Ensure channels-first
        if not self.estimator.channels_first:
            images = torch.permute(images, (0, 3, 1, 2))

        nb_samples = images.shape[0]

        image_mask = self._get_circular_patch_mask(nb_samples=nb_samples)
        image_mask = image_mask.float()

        self.image_shape = images.shape[1:]

        smallest_image_edge = np.minimum(
            self.image_shape[self.i_h], self.image_shape[self.i_w])

        image_mask = torchvision.transforms.functional.resize(
            img=image_mask,
            size=(smallest_image_edge, smallest_image_edge),
            interpolation=2,
        )

        pad_h_before = int(
            (self.image_shape[self.i_h] - image_mask.shape[self.i_h_patch + 1]) / 2)
        pad_h_after = int(
            self.image_shape[self.i_h] - pad_h_before - image_mask.shape[self.i_h_patch + 1])

        pad_w_before = int(
            (self.image_shape[self.i_w] - image_mask.shape[self.i_w_patch + 1]) / 2)
        pad_w_after = int(
            self.image_shape[self.i_w] - pad_w_before - image_mask.shape[self.i_w_patch + 1])

        image_mask = torchvision.transforms.functional.pad(
            img=image_mask,
            padding=[pad_w_before, pad_h_before, pad_w_after, pad_h_after],
            fill=0,
            padding_mode="constant",
        )

        if self.nb_dims == 4:
            image_mask = torch.unsqueeze(image_mask, dim=1)
            image_mask = torch.repeat_interleave(
                image_mask, dim=1, repeats=self.input_shape[0])

        image_mask = image_mask.float()

        contrast_alpha = np.random.uniform(
            self.contrast_min, self.contrast_max)
        mid_gray_value = 127
        patch = (patch * contrast_alpha) + \
            ((1 - contrast_alpha) * mid_gray_value)

        patch = patch.float()
        padded_patch = torch.stack([patch] * nb_samples)

        padded_patch = torchvision.transforms.functional.resize(
            img=padded_patch,
            size=(smallest_image_edge, smallest_image_edge),
            interpolation=2,
        )

        padded_patch = torchvision.transforms.functional.pad(
            img=padded_patch,
            padding=[pad_w_before, pad_h_before, pad_w_after, pad_h_after],
            fill=0,
            padding_mode="constant",
        )

        if self.nb_dims == 4:
            padded_patch = torch.unsqueeze(padded_patch, dim=1)
            padded_patch = torch.repeat_interleave(
                padded_patch, dim=1, repeats=self.input_shape[0])

        padded_patch = padded_patch.float()

        image_mask_list = []
        padded_patch_list = []

        for i_sample in range(nb_samples):
            if self.patch_location is None:
                if scale is None:
                    im_scale = np.random.uniform(
                        low=self.scale_min, high=self.scale_max)
                else:
                    im_scale = scale
            else:
                im_scale = self.patch_shape[self.i_h] / smallest_image_edge

            if mask is None:
                if self.patch_location is None:
                    padding_after_scaling_h = (
                        self.image_shape[self.i_h] - im_scale *
                        padded_patch.shape[self.i_h + 1]
                    ) / 2.0
                    padding_after_scaling_w = (
                        self.image_shape[self.i_w] - im_scale *
                        padded_patch.shape[self.i_w + 1]
                    ) / 2.0
                    x_shift = np.random.uniform(-padding_after_scaling_w,
                                                padding_after_scaling_w)
                    y_shift = np.random.uniform(-padding_after_scaling_h,
                                                padding_after_scaling_h)
                else:
                    padding_h = int(math.floor(
                        self.image_shape[self.i_h] - self.patch_shape[self.i_h]) / 2.0)
                    padding_w = int(math.floor(
                        self.image_shape[self.i_w] - self.patch_shape[self.i_w]) / 2.0)
                    x_shift = -padding_w + self.patch_location[0]
                    y_shift = -padding_h + self.patch_location[1]
            else:
                mask_2d = mask[i_sample, :, :]

                edge_x_0 = int(
                    im_scale * padded_patch.shape[self.i_w + 1]) // 2
                edge_x_1 = int(
                    im_scale * padded_patch.shape[self.i_w + 1]) - edge_x_0
                edge_y_0 = int(
                    im_scale * padded_patch.shape[self.i_h + 1]) // 2
                edge_y_1 = int(
                    im_scale * padded_patch.shape[self.i_h + 1]) - edge_y_0

                mask_2d[0:edge_x_0, :] = False
                if edge_x_1 > 0:
                    mask_2d[-edge_x_1:, :] = False
                mask_2d[:, 0:edge_y_0] = False
                if edge_y_1 > 0:
                    mask_2d[:, -edge_y_1:] = False

                num_pos = np.argwhere(mask_2d).shape[0]
                pos_id = np.random.choice(num_pos, size=1)
                pos = np.argwhere(mask_2d)[pos_id[0]]
                x_shift = pos[1] - self.image_shape[self.i_w] // 2
                y_shift = pos[0] - self.image_shape[self.i_h] // 2

            phi_rotate = float(
                np.random.uniform(-self.rotation_max, self.rotation_max))

            image_mask_i = image_mask[i_sample]

            height = padded_patch.shape[self.i_h + 1]
            width = padded_patch.shape[self.i_w + 1]

            half_height = height // 2
            half_width = width // 2
            topleft = [
                int(torch.randint(0, int(self.distortion_scale_max *
                    half_width) + 1, size=(1,)).item()),
                int(torch.randint(0, int(self.distortion_scale_max *
                    half_height) + 1, size=(1,)).item()),
            ]
            topright = [
                int(torch.randint(width - int(self.distortion_scale_max *
                    half_width) - 1, width, size=(1,)).item()),
                int(torch.randint(0, int(self.distortion_scale_max *
                    half_height) + 1, size=(1,)).item()),
            ]
            botright = [
                int(torch.randint(width - int(self.distortion_scale_max *
                    half_width) - 1, width, size=(1,)).item()),
                int(torch.randint(height - int(self.distortion_scale_max *
                    half_height) - 1, height, size=(1,)).item()),
            ]
            botleft = [
                int(torch.randint(0, int(self.distortion_scale_max *
                    half_width) + 1, size=(1,)).item()),
                int(torch.randint(height - int(self.distortion_scale_max *
                    half_height) - 1, height, size=(1,)).item()),
            ]
            startpoints = [[0, 0], [width - 1, 0],
                           [width - 1, height - 1], [0, height - 1]]
            endpoints = [topleft, topright, botright, botleft]

            image_mask_i = torchvision.transforms.functional.perspective(
                img=image_mask_i, startpoints=startpoints, endpoints=endpoints, interpolation=2, fill=None
            )

            image_mask_i = torchvision.transforms.functional.affine(
                img=image_mask_i,
                angle=phi_rotate,
                translate=[x_shift, y_shift],
                scale=im_scale,
                shear=[0, 0],
                interpolation=interpolation,
                fill=None,
            )

            image_mask_list.append(image_mask_i)

            padded_patch_i = padded_patch[i_sample]

            padded_patch_i = torchvision.transforms.functional.perspective(
                img=padded_patch_i, startpoints=startpoints, endpoints=endpoints, interpolation=2, fill=None
            )

            padded_patch_i = torchvision.transforms.functional.affine(
                img=padded_patch_i,
                angle=phi_rotate,
                translate=[x_shift, y_shift],
                scale=im_scale,
                shear=[0, 0],
                interpolation=interpolation,
                fill=None,
            )

            padded_patch_list.append(padded_patch_i)

        image_mask = torch.stack(image_mask_list, dim=0)
        padded_patch = torch.stack(padded_patch_list, dim=0)
        inverted_mask = (
            torch.from_numpy(np.ones(shape=image_mask.shape, dtype=np.float32)).to(
                self.estimator.device) - image_mask
        )

        patched_images = images * inverted_mask + padded_patch * image_mask

        if not self.estimator.channels_first:
            patched_images = torch.permute(patched_images, (0, 2, 3, 1))

        return patched_images

    def generate(  # type: ignore
        self, x: np.ndarray | list, y: np.ndarray | None = None, transform: torchvision.transforms | None = None, **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate an adversarial patch and return the patch and its mask in arrays.

        :param x: A numpy array with the original input images of shape NCHW or input videos of shape NFCHW.
                  Alternatively for large datasets, a list of filepaths.
        :param y: Untargeted attack: An array with the original true labels. Targeted Attack: The target labels (boxes and classes)
        :param mask: A boolean array of shape equal to the shape of a single samples (1, H, W) or the shape of `x`
                     (N, H, W) without their channel dimensions. Any features for which the mask is True can be the
                     center location of the patch during sampling.
        :type mask: `np.ndarray`
        :transform: If x is a list of filepaths, this is the transformation to apply to the loaded images.
        :return: An array with adversarial patch and an array of the patch mask.
        """
        import torch
        print("USING DEVICE:", self.estimator.device)  # TODO REMOVE

        shuffle = kwargs.get("shuffle", True)
        mask = kwargs.get("mask")
        if mask is not None:
            mask = mask.copy()
        mask = self._check_mask(mask=mask, x=x)

        if self.patch_location is not None and mask is not None:
            raise ValueError(
                "Masks can only be used if the `patch_location` is `None`.")

        if y is None:  # pragma: no cover
            logger.info(
                "Setting labels to estimator predictions and running untargeted attack because `y=None`.")
            y = to_categorical(np.argmax(self.estimator.predict(
                x=x), axis=1), nb_classes=self.estimator.nb_classes)

        if hasattr(self.estimator, "nb_classes"):
            y = check_and_transform_label_format(
                labels=y, nb_classes=self.estimator.nb_classes)

            # check if logits or probabilities
            y_pred = self.estimator.predict(x=x[[0]])

            if is_probability(y_pred):
                self.use_logits = False
            else:
                self.use_logits = True

        if isinstance(y, np.ndarray):
            x_tensor = torch.Tensor(x)
            y_tensor = torch.Tensor(y)

            if mask is None:
                dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
                data_loader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=self.batch_size,
                    shuffle=shuffle,
                    drop_last=False,
                )
            else:
                mask_tensor = torch.Tensor(mask)
                dataset = torch.utils.data.TensorDataset(
                    x_tensor, y_tensor, mask_tensor)
                data_loader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=self.batch_size,
                    shuffle=shuffle,
                    drop_last=False,
                )
        else:

            from PIL import Image

            # This class redefines Dataset to support a dataset consisting of filepaths instead of raw images
            # this means we do not need to store the whole dataset in memory but load the images when needed
            class FilepathDataset(torch.utils.data.Dataset):
                def __init__(self, image_paths, y, transform=None):
                    self.image_paths = image_paths  # list of paths to images
                    self.y = y  # corresponding labels
                    self.transform = transform

                def __len__(self):
                    return len(self.image_paths)

                def __getitem__(self, idx):
                    image = Image.open(self.image_paths[idx])

                    if self.transform:
                        image = self.transform(image)
                    image = torch.from_numpy(np.array(image)*255)

                    target = {}
                    target["boxes"] = torch.from_numpy(self.y[idx]["boxes"])
                    target["labels"] = torch.from_numpy(self.y[idx]["labels"])
                    target["scores"] = torch.from_numpy(self.y[idx]["scores"])

                    return image, target

            class ObjectDetectionDataset(torch.utils.data.Dataset):
                """
                Object detection dataset in PyTorch.
                """

                def __init__(self, x, y):
                    self.x = x
                    self.y = y

                def __len__(self):
                    return self.x.shape[0]

                def __getitem__(self, idx):
                    img = torch.from_numpy(self.x[idx])

                    target = {}
                    target["boxes"] = torch.from_numpy(self.y[idx]["boxes"])
                    target["labels"] = torch.from_numpy(self.y[idx]["labels"])
                    target["scores"] = torch.from_numpy(self.y[idx]["scores"])

                    return img, target

            class ObjectDetectionDatasetMask(torch.utils.data.Dataset):
                """
                Object detection dataset in PyTorch.
                """

                def __init__(self, x, y, mask):
                    self.x = x
                    self.y = y
                    self.mask = mask

                def __len__(self):
                    return self.x.shape[0]

                def __getitem__(self, idx):
                    img = torch.from_numpy(self.x[idx])

                    target = {}
                    target["boxes"] = torch.from_numpy(self.y[idx]["boxes"])
                    target["labels"] = torch.from_numpy(self.y[idx]["labels"])
                    target["scores"] = torch.from_numpy(self.y[idx]["scores"])
                    mask_i = torch.from_numpy(self.mask[idx])

                    return img, target, mask_i

            dataset_object_detection: ObjectDetectionDataset | ObjectDetectionDatasetMask | FilepathDataset
            if mask is None:
                if type(x) is np.ndarray:  # Images are provided as a np.array
                    dataset_object_detection = ObjectDetectionDataset(x, y)
                else:  # Images are provided as a list of filepaths
                    dataset_object_detection = FilepathDataset(x, y, transform)

            else:
                dataset_object_detection = ObjectDetectionDatasetMask(
                    x, y, mask)

            data_loader = torch.utils.data.DataLoader(
                dataset=dataset_object_detection,
                batch_size=self.batch_size,
                shuffle=shuffle,
                drop_last=False,
            )

        training_loss = []
        i_step = 0
        # for i_iter in trange(self.max_epochs, desc="Adversarial Patch PyTorch - Epochs", disable=not self.verbose):
        for i_iter in range(self.max_epochs):
            if self.max_steps and i_step >= self.max_steps:
                break
            if mask is None:
                loss_epoch = []
                prev_loss = sum(
                    training_loss[-1]) if training_loss else 'undefined'
                prev_loss = prev_loss * \
                    (-1) if (self._optimizer_string == "pgd") else prev_loss

                for images, target in tqdm(data_loader, desc=f"Training Steps in Epoch {i_iter}/{self.max_epochs}. Previous Loss: {prev_loss}"):
                    # for images, target in torchtnt.utils.tqdm.create_progress_bar(data_loader, desc=f"Training Steps max {self.max_epochs} Epochs", num_epochs_completed=i_iter):
                    if self.max_steps and i_step >= self.max_steps:
                        break
                    images = images.to(self.estimator.device)
                    if isinstance(target, torch.Tensor):
                        target = target.to(self.estimator.device)
                    else:
                        targets = []
                        for idx in range(target["boxes"].shape[0]):
                            targets.append(
                                {
                                    "boxes": target["boxes"][idx].to(self.estimator.device),
                                    "labels": target["labels"][idx].to(self.estimator.device),
                                    "scores": target["scores"][idx].to(self.estimator.device),
                                }
                            )
                        target = targets
                    loss_train_batch = self._train_step(
                        images=images, target=target, mask=None)  # TODO TRACCK LOSS
                    # print("-----")
                    # print(loss_train_batch.item())
                    loss_epoch.append(loss_train_batch.item())
                    i_step = i_step + 1

            else:
                for images, target, mask_i in data_loader:
                    images = images.to(self.estimator.device)
                    if isinstance(target, torch.Tensor):
                        target = target.to(self.estimator.device)
                    else:
                        targets = []
                        for idx in range(target["boxes"].shape[0]):
                            targets.append(
                                {
                                    "boxes": target["boxes"][idx].to(self.estimator.device),
                                    "labels": target["labels"][idx].to(self.estimator.device),
                                    "scores": target["scores"][idx].to(self.estimator.device),
                                }
                            )
                        target = targets
                    mask_i = mask_i.to(self.estimator.device)
                    _ = self._train_step(
                        images=images, target=target, mask=mask_i)

            training_loss.append(loss_epoch)

            # Write summary
            if self.summary_writer is not None:  # pragma: no cover
                x_patched = (
                    self._random_overlay(
                        images=torch.from_numpy(x).to(self.estimator.device), patch=self._patch, mask=mask
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

                self.summary_writer.update(
                    batch_id=0,
                    global_step=i_iter,
                    grad=None,
                    patch=self._patch,
                    estimator=self.estimator,
                    x=x_patched,
                    y=y,
                    targeted=self.targeted,
                )

        if self.summary_writer is not None:
            self.summary_writer.reset()

        return (
            self._patch.detach().cpu().numpy(),
            self._get_circular_patch_mask(nb_samples=1).cpu().numpy()[0],
            training_loss,
        )

    def _check_mask(self, mask: np.ndarray | None, x: np.ndarray) -> np.ndarray | None:
        if mask is not None and (  # pragma: no cover
            (mask.dtype != bool)
            or not (mask.shape[0] == 1 or mask.shape[0] == x.shape[0])
            or not (mask.shape[1] == x.shape[self.i_h + 1] and mask.shape[2] == x.shape[self.i_w + 1])
        ):
            raise ValueError(
                "The shape of `mask` has to be equal to the shape of a single samples (1, H, W) or the"
                "shape of `x` (N, H, W) without their channel dimensions."
            )

        if mask is not None and mask.shape[0] == 1:
            mask = np.repeat(mask, repeats=x.shape[0], axis=0)

        return mask

    def apply_patch(
        self,
        x: np.ndarray,
        scale: float,
        patch_external: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        split: bool = False,
        split_keep_both: bool = True,
        half_to_keep: str = "right",
        return_patch_outlines: bool = False,
    ) -> np.ndarray:
        """
        A function to apply the learned adversarial patch to images or videos.

        :param x: Instances to apply randomly transformed patch.
        :param scale: Scale of the applied patch in relation to the estimator input shape.
        :param patch_external: External patch to apply to images `x`.
        :param mask: A boolean array of shape equal to the shape of a single samples (1, H, W) or the shape of `x`
                     (N, H, W) without their channel dimensions. Any features for which the mask is True can be the
                     center location of the patch during sampling.
        :return: The patched samples.
        """
        import torch

        if mask is not None:
            mask = mask.copy()
        mask = self._check_mask(mask=mask, x=x)
        x_tensor = torch.Tensor(x).to(self.estimator.device)
        if mask is not None:
            mask_tensor = torch.Tensor(mask).to(self.estimator.device)
        else:
            mask_tensor = None
        if isinstance(patch_external, np.ndarray):
            patch_tensor = torch.Tensor(
                patch_external).to(self.estimator.device)
        else:
            patch_tensor = self._patch

        if split:
            patched_images, locations, shifts = self._random_overlay_get_patch_location_split(
                images=x_tensor, patch=patch_tensor, scale=scale, mask=mask_tensor, split=split, split_keep_both=split_keep_both, half_to_keep=half_to_keep)
            if return_patch_outlines:
                return patched_images.detach().cpu().numpy(), locations
            return patched_images.detach().cpu().numpy()

        return (
            self._random_overlay(
                images=x_tensor, patch=patch_tensor, scale=scale, mask=mask_tensor)
            .detach()
            .cpu()
            .numpy()
        )

    def reset_patch(self, initial_patch_value: float | np.ndarray | None = None) -> None:
        """
        Reset the adversarial patch.

        :param initial_patch_value: Patch value to use for resetting the patch.
        """
        import torch

        if initial_patch_value is None:
            self._patch.data = torch.Tensor(self._initial_value).double()
        elif isinstance(initial_patch_value, float):
            initial_value = np.ones(self.patch_shape) * initial_patch_value
            self._patch.data = torch.Tensor(initial_value).double()
        elif self._patch.shape == initial_patch_value.shape:
            self._patch.data = torch.Tensor(initial_patch_value).double()
        else:
            raise ValueError("Unexpected value for initial_patch_value.")

    @staticmethod
    def insert_transformed_patch(x: np.ndarray, patch: np.ndarray, image_coords: np.ndarray):
        """
        Insert patch to image based on given or selected coordinates.

        :param x: The image to insert the patch.
        :param patch: The patch to be transformed and inserted.
        :param image_coords: The coordinates of the 4 corners of the transformed, inserted patch of shape
            [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] in pixel units going in clockwise direction, starting with upper
            left corner.
        :return: The input `x` with the patch inserted.
        """
        return insert_transformed_patch(x, patch, image_coords)

    def _check_params(self) -> None:
        super()._check_params()

        if not isinstance(self.distortion_scale_max, (float, int)) or 1.0 <= self.distortion_scale_max < 0.0:
            raise ValueError(
                "The maximum distortion scale has to be greater than or equal 0.0 or smaller than 1.0.")

        if self.patch_location is not None and not (
            isinstance(self.patch_location, tuple)
            and len(self.patch_location) == 2
            and isinstance(self.patch_location[0], int)
            and self.patch_location[0] >= 0
            and isinstance(self.patch_location[1], int)
            and self.patch_location[1] >= 0
        ):
            raise ValueError(
                "The patch location has to be either `None` or a tuple of two integers greater than or equal 0."
            )

        if self.patch_type not in ["circle", "square"]:
            raise ValueError(
                "The patch type has to be either `circle` or `square`.")
