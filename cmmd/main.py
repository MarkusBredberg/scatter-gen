# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The main entry point for the CMMD calculation."""

from absl import app
from absl import flags
import cmmd.distance as distance
import cmmd.embedding as embedding
import cmmd.io_util as io_util
import numpy as np
import torch


_BATCH_SIZE = flags.DEFINE_integer("batch_size", 32, "Batch size for embedding generation.")
_MAX_COUNT = flags.DEFINE_integer("max_count", -1, "Maximum number of images to read from each directory.")
_REF_EMBED_FILE = flags.DEFINE_string(
    "ref_embed_file", None, "Path to the pre-computed embedding file for the reference images."
)


def compute_cmmd_from_tensors(ref_images, eval_images, batch_size=32, max_count=-1):
    """
    Calculates the CMMD distance between reference and evaluation image sets using tensors of images.

    Args:
      ref_images: Tensor of reference images (e.g., torch.Tensor of shape [N, C, H, W]).
      eval_images: Tensor of evaluation images (e.g., torch.Tensor of shape [M, C, H, W]).
      batch_size: Batch size used in the CLIP embedding calculation.
      max_count: Maximum number of images to use from each set. A non-positive value uses all available images.

    Returns:
      The CMMD value between the image sets.
    """
    # Ensure ref_images and eval_images are tensors
    if not isinstance(ref_images, torch.Tensor) or not isinstance(eval_images, torch.Tensor):
        raise ValueError("ref_images and eval_images must be torch.Tensor")

    # Limit the number of images to max_count if specified
    if max_count > 0:
        ref_images = ref_images[:max_count]
        eval_images = eval_images[:max_count]

    # Initialize the CLIP embedding model
    embedding_model = embedding.ClipEmbeddingModel()

    # Compute embeddings for the reference and evaluation images in batches
    def compute_embeddings(images):
        embeddings = []
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_embeddings = embedding_model(batch_images)
            embeddings.append(batch_embeddings)
        return torch.cat(embeddings, dim=0).cpu().numpy()

    # Get embeddings for both sets of images
    ref_embs = compute_embeddings(ref_images).astype("float32")
    eval_embs = compute_embeddings(eval_images).astype("float32")

    # Calculate CMMD (MMD) between the two sets of embeddings
    cmmd_value = distance.mmd(ref_embs, eval_embs)

    return cmmd_value.numpy()


def compute_cmmd(ref_dir, eval_dir, ref_embed_file=None, batch_size=32, max_count=-1):
    """Calculates the CMMD distance between reference and eval image sets.

    Args:
      ref_dir: Path to the directory containing reference images.
      eval_dir: Path to the directory containing images to be evaluated.
      ref_embed_file: Path to the pre-computed embedding file for the reference images.
      batch_size: Batch size used in the CLIP embedding calculation.
      max_count: Maximum number of images to use from each directory. A
        non-positive value reads all images available except for the images
        dropped due to batching.

    Returns:
      The CMMD value between the image sets.
    """
    if ref_dir and ref_embed_file:
        raise ValueError("`ref_dir` and `ref_embed_file` both cannot be set at the same time.")
    embedding_model = embedding.ClipEmbeddingModel()
    if ref_embed_file is not None:
        ref_embs = np.load(ref_embed_file).astype("float32")
    else:
        ref_embs = io_util.compute_embeddings_for_dir(ref_dir, embedding_model, batch_size, max_count).astype(
            "float32"
        )
    eval_embs = io_util.compute_embeddings_for_dir(eval_dir, embedding_model, batch_size, max_count).astype("float32")
    val = distance.mmd(ref_embs, eval_embs)
    return val.numpy()


def main(argv):
    if len(argv) != 3:
        raise app.UsageError("Too few/too many command-line arguments.")
    _, dir1, dir2 = argv
    print(
        "The CMMD value is: "
        f" {compute_cmmd(dir1, dir2, _REF_EMBED_FILE.value, _BATCH_SIZE.value, _MAX_COUNT.value):.3f}"
    )


if __name__ == "__main__":
    app.run(main)
