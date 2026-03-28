import numpy as np
import math

from utils.gan_image_norm import batch_to_preview_uint8, tensor_to_image_uint8


def _as_numpy_batch(x):
    """Keras predict may return a list/tuple for multi-output models; unwrap single output."""
    if isinstance(x, (list, tuple)):
        if len(x) != 1:
            raise ValueError(f"Expected a single output tensor or array, got {len(x)} outputs")
        x = x[0]
    return np.asarray(x)


def _ensure_nhwc_rgb_batch(generated_images: np.ndarray) -> np.ndarray:
    """
    Normalise generator output to (N, H, W, 3) channels-last RGB for tiling.
    Handles optional NCHW (N, 3, H, W) by transposing.
    """
    x = _as_numpy_batch(generated_images)
    if x.ndim == 3:
        x = x[np.newaxis, ...]
    if x.ndim != 4:
        raise ValueError(f"Preview grid needs rank-3 single image or rank-4 batch; got shape {x.shape}")
    # Heuristic: NCHW if channel dim is 3 at axis 1 and last dim is not 3
    if x.shape[1] == 3 and x.shape[-1] != 3:
        x = np.transpose(x, (0, 2, 3, 1))
    if x.shape[-1] != 3:
        raise ValueError(f"Expected RGB (last dim 3); got shape {x.shape}")
    return x


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = img[:, :, 0]
    return image


def combine_rgb_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[1:3]
    image = np.zeros(
        (height * shape[0], width * shape[1], 3),
        dtype=generated_images.dtype,
    )
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0] : (i + 1) * shape[0], j * shape[1] : (j + 1) * shape[1], :] = img[
            :, :, :
        ]
    return image


def save_real_reference_pngs(
    x_train: np.ndarray,
    output_dir: str,
    *,
    count: int = 3,
    from_tanh: bool = True,
    spread_indices: bool = True,
) -> list[str]:
    """
    Save ground-truth training images as PNGs for side-by-side comparison with *_g.png fakes.

    Writes ``real_reference_0..N-1.png`` plus ``real_reference_grid.png`` (tiled).
    Uses the same uint8 mapping as previews (``from_tanh`` when training uses ``--image_range tanh``).

    If ``spread_indices`` and enough samples exist, picks beginning / middle / end of the training
    set so the three images are not identical crops of the same scene.
    """
    from PIL import Image

    n_train = int(x_train.shape[0])
    n = min(count, n_train)
    if n == 0:
        return []

    if spread_indices and n_train >= count:
        idx = [0, n_train // 2, n_train - 1][:n]
    else:
        idx = list(range(n))

    batch = np.stack([x_train[i].astype(np.float32) for i in idx], axis=0)
    u8 = batch_to_preview_uint8(batch, from_tanh=from_tanh)
    saved: list[str] = []
    for i in range(n):
        path = os.path.join(output_dir, f"real_reference_{i}.png")
        Image.fromarray(u8[i], mode="RGB").save(path)
        saved.append(path)
    grid = combine_rgb_images(u8)
    grid_path = os.path.join(output_dir, "real_reference_grid.png")
    Image.fromarray(grid, mode="RGB").save(grid_path)
    saved.append(grid_path)
    return saved


def combine_rgb_preview_grid(generated_images: np.ndarray, *, from_tanh: bool = True) -> np.ndarray:
    """
    Build a tiled grid for PNG saving from generator output in [-1, 1] (tanh) or [0, 1].

    Returns uint8 (H, W, 3). PIL cannot save float32 grids reliably — always uint8 here.
    """
    x = _ensure_nhwc_rgb_batch(generated_images)
    u8 = batch_to_preview_uint8(x, from_tanh=from_tanh)
    return combine_rgb_images(u8)