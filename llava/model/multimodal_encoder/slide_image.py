import math
import torch
from PIL import Image

def slice_image(
    image, max_slice_nums=9, scale_resolution=448, patch_size=14, never_split=False
):
    # 原始图片尺寸
    original_size = image.size
    original_width, original_height = original_size
    # 宽高比的对数值，用于后续比较和优化
    log_ratio = math.log(original_width / original_height)
    # 比例
    ratio = original_width * original_height / (scale_resolution * scale_resolution)
    multiple = min(math.ceil(ratio), max_slice_nums)

    source_image = None
    best_grid = None
    patches = []

    if multiple <= 1 or never_split:
        # dont need to slice, upsample
        best_size = find_best_resize(
            original_size, scale_resolution, patch_size, allow_upscale=True
        )
        source_image = image.resize(best_size, Image.Resampling.BICUBIC)
    else:
        candidate_split_grids_nums = []
        for i in [multiple - 1, multiple, multiple + 1]:
            if i == 1 or i > max_slice_nums:
                continue
            candidate_split_grids_nums.append(i)

        # source image, down-sampling and ensure divided by patch_size
        best_resize = find_best_resize(
            original_size, scale_resolution, patch_size)
        source_image = image.copy().resize(best_resize, Image.Resampling.BICUBIC)
        candidate_grids = []

        # find best grid
        for split_grids_nums in candidate_split_grids_nums:
            m = 1
            while m <= split_grids_nums:
                if split_grids_nums % m == 0:
                    candidate_grids.append([m, split_grids_nums // m])
                m += 1

        best_grid = [1, 1]
        min_error = float("inf")
        for grid in candidate_grids:
            error = abs(log_ratio - math.log(grid[0] / grid[1]))
            if error < min_error:
                best_grid = grid
                min_error = error

        refine_size = get_refine_size(
            original_size, best_grid, scale_resolution, patch_size, allow_upscale=True
        )

        refine_image = image.resize(refine_size, Image.Resampling.BICUBIC)
        patches = split_to_patches(refine_image, best_grid)

    return source_image, patches, best_grid


def ensure_divide(length, patch_size):
    return max(round(length / patch_size) * patch_size, patch_size)


def find_best_resize(original_size, scale_resolution, patch_size, allow_upscale=False):
    width, height = original_size
    if (width * height > scale_resolution * scale_resolution) or allow_upscale:
        r = width / height
        height = int(scale_resolution / math.sqrt(r))
        width = int(height * r)
    best_width = ensure_divide(width, patch_size)
    best_height = ensure_divide(height, patch_size)
    return best_width, best_height


def get_refine_size(
    original_size, grid, scale_resolution, patch_size, allow_upscale=False
):
    width, height = original_size
    grid_x, grid_y = grid

    refine_width = ensure_divide(width, grid_x)
    refine_height = ensure_divide(height, grid_y)

    grid_width = refine_width / grid_x
    grid_height = refine_height / grid_y

    best_grid_size = find_best_resize(
        (grid_width, grid_height),
        scale_resolution,
        patch_size,
        allow_upscale=allow_upscale,
    )

    refine_size = (best_grid_size[0] * grid_x, best_grid_size[1] * grid_y)

    return refine_size


def split_to_patches(image, grid):
    patches = []
    width, height = image.size
    grid_x = int(width / grid[0])
    grid_y = int(height / grid[1])

    for i in range(0, height, grid_y):
        images = []
        for j in range(0, width, grid_x):
            box = (j, i, j + grid_x, i + grid_y)
            patch = image.crop(box)
            images.append(patch)
        patches.append(images)

    return patches


def get_grid_placeholder(tokenizer, grid, query_num, new_schema=False):
    if new_schema:
        image_placeholder = (
            tokenizer.slice_start + tokenizer.unk_token * query_num + tokenizer.slice_end
        )
    else:
        image_placeholder = (
            tokenizer.im_start + tokenizer.unk_token * query_num + tokenizer.im_end
        )

    cols = grid[0]
    rows = grid[1]
    slices = []
    for i in range(rows):
        lines = []
        for j in range(cols):
            lines.append(image_placeholder)
        slices.append("".join(lines))
    if new_schema:
        slice_placeholder = '\n'.join(slices)
    else:
        slice_placeholder = tokenizer.slice_start + \
        "\n".join(slices) + tokenizer.slice_end
    return slice_placeholder


def reshape_by_patch(image_tensor, patch_size):
    """
    :param image_tensor: shape [3, H, W]
    :param patch_size:
    :return: [3, patch_size, HW/patch_size]
    """
    patches = torch.nn.functional.unfold(
        image_tensor, (patch_size, patch_size), stride=(patch_size, patch_size)
    )

    patches = patches.reshape(image_tensor.size(0), patch_size, patch_size, -1)
    patches = patches.permute(0, 1, 3, 2).reshape(
        image_tensor.size(0), patch_size, -1)
    return patches


if __name__ == "__main__":
    image = Image.open("/data/LLaVA-NeXT/docs/ov_chat_images/example1_tree.png")
    source_image, patches, best_grid = slice_image(
        image, max_slice_nums=9, scale_resolution=224, patch_size=14, never_split=False
    )
    print(image.size, source_image.size)
    print(source_image)
    print(patches)
    print(best_grid)
