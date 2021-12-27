import json
import math
from pathlib import Path

import exiftool
import mercantile
import numpy as np
from PIL import Image
from tqdm import tqdm

# Параметры на глазок
scale = 1 / 4 * 1 / 3
merc_pos_scale = 1 / 3 * 1.2 / 3000000

# Классы
classes = ["net", "metall", "wood", "plastic"]


def rotate(x, y, phi, center):
    return (
        (x - center[0]) * math.cos(phi) - (y - center[1]) * math.sin(phi) + center[0],
        (x - center[0]) * math.sin(phi) + (y - center[1]) * math.cos(phi) + center[1],
    )


# Получает информацию о преобразованиях изображений и точке отчёта
def get_transforms_and_zero_point(images):
    transformations = []
    zero_point = None
    for image in tqdm(images):
        img = Image.open(image).convert("RGBA")
        (width, height) = (img.width // 2, img.height // 2)
        img = img.resize((width, height))
        with exiftool.ExifTool() as et:
            metadata = et.get_metadata_batch([str(image)])
            # print(metadata)
            yaw = metadata[0]["MakerNotes:CameraYaw"]
            lat = metadata[0]["EXIF:GPSLatitude"]
            lng = metadata[0]["EXIF:GPSLongitude"]
            alt = metadata[0]["EXIF:GPSAltitude"]
            phi = (-yaw) * math.pi / 180

            item = mercantile.tile(lat=lat, lng=lng, zoom=50)

            x = item.x
            y = item.y
            if zero_point is None:
                zero_point = (x, y)
            center_position = (
                int((x - zero_point[0]) * merc_pos_scale),
                int((y - zero_point[1]) * merc_pos_scale),
            )
            transformations.append(
                {
                    "phi": phi,
                    "yaw": -yaw,
                    "image": image,
                    "x": x,
                    "y": y,
                    "alt": alt,
                    "center_position": center_position,
                }
            )
    return transformations, zero_point


# Преобразоване XY в lan, lat
def orto_to_lanLat(x, y, min_pos_x, min_pos_y, max_img_size, zero_point):
    pos = (
        np.array([x, y])
        + [min_pos_x, min_pos_y]
        - [max_img_size[0] * 1.5, max_img_size[1] * 1.5]
    ) / merc_pos_scale + [zero_point[0], zero_point[1]]
    pos = pos.astype(np.int64)
    pos = mercantile.ul(pos[0], pos[1], 50)
    return pos


def create_orto(images, out_path):
    out_path = Path(out_path)

    transformations, zero_point = get_transforms_and_zero_point(images)

    min_pos_x = np.min([item["center_position"][0] for item in transformations])
    min_pos_y = np.min([item["center_position"][1] for item in transformations])
    max_pos_x = np.max([item["center_position"][0] for item in transformations])
    max_pos_y = np.max([item["center_position"][1] for item in transformations])

    img = Image.open(images[0])
    width, height = (img.width // 2, img.height // 2)

    max_img_size = max(width * scale, height * scale)
    max_img_size = (max_img_size, max_img_size)

    sheet_size = (
        int(max_pos_x - min_pos_x + 3 * max_img_size[0]),
        int(max_pos_y - min_pos_y + 3 * max_img_size[1]),
    )

    image_sheet = Image.new("RGBA", sheet_size)
    image_all_masks = Image.new("RGBA", sheet_size)
    image_masks = {
        "net": Image.new("RGBA", sheet_size),
        "metall": Image.new("RGBA", sheet_size),
        "wood": Image.new("RGBA", sheet_size),
        "plastic": Image.new("RGBA", sheet_size),
    }

    for image, tr in tqdm(list(zip(images, transformations))):
        img = Image.open(image).convert("RGBA")
        img_id = "_".join(image.name.split("_")[:-1])

        mask = None
        masks = {}
        for c in classes:
            c_mask = image.parent / (img_id + "_" + c + ".png")
            if not c_mask.is_file():
                continue
            c_mask = Image.open(c_mask)
            masks[c] = c_mask
            if mask is None:
                mask = c_mask
            else:
                mask.paste(c_mask)

        img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))
        mask = mask.resize((int(mask.size[0] * scale), int(mask.size[1] * scale)))
        for c in classes:
            if c not in masks:
                continue
            c_mask = masks[c]
            masks[c] = c_mask.resize(
                (int(c_mask.size[0] * scale), int(c_mask.size[1] * scale))
            )

        center_position = (
            int((tr["x"] - zero_point[0]) * merc_pos_scale + max_img_size[0] * 1.5)
            - min_pos_x,
            int((tr["y"] - zero_point[1]) * merc_pos_scale + max_img_size[1] * 1.5)
            - min_pos_y,
        )

        img = img.rotate(tr["yaw"], expand=1)
        mask = mask.rotate(tr["yaw"], expand=1)
        for c in classes:
            if c not in masks:
                continue
            c_mask = masks[c]
            masks[c] = c_mask.rotate(tr["yaw"], expand=1)

        tl_position = (
            center_position[0] - img.size[0] // 2,
            center_position[1] - img.size[1] // 2,
        )

        image_all_masks.paste(mask, tl_position, mask)
        image_sheet.paste(img, tl_position, img)
        for c in classes:
            if c not in masks:
                continue
            image_masks[c].paste(masks[c], tl_position, masks[c])

    image_all_masks.save(out_path / "all_masks.png")
    image_sheet.save(out_path / "orto.png")
    for c in classes:
        image_masks[c].save(out_path / f"{c}_mask.png")

    # Положение ортофотографии на карте (левый верхний и нижний правый)
    tl_pos = orto_to_lanLat(0, 0, min_pos_x, min_pos_y, max_img_size, zero_point)
    br_pos = orto_to_lanLat(2327, 4010, min_pos_x, min_pos_y, max_img_size, zero_point)
    with open(out_path / "positions.json", "w") as stream:
        json.dump(
            {
                "tl_lat": tl_pos.lat,
                "tl_lng": tl_pos.lng,
                "br_lat": br_pos.lat,
                "br_lng": br_pos.lng,
            },
            stream,
        )

    # Heatmap
    masks = np.array(image_all_masks)
    masks
    window_size = (100, 100)
    bin_mask = (masks[:, :].sum(2) > 0).astype(np.float64)
    m = bin_mask.copy()
    mP = np.zeros(
        (
            len(range(0, bin_mask.shape[0], window_size[0])),
            len(range(0, bin_mask.shape[1], window_size[1])),
        )
    )
    for pi, i in enumerate(range(0, bin_mask.shape[0], window_size[0])):
        for pj, j in enumerate(range(0, bin_mask.shape[1], window_size[1])):
            mP[pi, pj] = np.log(
                bin_mask[i : i + window_size[0], j : j + window_size[1]].sum() + 1
            )
            m[i : i + window_size[0], j : j + window_size[1]] = np.log(
                bin_mask[i : i + window_size[0], j : j + window_size[1]].sum() + 1
            )
    m = (m - m.min()) / (m.max() - m.min())
    mP = (mP - mP.min()) / (mP.max() - mP.min())

    heatmap_layer = np.zeros((m.shape[0], m.shape[1], 3))
    heatmap_layer[:, :, 0] = m * 255
    heatmap_layer = heatmap_layer.astype(np.uint8)
    hm = Image.fromarray(heatmap_layer)
    a_channel = Image.new("L", hm.size, 230)
    hm.putalpha(a_channel)
    hm = np.array(hm)
    hm[:, :, 3] = 150 * (np.array(image_sheet).sum(2) != 0)
    hm = Image.fromarray(hm)
    hm.save(str(out_path / "heatmap.png"))

    image_with_hm = Image.new("RGBA", sheet_size)
    image_with_hm.paste(image_sheet)
    for c in classes:
        image_with_hm.paste(image_masks[c], (0, 0),image_masks[c]
    # image_with_hm.paste(image_all_masks, (0, 0), image_all_masks)
    image_with_hm.paste(hm, (0, 0), hm)
    image_with_hm.save(str(out_path / "orto_with_heatmap.png"))


if __name__ == "__main__":
    files = sorted(list(Path("data/raw/охотское_море/example").glob("*.JPG")))
    create_orto(files, "test")
