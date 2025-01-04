import cv2
import glob
import json
import math
import numpy as np
from pathlib import Path
import typer
from typing import Optional
from typing_extensions import Annotated


app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


espace = 30
border_size = 8


def read_config_file(filepath):
    f = open(filepath)
    config = json.load(f)
    f.close()
    return config


def scale_images(images):
    maxhauteur = 0
    for image in images:
        hauteur, largeur = image.shape[:2]
        if hauteur > maxhauteur:
            maxhauteur = hauteur
    scaled_images = []
    for image in images:
        hauteur, largeur = image.shape[:2]
        facteur = maxhauteur / hauteur
        res = cv2.resize(
            image,
            None,
            fx=facteur,
            fy=facteur,
            interpolation=cv2.INTER_CUBIC,
        )
        scaled_images.append(res)
    return scaled_images


def get_max_largeur_for_layout(images, layout):
    maxlargeur = 0
    index_debut = 0
    nbs_images = layout.split("-")
    for nb_images in nbs_images:
        index_fin = index_debut + int(nb_images)
        slice_images = images[index_debut:index_fin]
        largeur_espaces = espace * (len(slice_images) - 1)
        largeur = 0
        for image in slice_images:
            h, l = image.shape[:2]
            largeur += l
        largeur += largeur_espaces

        if largeur > maxlargeur:
            maxlargeur = largeur

        index_debut = index_fin
    return maxlargeur


@app.command()
def generate(
    directory: Annotated[Optional[Path], typer.Option(file_okay=False, dir_okay=True, help="Source folder path")],
    output: Annotated[
        Optional[Path], typer.Option(file_okay=False, dir_okay=True, help="Output folder path")
    ] = "export",
    formats: Annotated[Optional[str], typer.Option(help='Output formats inside double quotes (e.g.: "2-2 4")')] = None,
) -> None:
    config = read_config_file(f"{directory}/config.json")
    if not formats:
        formats = config["formats"]

    output.mkdir(parents=True, exist_ok=True)
    for format in formats:
        layout = format["layout"]
        images = (
            [cv2.imread(file) for file in sorted(glob.glob(f"{directory}/{format['subFolder']}/*.png"))]
            if "subFolder" in format
            else [cv2.imread(file) for file in sorted(glob.glob(f"{directory}/*.png"))]
        )
        scaled_images = scale_images(images)

        largeur_image = get_max_largeur_for_layout(scaled_images, layout)

        hauteur_image = 0
        index_debut = 0
        nbs_images = layout.split("-")
        rescaled_images = []

        # On fait la matrice d'images rescaled_images
        for nb_images in nbs_images:
            index_fin = index_debut + int(nb_images)
            slice_images = scaled_images[index_debut:index_fin]
            largeur_espaces = espace * (len(slice_images) - 1)
            largeur = 0
            for image in slice_images:
                h, l = image.shape[:2]
                largeur += l
            facteur = (largeur_image - largeur_espaces) / largeur

            rescaled_row = []
            for image in slice_images:
                res = cv2.resize(image, None, fx=facteur, fy=facteur, interpolation=cv2.INTER_CUBIC)
                rescaled_row.append(res)

            rescaled_images.append(rescaled_row)
            hauteur_image += slice_images[0].shape[0] * facteur
            index_debut = index_fin
        hauteur_image += espace * (len(rescaled_images) - 1)

        largeur_image = math.ceil(largeur_image)
        hauteur_image = math.ceil(hauteur_image)
        # On construit le png final
        output_image = np.zeros((hauteur_image, largeur_image, 4), np.uint8)
        x = y = 0
        for row in rescaled_images:
            h = 0
            for image in row:
                h, l = image.shape[:2]
                cropped_image = image[border_size : h - border_size, border_size : l - border_size]
                border_image = cv2.copyMakeBorder(
                    cropped_image,
                    top=border_size,
                    bottom=border_size,
                    left=border_size,
                    right=border_size,
                    borderType=cv2.BORDER_CONSTANT,
                    value=[0, 0, 0],
                )
                border_image = np.concatenate((border_image, np.full((h, l, 1), 255)), axis=2)
                output_image[y : y + border_image.shape[0], x : x + border_image.shape[1]] = border_image
                x += l + espace
            x = 0
            y += h + espace
        copyright = cv2.imread("copyright.png")
        copyright = cv2.resize(copyright, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        copyrighth, copyrightl = copyright.shape[:2]
        copyright = np.concatenate((copyright, np.full((copyrighth, copyrightl, 1), 255)), axis=2)
        output_image[
            hauteur_image - copyrighth - border_size : hauteur_image - border_size,
            largeur_image - copyrightl - border_size : largeur_image - border_size,
        ] = copyright

        cv2.imwrite(
            f"{output}/{layout}.png",
            output_image,
            [cv2.IMWRITE_PNG_COMPRESSION, 9],
        )

        for resize_width in format["resize_widths"]:
            ratio = resize_width / largeur_image
            resized_output_image = cv2.resize(
                output_image,
                None,
                fx=ratio,
                fy=ratio,
                interpolation=cv2.INTER_AREA,
            )
            cv2.imwrite(
                f"{output}/{layout}_{str(resize_width)}.png",
                resized_output_image,
                [cv2.IMWRITE_PNG_COMPRESSION, 9],
            )
