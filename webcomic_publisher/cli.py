import cv2
import glob
import json
import math
import numpy as np
import os
from pathlib import Path
import typer
from typing import Optional
from typing_extensions import Annotated


app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


espace = 30
bordersize = 8


def readConfigFile(filepath):
    f = open(filepath)
    config = json.load(f)
    f.close()
    return config


def scaleImages(images):
    maxhauteur = 0
    for image in images:
        hauteur, largeur = image.shape[:2]
        if hauteur > maxhauteur:
            maxhauteur = hauteur
    scaledimages = []
    for image in images:
        hauteur, largeur = image.shape[:2]
        facteur = maxhauteur / hauteur
        res = cv2.resize(image, None, fx=facteur, fy=facteur, interpolation=cv2.INTER_CUBIC)
        scaledimages.append(res)
    return scaledimages


def getMaxLargeurForLayout(images, layout):
    maxlargeur = 0
    indexDebut = 0
    nbsImages = layout.split("-")
    for nbImages in nbsImages:
        indexFin = indexDebut + int(nbImages)
        sliceImages = images[indexDebut:indexFin]
        largeurEspaces = espace * (len(sliceImages) - 1)
        largeur = 0
        for image in sliceImages:
            h, l = image.shape[:2]
            largeur += l
        largeur += largeurEspaces

        if largeur > maxlargeur:
            maxlargeur = largeur

        indexDebut = indexFin
    return maxlargeur


@app.command()
def generate(
    directory: Annotated[Optional[Path], typer.Option(file_okay=False, dir_okay=True, help="Source folder path")],
    output: Annotated[
        Optional[Path], typer.Option(file_okay=False, dir_okay=True, help="Output folder path")
    ] = "export",
    formats: Annotated[Optional[str], typer.Option(help='Output formats inside double quotes (e.g.: "2-2 4")')] = None,
) -> None:
    config = readConfigFile(f"{directory}/config.json")
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
        scaledImages = scaleImages(images)

        largeurImage = getMaxLargeurForLayout(scaledImages, layout)

        hauteurImage = 0
        indexDebut = 0
        nbsImages = layout.split("-")
        rescaledImages = []

        # On fait la matrice d'images rescaledImages
        for nbImages in nbsImages:
            indexFin = indexDebut + int(nbImages)
            sliceImages = scaledImages[indexDebut:indexFin]
            largeurEspaces = espace * (len(sliceImages) - 1)
            largeur = 0
            for image in sliceImages:
                h, l = image.shape[:2]
                largeur += l
            facteur = (largeurImage - largeurEspaces) / largeur

            rescaledRow = []
            for image in sliceImages:
                res = cv2.resize(image, None, fx=facteur, fy=facteur, interpolation=cv2.INTER_CUBIC)
                rescaledRow.append(res)

            rescaledImages.append(rescaledRow)
            hauteurImage += sliceImages[0].shape[0] * facteur
            indexDebut = indexFin
        hauteurImage += espace * (len(rescaledImages) - 1)

        largeurImage = math.ceil(largeurImage)
        hauteurImage = math.ceil(hauteurImage)
        # On construit le png final
        outputImage = np.zeros((hauteurImage, largeurImage, 4), np.uint8)
        x = y = 0
        for row in rescaledImages:
            h = 0
            for image in row:
                h, l = image.shape[:2]
                croppedimage = image[bordersize : h - bordersize, bordersize : l - bordersize]
                borderimage = cv2.copyMakeBorder(
                    croppedimage,
                    top=bordersize,
                    bottom=bordersize,
                    left=bordersize,
                    right=bordersize,
                    borderType=cv2.BORDER_CONSTANT,
                    value=[0, 0, 0],
                )
                borderimage = np.concatenate((borderimage, np.full((h, l, 1), 255)), axis=2)
                outputImage[y : y + borderimage.shape[0], x : x + borderimage.shape[1]] = borderimage
                x += l + espace
            x = 0
            y += h + espace
        copyright = cv2.imread("copyright.png")
        copyright = cv2.resize(copyright, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        copyrighth, copyrightl = copyright.shape[:2]
        copyright = np.concatenate((copyright, np.full((copyrighth, copyrightl, 1), 255)), axis=2)
        outputImage[
            hauteurImage - copyrighth - bordersize : hauteurImage - bordersize,
            largeurImage - copyrightl - bordersize : largeurImage - bordersize,
        ] = copyright

        cv2.imwrite(
            f"{output}/{layout}.png",
            outputImage,
            [cv2.IMWRITE_PNG_COMPRESSION, 9],
        )

        for resizeWidth in format["resizeWidths"]:
            ratio = resizeWidth / largeurImage
            resizedOutputImage = cv2.resize(
                outputImage,
                None,
                fx=ratio,
                fy=ratio,
                interpolation=cv2.INTER_AREA,
            )
            cv2.imwrite(
                f"{output}/{layout}_{str(resizeWidth)}.png",
                resizedOutputImage,
                [cv2.IMWRITE_PNG_COMPRESSION, 9],
            )
