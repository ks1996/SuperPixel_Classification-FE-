#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
#  Copyright Kitware Inc.
#
#  Licensed under the Apache License, Version 2.0 ( the "License" );
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################

import argparse
import collections
import json
import os
import re

import numpy
import PIL.Image


def decodeSuperpixelIndex(rgbValue):
    """
    Decode an RGB representation of a superpixel label into its native scalar
    value.
    :param pixelValue: A single pixel, or a 3-channel image.
    :type pixelValue: numpy.ndarray of uint8, with a shape [3] or [n, m, 3]
    """
    return \
        (rgbValue[..., 0].astype(numpy.uint64)) + \
        (rgbValue[..., 1].astype(numpy.uint64) << numpy.uint64(8)) + \
        (rgbValue[..., 2].astype(numpy.uint64) << numpy.uint64(16))


def createDir(dirPath):
    """
    Ensure that a directory exists.
    :param pixelValue: A path to a desired directory.
    :type pixelValue: str
    """
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)


def cropImage(image, mask, keepContext):
    """
    Crop an image, to include only the rectangular bounding box containing a
    given mask.
    :param image: A full image to be cropped.
    :type image: numpy.ndarray of uint8, with a shape [n, m, *]
    :param mask: A boolean mask, specifying which image regions to preserve.
    :type mask: numpy.ndarray of bool, with a shape [n, m]
    :param keepContext: Whether image pixels strictly outside the mask should be
                        kept (instead of set to black).
    :type keepContext: bool
    """
    cropIndices = numpy.ix_(mask.any(1), mask.any(0))
    croppedImage = image[cropIndices]
    croppedMask = mask[cropIndices]

    if not keepContext:
        croppedImage = croppedImage.copy()
        croppedImage[numpy.logical_not(croppedMask)] = 0
    return croppedImage


def extractSuperpixels(imageFilePath, superpixelsFilePath,
                       featuresFilePath, outputDirPath, keepContext):
    image = numpy.array(
        PIL.Image.open(imageFilePath))
    superpixels = decodeSuperpixelIndex(
        numpy.array(
            PIL.Image.open(superpixelsFilePath)))
    with open(featuresFilePath, 'rb') as featuresFileStream:
        features = json.load(featuresFileStream)
    createDir(outputDirPath)

    imageName = os.path.splitext(
        os.path.basename(imageFilePath))[0]

    for featureName, featureValues in features.items():
        outputFeatureDirPath = os.path.join(outputDirPath, featureName)
        createDir(outputFeatureDirPath)

        for superpixelNum, featureValue in enumerate(featureValues):
            superpixelsMask = superpixels == superpixelNum
            maskedImage = cropImage(image, superpixelsMask, keepContext)

            featurePresence = 'present' if featureValue != 0 else 'absent'
            outputFeaturePresenceDirPath = os.path.join(
                outputFeatureDirPath, featurePresence)
            createDir(outputFeaturePresenceDirPath)

            outputFilePath = os.path.join(
                outputFeaturePresenceDirPath,
                '%s_%s_%04d_%s.png' % (
                    imageName, featureName, superpixelNum, featurePresence))
            PIL.Image.fromarray(maskedImage).save(outputFilePath, 'PNG')


def getPart2Paths(dataPath, part2GroundTruthPath):
    part2Paths = collections.defaultdict(dict)

    for fileName in os.listdir(dataPath):
        imageRe = re.match(r'^(ISIC_\d{7}).jpg$', fileName)
        superpixelsRe = re.match(r'^(ISIC_\d{7})_superpixels.png$', fileName)
        if imageRe:
            imageName = imageRe.group(1)
            part2Paths[imageName]['image'] = os.path.join(dataPath, fileName)
        elif superpixelsRe:
            imageName = superpixelsRe.group(1)
            part2Paths[imageName]['superpixels'] = os.path.join(
                dataPath, fileName)
    for fileName in os.listdir(part2GroundTruthPath):
        featuresRe = re.match(r'^(ISIC_\d{7})_features.json$', fileName)
        if featuresRe:
            imageName = featuresRe.group(1)
            part2Paths[imageName]['features'] = os.path.join(
                part2GroundTruthPath, fileName)

    return part2Paths


def main():
    parser = argparse.ArgumentParser(
        description='Extract ISIC-2017 Part 2 features into individual '
                    'superpixel tiles.')
    parser.add_argument(
        'dataDirPath',
        help='The full path to the extracted data directory '
             '(e.g. "ISIC-2017_Training_Data").')
    parser.add_argument(
        'part2GroundTruthDirPath',
        help='The full path to the extracted Part 2 GroundTruth directory '
             '(e.g. "ISIC-2017_Training_Part2_GroundTruth").')
    parser.add_argument(
        'outputDirPath',
        help='The full path to the directory where output will be written.')
    parser.add_argument(
        '--separateImages',
        action='store_true',
        help='Features will be extracted to separate folders per image.')
    parser.add_argument(
        '--keepContext',
        action='store_true',
        help='The image content surrounding each superpixel tile will be kept '
             '(instead of being masked to black).')

    args = parser.parse_args()

    part2Paths = getPart2Paths(
        args.dataDirPath,
        args.part2GroundTruthDirPath)
    for imageName, imagePaths in sorted(part2Paths.items()):
        print('Extracting %s' % imageName)
        outputDirPath = os.path.join(args.outputDirPath, imageName) \
            if args.separateImages else args.outputDirPath
        extractSuperpixels(
            imagePaths['image'],
            imagePaths['superpixels'],
            imagePaths['features'],
            outputDirPath,
            args.keepContext)


if __name__ == '__main__':
   main()
