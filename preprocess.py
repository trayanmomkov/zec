# Copyright 2020 Trayan Momkov
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

from __future__ import print_function

import sys
print('Python version: ', sys.version, '\n')

import os
import errno
import random
import csv
from PIL import Image
from os import listdir
from os.path import isfile, join
from shutil import copy2

if hasattr(__builtins__, 'raw_input'):
    input = raw_input

IMAGE_W = 32
IMAGE_H = 32
WHITE = 255
BLACK = 0
GREY = 128


def mkdir(dir_path):
    """
    Creates dir with its parents without error if already exists.
    """
    try:
        os.makedirs(dir_path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(dir_path):
            pass


def extract_number_from_filename(image_filename):
    return os.path.basename(image_filename)[0]


def count_numbers(training_and_validation_files, test_files):
    training_eights, training_zeroes, test_eights, test_zeroes = 0, 0, 0, 0
    for file in training_and_validation_files:
        number = extract_number_from_filename(file)
        if number == '0':
            training_zeroes += 1
        else:
            training_eights += 1

    for file in test_files:
        number = extract_number_from_filename(file)
        if number == '0':
            test_zeroes += 1
        else:
            test_eights += 1

    return training_eights, training_zeroes, test_eights, test_zeroes


def add_to_csv(image, number, csv_writer):
    pixels = image.load()
    width = image.size[0]
    height = image.size[1]

    values = [number]
    for x in range(width):
        for y in range(height):
            values.append(str(pixels[x, y]))

    csv_writer.writerow(values)


def find_black_and_white_average(image):
    width, height = image.size
    pixels = image.load()
    total = 0
    for x in range(width):
        for y in range(height):
            total += pixels[x, y]

    return round(total / float(width * height))


def scale(image):
    max_length_pixels = IMAGE_W
    scale_factor = max(
                image.width / float(max_length_pixels),
                image.height / float(max_length_pixels))

    return image.resize((int(image.width / scale_factor), int(image.height / scale_factor)))


def convert_to_black_and_white(image):
    threshold = find_black_and_white_average(image)
    pixels = image.load()

    width, height = image.size
    for x in range(width):
        for y in range(height):
            if pixels[x, y] < threshold:
                image.putpixel((x, y), BLACK)
            else:
                image.putpixel((x, y), WHITE)
    return image


def add_frame(image):
    old_size = image.size
    new_size = (IMAGE_W, IMAGE_H)
    new_image = Image.new(mode=image.mode, color=255, size=new_size)
    new_image.paste(image, (int((new_size[0]-old_size[0])/2.0), int((new_size[1]-old_size[1])/2.0)))
    return new_image


def center_image(image):
    total_mass_x = 0
    top = IMAGE_H
    bottom = 0
    black_pixels_count = 0
    total_black_pixels_mass = 0
    mean_color_value = 0

    pixels = image.load()
    for x in range(IMAGE_W):
        for y in range(IMAGE_H):
            mean_color_value += pixels[x, y]

    mean_color_value /= float(IMAGE_W * IMAGE_H)

    for x in range(IMAGE_W):
        for y in range(IMAGE_H):
            color_value = pixels[x, y]

            if color_value < mean_color_value:
                mass = (255 - color_value)
                total_mass_x += mass * x
                total_black_pixels_mass += mass
                black_pixels_count += 1

                top = y if y < top else top
                bottom = y if y > bottom else bottom

    mean_black_pixel_mass = total_black_pixels_mass / float(black_pixels_count)
    center_weight_x = round(total_mass_x / mean_black_pixel_mass / float(black_pixels_count))
    center_form_y = round(top + (bottom-top) / float(2))

    return move(image, pixels, round(IMAGE_W / 2.0 - center_weight_x), round(IMAGE_H / 2.0 - center_form_y))


def move(image, pixels, diff_x, diff_y):
    new_pixels = []

    for x in range(IMAGE_W):
        new_pixels.append([])
        for y in range(IMAGE_H):
            old_x = x - diff_x
            old_y = y - diff_y

            if old_x < 0 or old_x >= IMAGE_W or old_y < 0 or old_y >= IMAGE_H:
                new_pixels[x].append(255)
            else:
                new_pixels[x].append(pixels[old_x, old_y])

    for x in range(IMAGE_W):
        for y in range(IMAGE_H):
            image.putpixel((x, y), new_pixels[x][y])

    return image


def convert_to_monochrome(image):
    return image.convert('L')


def split(files, for_training):
    randomness = random.Random()
    ok = 'n'
    while ok != 'y':
        randomness.shuffle(files)
        split_index = int(round(len(files) * for_training))

        original_training_files = files[:split_index]
        original_test_files = files[split_index:]

        training_eights, training_zeroes, test_eights, test_zeroes \
            = count_numbers(original_training_files, original_test_files)
        print('training zeroes:', training_zeroes)
        print('training eights', training_eights)
        print('test zeroes', test_zeroes)
        print('test eights', test_eights)

        ok = input('Is the distribution ok (y/n)?: ')

    return original_training_files, original_test_files


def preprocess_files(files, src_dir, dst_dir, csv_filename):
    with open(csv_filename, 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        for file in files:
            image = center_image(
                        add_frame(
                            convert_to_black_and_white(
                                scale(
                                    convert_to_monochrome(Image.open(join(src_dir, file)))))))

            number = extract_number_from_filename(file)
            add_to_csv(image, number, csv_writer)
            image.save(join(dst_dir, file), "png")


def main():
    original_dir = join('dataset', 'original')
    preprocessed_dir = join('dataset', 'preprocessed')
    training_dir = join(preprocessed_dir, 'training')
    test_dir = join(preprocessed_dir, 'test')

    mkdir(training_dir)
    [os.remove(join(training_dir, f)) for f in os.listdir(training_dir) if f.endswith(".png")]

    mkdir(test_dir)
    [os.remove(join(test_dir, f)) for f in os.listdir(test_dir) if f.endswith(".png")]

    files = [f for f in listdir(original_dir) if isfile(join(original_dir, f))]
    original_training_files, original_test_files = split(files, 0.9)    # 90% for training

    preprocess_files(files=original_training_files,
                     src_dir=original_dir,
                     dst_dir=training_dir,
                     csv_filename=join('dataset', 'training.csv'))

    preprocess_files(files=original_test_files,
                     src_dir=original_dir,
                     dst_dir=test_dir,
                     csv_filename=join('dataset', 'test.csv'))


if __name__ == '__main__':
    main()
