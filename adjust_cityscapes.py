import re
import numpy as np
import os.path
import shutil
from PIL import Image
from glob import glob
import sshelper


def resize(city):
# resize and manipulate cityscapes images to prep them for use in our model
    source_folder = '/home/teeekaay/Car-ND/CarND-Semantic-Segmentation/data/cityscapes'
    image_paths = glob(os.path.join(source_folder, 'leftImg8bit/train/', city, '*.png'))
    label_paths = glob(os.path.join(source_folder, 'gtFine/train/', city, '*_gtFine_color.png'))
    output_image_folder = '/home/teeekaay/Car-ND/CarND-Semantic-Segmentation/data/combined/images'
    output_labels_folder = '/home/teeekaay/Car-ND/CarND-Semantic-Segmentation/data/combined/labels'
    for images in image_paths:
        im = Image.open(images)
        im = im.crop((0, 228, 2047, 796))
        im = im.resize((576, 160), Image.LANCZOS)
        outname = os.path.join(output_image_folder, os.path.basename(images))
        im.save(outname)

    for images in label_paths:
        im = Image.open(images)
        im = im.crop((0, 228, 2047, 796))
        im = im.resize((576, 160), Image.LANCZOS)
        #im.load()  # needed for split()
        color = (255,255,255)
        background = Image.new('RGB', im.size, color)
        background.paste(im, mask=im.split()[3])
        np_img = np.array(background)
        print("shape = {}".format(np_img.shape))
        road_color = np.array([128, 64, 128])
        # red, green, blue = np_img[:, :, 0], np_img[:, :, 1], np_img[:, :, 2]
        mask = np.all(np_img != road_color, axis=2)
        # (red != road_color[0]) or (green != road_color[1]) or (blue != road_color[2])
        np_img[:, :, :3][mask] = [255, 0, 0]
        im = Image.fromarray(np_img)
        outname = os.path.join(output_labels_folder, os.path.basename(images))
        im.save(outname)

def resize2(city):
# resize and manipulate cityscapes images to prep them for use in our model
    source_folder = '/home/teeekaay/Car-ND/CarND-Semantic-Segmentation/data/cityscapes'
    image_paths = glob(os.path.join(source_folder, 'leftImg8bit/train/', city, '*.png'))
    label_paths = glob(os.path.join(source_folder, 'gtFine/train/', city, '*_gtFine_color.png'))
    output_image_folder = '/home/teeekaay/Car-ND/CarND-Semantic-Segmentation/data/combined/images'
    output_labels_folder = '/home/teeekaay/Car-ND/CarND-Semantic-Segmentation/data/combined/labels'
    for images in image_paths:
        im = Image.open(images)
        im = im.crop((0, 228, 2047, 796))
        im = im.resize((576, 160), Image.LANCZOS)
        outname = os.path.join(output_image_folder, os.path.basename(images))
        im.save(outname)

    for images in label_paths:
        im = Image.open(images)
        im = im.crop((0, 228, 2047, 796))
        im = im.resize((576, 160), Image.LANCZOS)
        #im.load()  # needed for split()
        color = (255,255,255)
        background = Image.new('RGB', im.size, color)
        background.paste(im, mask=im.split()[3])
        np_img = np.array(background)
        print("shape = {}".format(np_img.shape))
        road_color = np.array([128, 64, 128])
        # red, green, blue = np_img[:, :, 0], np_img[:, :, 1], np_img[:, :, 2]
        mask = np.all(np_img != road_color, axis=2)
        # (red != road_color[0]) or (green != road_color[1]) or (blue != road_color[2])
        np_img[:, :, :3][mask] = [255, 0, 0]
        im = Image.fromarray(np_img)
        outname = os.path.join(output_labels_folder, os.path.basename(images))
        im.save(outname)
        

resize('darmstadt')
resize('cologne')
resize('bochum')
resize('bremen')
resize('aachen')
resize('dusseldorf')
resize('erfurt')
resize('hamburg')
resize('hanover')
resize('jena')
resize('krefield')
resize('monchengladbach')
resize('strasbourg')
resize('stuttgart')
resize('tubingen')
resize('ulm')
resize('weimar')
resize('zurich')


