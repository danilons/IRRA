import os
import glob
import cv2
import numpy as np
from click import progressbar
from django.core.management.base import BaseCommand, CommandError
from keras.models import load_model


class Command(BaseCommand):
    help = 'Create Segmentation dataset'

    def add_arguments(self, parser):
        parser.add_argument('--experiment', dest='experiment', type=int, default=1)
        parser.add_argument('--input_path', dest='input_path', type=str, default='data/segmentation/image/test')
        parser.add_argument('--output_path', dest='output_path', type=str, default='data/segmented/test')
        parser.add_argument('--model', dest='model', type=str, default='data/model.hdf5')
        parser.add_argument('--mean', dest='mean', type=str, default='data/mean.npy')

    def handle(self, *args, **options):
        model = load_model(options['model'])
        mean = np.load(options['mean']).transpose(2, 1, 0)
        imnames = glob.glob(os.path.join(options['input_path'], '*.jpg'))


        with progressbar(length=len(imnames), show_pos=True, show_percent=True) as bar:
            for imname in imnames:
                basename = os.path.basename(imname)
                outfile = os.path.join(options['output_path'], basename.replace('.jpg', '.png'))
                image = cv2.imread(imname) - mean
                image = image[:, :, (2, 1, 0)]
                segmented = model.predict(image[np.newaxis])
                segment = segmented[0].argmax(axis=2)
                cv2.imwrite(outfile, segment)
                bar.update(1)



