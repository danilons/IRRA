import os
import numpy as np
import cv2
from click import progressbar
from django.core.management.base import BaseCommand, CommandError
from irra.apps.dataset.models import Image, Object, ImageObject, Experiment


class Command(BaseCommand):
    help = 'Create Segmentation dataset'

    def add_arguments(self, parser):
        parser.add_argument('--experiment', dest='experiment', type=int, default=1)
        parser.add_argument('--input_path', dest='input_path', type=str, default='data/images/dataset')
        parser.add_argument('--output_path', dest='output_path', type=str, default='data/segmentation')
        parser.add_argument('--force', action='store_true', dest='force', default=False)


    def handle(self, *args, **options):
        experiment = Experiment.objects.get(exp=options['experiment'])
        alias = {'1': 'train', '2': 'test'}
        with progressbar(length=Image.objects.count(), show_pos=True, show_percent=True) as bar:
            for image in Image.objects.all():
                fullpath = os.path.join(options['output_path'], alias[image.get_trainset_display()])
                if not os.path.exists(fullpath):
                    os.makedirs(fullpath)

                imname = image.filename.replace('.jpg', '.png')
                if os.path.exists(imname) and not options['force']:
                    bar.update(1)
                    continue

                img = cv2.imread(os.path.join(options['input_path'], image.filename))
                if img is None:
                    print("Unable to read image {}".format(image.filename))
                    bar.update(1)
                    continue

                ground_truth_image = np.zeros(img.shape, dtype=img.dtype)
                colors = [('background', "rgb(0, 0, 0)")]

                for obj in ImageObject.objects.filter(image=image,experiment=experiment):
                    ground_truth = Object.objects.get(pk=obj.object.pk)
                    color = tuple((ground_truth.b, ground_truth.g, ground_truth.r))
                    colors.append((ground_truth.name, "rgb({},{},{})".format(color[2], color[1], color[0])))
                    coords = np.vstack((np.fromiter(obj.x.replace('[', '').replace(']', '').split(), dtype=np.float32),
                                        np.fromiter(obj.y.replace('[', '').replace(']', '').split(), dtype=np.float32)))
                    cv2.drawContours(ground_truth_image, [coords.T.astype(np.int32)], -1, color, -1)

                imname = image.filename.replace('.jpg', '.png')
                cv2.imwrite(os.path.join(fullpath, imname), ground_truth_image)
                bar.update(1)