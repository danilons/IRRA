import os
import numpy as np
import cv2
from click import progressbar
from django.conf import settings
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
                # imname = image.filename.replace('.jpg', '.png')
                # if os.path.exists(imname) and not options['force']:
                #     bar.update(1)
                #     continue

                img = cv2.imread(os.path.join(options['input_path'], image.filename))
                if img is None:
                    print("Unable to read image {}".format(image.filename))
                    bar.update(1)
                    continue

                w, h = settings.IMAGE_SHAPE
                w1, h1 = img.shape[:2]
                fy = w / float(w1)
                fx = h / float(h1)
                scale = np.array([fx, fy])
                img = cv2.resize(img, (w, h))

                ground_truth_image = np.zeros(img.shape[:2], dtype=img.dtype)

                for obj in ImageObject.objects.filter(image=image,experiment=experiment):
                    ground_truth = Object.objects.get(pk=obj.object.pk)
                    # color = tuple((ground_truth.b, ground_truth.g, ground_truth.r))
                    color = ground_truth.db_index
                    coords = np.vstack((np.fromiter(obj.x.replace('[', '').replace(']', '').split(), dtype=np.float32),
                                        np.fromiter(obj.y.replace('[', '').replace(']', '').split(), dtype=np.float32)))
                    cv2.drawContours(ground_truth_image, [(coords.T * scale).astype(np.int32)], -1, color, -1)

                fullpath = os.path.join(options['output_path'], 'image', alias[image.get_trainset_display()])
                cv2.imwrite(os.path.join(fullpath, image.filename), img)

                fullpath = os.path.join(options['output_path'], 'label', alias[image.get_trainset_display()])
                imname = image.filename.replace('.jpg', '.png')
                cv2.imwrite(os.path.join(fullpath, imname), ground_truth_image)
                bar.update(1)