from click import progressbar
from scipy.io import loadmat
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from irra.apps.dataset.models import Image, Object, ImageObject


class Command(BaseCommand):
    help = 'Create DB from SUN09'

    def add_arguments(self, parser):
        parser.add_argument('--dataset', dest='dataset', type=str, default=settings.SUN09)
        parser.add_argument('--query', dest='query', type=str, default=settings.QUERY_TRAIN)

    def handle(self, *args, **options):
        db = loadmat(options['dataset'], struct_as_record=True, chars_as_strings=True, squeeze_me=True)
        size = len(db['Dtraining']) + len(db['Dtest'])
        query = loadmat(options['query'], struct_as_record=True, chars_as_strings=True, squeeze_me=True)
        names = query['names'].tolist()

        with progressbar(length=size, show_pos=True, show_percent=True) as bar:
            for trainset, mode in ((1, 'Dtraining'), (2, 'Dtest')):
                for image_index, annotation in enumerate(db[mode]['annotation']):
                    image = Image(filename=annotation['filename'].item(), index=image_index, trainset=trainset)
                    image.save()

                    objects = annotation['object'].item()
                    if annotation['object'].item().size == 1:
                        objects = [annotation['object'].item()]

                    for contour in objects:
                        name = str(contour['name'])
                        if name in names:
                            db_obj = Object(name=name, index=names.index(name))
                            db_obj.save()

                            try:
                                coords = (contour['polygon']['x'].item(), contour['polygon']['y'].item())
                            except IndexError:
                                coords = (contour['polygon'].item()['x'].item(), contour['polygon'].item()['y'].item())

                            image_object = ImageObject(image=image, object=db_obj, x=coords[0], y=coords[1])
                            image_object.save()

                    bar.update(1)