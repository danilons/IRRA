from click import progressbar
from scipy.io import loadmat
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from irra.apps.dataset.models import Image, Object, ImageObject, Experiment


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
        aliases = sorted(list(set([settings.ALIASES.get(name, name) for name in names])))

        experiment, _ = Experiment.objects.get_or_create(exp=1)
        experiment.save()

        bkg, _ = Object.objects.get_or_create(name='__background__', db_index=0, index=0, r=0, g=0, b=0)
        bkg.save()

        with progressbar(length=size, show_pos=True, show_percent=True) as bar:
            for trainset, mode in ((1, 'Dtraining'), (2, 'Dtest')):
                for image_index, annotation in enumerate(db[mode]['annotation']):
                    image, _ = Image.objects.get_or_create(filename=annotation['filename'].item(),
                                                           index=image_index,
                                                           trainset=trainset)
                    image.save()

                    objects = annotation['object'].item()
                    if annotation['object'].item().size == 1:
                        objects = [annotation['object'].item()]

                    for contour in objects:
                        name = str(contour['name'])
                        name = settings.ALIASES.get(name, name)
                        if name in names:
                            index = names.index(name)
                            db_index = aliases.index(name) + 1      # aliases do not contain background
                            color = settings.PALETTE[db_index]
                            r, g, b = color
                            db_obj, created = Object.objects.get_or_create(name=name,
                                                                           db_index=db_index,
                                                                           index=index,
                                                                           r=r, g=g, b=b)
                            if not created:
                                db_obj.save()

                            try:
                                coords = (contour['polygon']['x'].item(), contour['polygon']['y'].item())
                            except IndexError:
                                coords = (contour['polygon'].item()['x'].item(), contour['polygon'].item()['y'].item())

                            image_object = ImageObject(image=image,
                                                       object=db_obj,
                                                       experiment=experiment,
                                                       x=coords[0].tostring(),
                                                       y=coords[1].tostring())
                            image_object.save()


                    bar.update(1)