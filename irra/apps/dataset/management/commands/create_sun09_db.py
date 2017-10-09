import base64
from click import progressbar
from scipy.io import loadmat
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from irra.apps.dataset.models import Image, Object, ImageObject, Experiment, ObjectIndex


class Command(BaseCommand):
    help = 'Create DB from SUN09'

    def add_arguments(self, parser):
        parser.add_argument('--dataset', dest='dataset', type=str, default=settings.SUN09)
        parser.add_argument('--query', dest='query', type=str, default=settings.QUERY_TRAIN)
        parser.add_argument('--reset', action='store_true', dest='reset', default=True)

    def handle(self, *args, **options):
        db = loadmat(options['dataset'], struct_as_record=True, chars_as_strings=True, squeeze_me=True)
        size = len(db['Dtraining']) + len(db['Dtest'])
        query = loadmat(options['query'], struct_as_record=True, chars_as_strings=True, squeeze_me=True)
        names = query['names'].tolist()
        aliases = sorted(list(set([settings.ALIASES.get(name, name) for name in names])))
        aliases = {name: index + 1 for index, name in enumerate(aliases)}

        if options['reset']:
            self.delete_all(Experiment)
            self.delete_all(Object)
            self.delete_all(ObjectIndex)
            self.delete_all(Image)
            self.delete_all(ImageObject)

        experiment, _ = Experiment.objects.get_or_create(exp=1)
        experiment.save()

        bkg, _ = Object.objects.get_or_create(name='background', index=0, r=0, g=0, b=0)
        bkg.save()

        for index, name in enumerate(query['names']):
            obj_name = settings.ALIASES.get(name, name)
            db_index = index + 1

            obj_index = aliases[obj_name]
            color = settings.PALETTE[obj_index]
            r, g, b = color
            db_obj, _ = Object.objects.get_or_create(name=obj_name,
                                                     index=obj_index,
                                                     r=r, g=g, b=b)
            db_obj.save()

            obj_index, _ = ObjectIndex.objects.get_or_create(object=db_obj, name=obj_name, alias=name, index=db_index)
            obj_index.save()

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
                        if name in names:
                            obj_index = ObjectIndex.objects.get(alias=name)
                            db_obj = Object.objects.get(pk=obj_index.object.pk)

                            try:
                                coords = (contour['polygon']['x'].item(), contour['polygon']['y'].item())
                            except IndexError:
                                coords = (contour['polygon'].item()['x'].item(), contour['polygon'].item()['y'].item())

                            image_object = ImageObject(image=image,
                                                       object=db_obj,
                                                       experiment=experiment,
                                                       x=base64.encodestring(coords[0].tostring()),
                                                       y=base64.encodestring(coords[1].tostring()))
                            image_object.save()


                    bar.update(1)

    def delete_all(self, model):
        print("Cleaning model {}".format(model))
        for obj in model.objects.all():
            obj.delete()