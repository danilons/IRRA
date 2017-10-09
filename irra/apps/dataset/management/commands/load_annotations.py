from scipy.io import loadmat
import numpy as np
from click import progressbar
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from irra.apps.dataset.models import Preposition, Query, Object, ObjectIndex, StructuredQueryDataset, Image, QueryGroundTruth, QueryNoun


class Command(BaseCommand):
    help = 'Create query ground truth dataset'

    def add_arguments(self, parser):
        parser.add_argument('--query', dest='query', type=str, default=settings.QUERY_TEST)
        parser.add_argument('--reset', action='store_true', dest='reset', default=True)

    def handle(self, *args, **options):
        db = loadmat(options['query'], struct_as_record=True, chars_as_strings=True, squeeze_me=True)

        if options['reset']:
            # delete all
            self.delete_all(Preposition)
            self.delete_all(StructuredQueryDataset)
            self.delete_all(Query)
            self.delete_all(QueryNoun)
            self.delete_all(QueryGroundTruth)

        trainset = 1 if options['query'] == settings.QUERY_TRAIN else 2
        for index, name in enumerate(db['relations']):
            if name in ('above', 'below'):
                prep, _ = Preposition.objects.get_or_create(index=index + 1, name=name)
                prep.save()

        with progressbar(length=len(db['Query']), show_pos=True, show_percent=True) as bar:
            for db_query in db['Query']:
                noun, query, ground_truth = db_query
                if query.shape == (3, 3):
                    query_type = 'e'
                    for q in query:
                        self.save_query(noun, q, ground_truth, query_type, trainset)

                elif len(query) == 3:
                    query_type = 'b' if isinstance(noun, int) else 'a'
                    self.save_query(noun, query, ground_truth, query_type, trainset)

                elif len(query) == 2:
                    query_type = 'd' if isinstance(noun, int) else 'c'
                    for q in query:
                        self.save_query(noun, q, ground_truth, query_type, trainset)

                bar.update(1)

    def save_query(self, noun, query, ground_truth, query_type, trainset):
        obj1_, obj2_, prep_ = query

        obj_index = ObjectIndex.objects.get(index=obj1_)
        obj1 = Object.objects.get(pk=obj_index.object.pk)

        obj_index = ObjectIndex.objects.get(index=obj2_)
        obj2 = Object.objects.get(pk=obj_index.object.pk)
        try:
            relation = Preposition.objects.get(index=prep_)
        except:
            print("Preposition not found {}".format(prep_))
            return

        query_name = "{obj1}-{relation}-{obj2}".format(obj1=obj1.name,
                                                       relation=relation.name,
                                                       obj2=obj2.name)

        # save queries
        db_query, _ = Query.objects.get_or_create(query_type=query_type, query_name=query_name, trainset=trainset)
        db_query.save()

        sq, _ = StructuredQueryDataset.objects.get_or_create(query=db_query,
                                                             object1=obj1,
                                                             object2=obj2,
                                                             preposition=relation)
        sq.save()

        if isinstance(noun, int):
            db_obj = ObjectIndex.objects.get(index=noun)
            obj_ = Object.objects.get(pk=db_obj.object.pk)
            qn, _ = QueryNoun.objects.get_or_create(query=db_query, object=obj_)
            qn.save()

        valids = np.nonzero(ground_truth)[0]
        for img in valids:
            image = Image.objects.get(index=img, trainset=trainset)
            qgt, _ = QueryGroundTruth.objects.get_or_create(query=db_query, image=image)
            qgt.save()

    def delete_all(self, model):
        print("Cleaning model {}".format(model))
        for obj in model.objects.all():
            obj.delete()