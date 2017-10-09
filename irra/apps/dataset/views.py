import os
import cv2
import base64
import numpy as np
from django.conf import settings
from django.views.generic import ListView, DetailView
from .models import Image, Object, ImageObject, Experiment, Query, StructuredQueryDataset, QueryNoun, ObjectIndex, Preposition, QueryGroundTruth
from .mixins import TrainTestFilterMixin, SearchableMixin
from .utils import image2base64


class ImageListView(TrainTestFilterMixin, SearchableMixin, ListView):
    model = Image
    context_object_name = 'images'
    template_name = 'dataset/image_list.html'
    paginate_by = 20
    search_by = 'filename'


class ImageDetailView(DetailView):
    model = Image
    context_object_name = 'image'
    template_name = 'dataset/image_detail.html'

    def get_context_data(self, **kwargs):
        context = super(ImageDetailView, self).get_context_data(**kwargs)
        alpha = 0.7
        image = cv2.imread(os.path.join(settings.STATIC_ROOT, 'dataset', context['image'].filename))

        w, h = settings.IMAGE_SHAPE
        w1, h1 = image.shape[:2]
        fy = w / float(w1)
        fx = h / float(h1)
        scale = np.array([fx, fy])

        image = cv2.resize(image, settings.IMAGE_SHAPE)
        ground_truth_image = np.zeros(image.shape, dtype=image.dtype)
        contour_image = np.zeros(image.shape, dtype=image.dtype)
        colors = []

        experiment = Experiment.objects.get(exp=self.request.GET.get('experiment', '1'))
        for obj in ImageObject.objects.filter(image=context['image'], experiment=experiment):
            ground_truth = Object.objects.get(pk=obj.object.pk)
            color = tuple((ground_truth.b, ground_truth.g, ground_truth.r))
            colors.append((ground_truth.name, "rgb({},{},{})".format(color[2], color[1], color[0])))
            coords = np.vstack((np.fromstring(base64.decodestring(bytes(obj.x, 'utf-8')), dtype=np.float32),
                                np.fromstring(base64.decodestring(bytes(obj.y, 'utf-8')), dtype=np.float32)))
            cv2.drawContours(ground_truth_image, [(coords.T * scale).astype(np.int32)], -1, color, -1)
            cv2.drawContours(contour_image, [(coords.T * scale).astype(np.int32)], -1, color, 3)

        cv2.addWeighted(ground_truth_image, alpha, image, 1 - alpha, 0, image)

        if context['image'].trainset == '2':
            segmented = cv2.imread(os.path.join(settings.STATIC_ROOT,
                                                'segmented', 'test',
                                                context['image'].filename.replace('.jpg', '.png')), 0)
            w, h = segmented.shape
            segmentation = np.zeros((w, h, 3), dtype=np.uint8)
            for pixel_value in np.unique(segmented):
                obj = Object.objects.get(index=pixel_value)
                x, y = np.where(segmented == pixel_value)
                color = tuple((obj.b, obj.g, obj.r))
                colors.append((obj.name, "rgb({},{},{})".format(color[2], color[1], color[0])))
                segmentation[x, y, :] = np.array([obj.b, obj.g, obj.r])
        else:
            segmentation = np.zeros(ground_truth_image.shape, dtype=np.uint8)

        context['segmented'] = image2base64(segmentation)
        context['ground_truth'] = image2base64(ground_truth_image)
        context['overlay'] = image2base64(image)
        context['contour'] = image2base64(contour_image)
        context['colors'] = sorted(list(set(colors)))
        return context


class QueryListView(TrainTestFilterMixin, SearchableMixin, ListView):
    model = Query
    context_object_name = 'queries'
    template_name = 'dataset/query_list.html'
    paginate_by = 10
    search_by = 'query_name'

    def get_queryset(self):
        qs = super(QueryListView, self).get_queryset()
        if self.request.GET.get('query_type'):
            qs = qs.filter(query_type=self.request.GET['query_type'])
        return qs

    def get_context_data(self, **kwargs):
        context = super(QueryListView, self).get_context_data(**kwargs)
        context['query_names'] = []
        for query in context['queries']:
            name = []
            for sq in StructuredQueryDataset.objects.filter(query=query):
                name.append("{} {} {}".format(sq.object1.name,
                                              sq.preposition.name,
                                              sq.object2.name))

            nouns = []
            for noun in QueryNoun.objects.filter(query=query):
                nouns.append(noun.object.name)

            if nouns:
                query_name = ", ".join(name) + ' & ' + ", ".join(nouns)
            else:
                query_name = ", ".join(name)

            context['query_names'].append((query.pk, query_name))

        print(context['query_names'])
        return context


class QueryDetailView(DetailView):
    model = Query
    context_object_name = 'query'
    template_name = 'dataset/query_detail.html'

    def get_context_data(self, **kwargs):
        context = super(QueryDetailView, self).get_context_data(**kwargs)
        context['images'] = [qgt.image for qgt in QueryGroundTruth.objects.filter(query=context['query'])]
        return context
