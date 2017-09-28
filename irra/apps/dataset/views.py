import os
import cv2
import numpy as np
from django.conf import settings
from django.views.generic import ListView, DetailView
from .models import Image, Object, ImageObject
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
        ground_truth_image = np.zeros(image.shape, dtype=image.dtype)
        colors = [('background', "rgb(0, 0, 0)")]

        for obj in ImageObject.objects.filter(image=context['image'], experiment=self.request.GET.get('experiment', 1)):
            ground_truth = Object.objects.get(pk=obj.object.pk)
            color = tuple((ground_truth.b, ground_truth.g, ground_truth.r))
            colors.append((ground_truth.name, "rgb({},{},{})".format(color[2], color[1], color[0])))
            coords = np.vstack((np.fromiter(obj.x.replace('[', '').replace(']', '').split(), dtype=np.float32),
                                np.fromiter(obj.y.replace('[', '').replace(']', '').split(), dtype=np.float32)))
            cv2.drawContours(ground_truth_image, [coords.T.astype(np.int32)], -1, color, -1)

        cv2.addWeighted(ground_truth_image, alpha, image, 1 - alpha, 0, image)

        segmented = cv2.imread(os.path.join(settings.STATIC_ROOT, 'segmented', context['image'].filename.replace('.jpg', '.png')))
        context['segmented'] = image2base64(segmented)
        context['ground_truth'] = image2base64(ground_truth_image)
        context['overlay'] = image2base64(image)
        context['colors'] = sorted(list(set(colors)))
        return context