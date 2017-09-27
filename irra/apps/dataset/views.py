from django.shortcuts import render
from django.views.generic import ListView
from django.views.generic.edit import FormMixin
from .models import Image
from .forms import TrainSetForm


class ImagesView(ListView):
    model = Image
    context_object_name = 'images'
    template_name = 'dataset/image_list.html'
    paginate_by = 20

    def get_queryset(self):
        qs = super(ImagesView, self).get_queryset()
        if self.request.GET.get('trainset'):
            return qs.filter(trainset=self.request.GET['trainset'])
        return qs