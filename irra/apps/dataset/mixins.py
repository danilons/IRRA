
class TrainTestFilterMixin(object):

    def get_queryset(self):
        qs = super(TrainTestFilterMixin, self).get_queryset()
        if self.request.GET.get('trainset'):
            qs = qs.filter(trainset=self.request.GET['trainset'])
        return qs


class ExperimentMixin(object):

    def get_context_data(self, **kwargs):
        context = super(ExperimentMixin, self).get_context_data(**kwargs)
        if self.request.GET.get('experiment'):
            context['experiment'] = self.request.GET.get('experiment')
        return context


class SearchableMixin(object):

    search_by = None

    def get_queryset(self):
        qs = super(SearchableMixin, self).get_queryset()
        term = self.request.GET.get('q')
        if term and self.search_by:
            qs = qs.filter(**{self.search_by + '__contains':term})
        return qs