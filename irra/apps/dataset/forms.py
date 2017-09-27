from django import forms

TRAINSET = ((1, '1 - Train'), (2, '2 - Test'))


class TrainSetForm(forms.Form):
    trainset = forms.MultipleChoiceField(
                    required=False,
                    widget=forms.CheckboxSelectMultiple(attrs={'onclick':'this.form.submit();'}),
                    choices=TRAINSET,
                    # initial=(1, 2),
                )