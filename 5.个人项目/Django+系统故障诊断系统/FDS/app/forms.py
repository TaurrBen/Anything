from django import forms
from .models import UploadTrainCSV

class uploadTrainForm(forms.Form):
    class Meta:
        model = UploadTrainCSV
        fields = ('date_uploaded', 'csv_file')

class uploadTestForm(forms.Form):
    csv_file = forms.FileField(widget=forms.FileInput(attrs={'accept': ".csv"}))
