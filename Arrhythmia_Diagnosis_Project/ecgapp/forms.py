from django import forms

class ECGUploadForm(forms.Form):
    file = forms.FileField(label='Upload ECG File (CSV or Image)')

from django import forms
from django.contrib.auth.models import User

class SimpleUserEditForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['username', 'first_name', 'last_name', 'email', 'is_active']
