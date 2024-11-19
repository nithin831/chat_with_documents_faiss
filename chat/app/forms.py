# from django import forms

# from django.core.exceptions import ValidationError


# class DocForm(forms.Form):
#     file = forms.FileField(
#         widget=forms.FileInput(attrs={'class': 'form-control-file'}),
#         label=''
#     )

#     def clean_file(self):
#         file = self.cleaned_data.get('file', False)
#         if not file:
#             raise forms.ValidationError("Please upload a file.")    
#         return file
