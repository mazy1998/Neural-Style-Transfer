# forms.py 
from django import forms 
from .models import *

class ContentForm(forms.ModelForm):
	class Meta: 
		model = Content 
		fields = ['content_Main_Img', 'style_Main_Img']
		# 'name',

