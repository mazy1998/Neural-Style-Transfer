from django.db import models

class Content(models.Model): 
	name = models.CharField(max_length=50) 
	content_Main_Img = models.ImageField(upload_to='images/')
	style_Main_Img = models.ImageField(upload_to='images/')

