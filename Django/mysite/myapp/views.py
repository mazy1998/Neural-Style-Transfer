from django.http import HttpResponse 
from django.shortcuts import render, redirect, render_to_response
from .forms import *

# Create your views here.

def index(request):
	Content.objects.all().delete()
	return render_to_response('index.html')

def gallery(request):
	return render_to_response('gallery.html')

def gallery2(request):
	return render_to_response('gallery2.html')

# Create your views here. 
def content_image_view(request): 

	if request.method == 'POST': 
		form = ContentForm(request.POST, request.FILES) 

		if form.is_valid(): 
			form.save() 
	else: 
		form = ContentForm() 
	return render(request, 'content_image_form.html', {'form' : form}) 

def success(request):
	Contents = Content.objects.all()
	return render(request, 'display_content_images.html', {'content_images' : Contents})

def final_image(request):
	return render_to_response('final_image.html') 









