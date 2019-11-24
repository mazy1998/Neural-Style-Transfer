from django.http import HttpResponse 
from django.shortcuts import render, redirect, render_to_response
from .forms import *

# Create your views here.

def index(request):
	return render_to_response('index.html')

# Create your views here. 
def content_image_view(request): 

	if request.method == 'POST': 
		form = ContentForm(request.POST, request.FILES) 

		if form.is_valid(): 
			form.save() 
			return redirect('success') 
	else: 
		form = ContentForm() 
	return render(request, 'content_image_form.html', {'form' : form}) 


def success(request):
	#return HttpResponse('successfuly uploaded')
	Contents = Content.objects.all()
	return render(request, 'display_content_images.html', {'content_images' : Contents})

# def display_content_images(request): 
#     if request.method == 'GET': 
  
#          