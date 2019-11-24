from django.http import HttpResponse 
from django.shortcuts import render, redirect, render_to_response
from .forms import *
from grey import greyscale

# Create your views here.

def index(request):
	Content.objects.all().delete()
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
	#Content.objects.all().delete()
	return render(request, 'display_content_images.html', {'content_images' : Contents})

def final_image(request):
	img = greyscale('../media/'+nameoffile)
	
	return render_to_response('final_image.html')     