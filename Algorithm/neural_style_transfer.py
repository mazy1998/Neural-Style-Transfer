# -*- coding: utf-8 -*-
"""neural_style_transfer.ipynb
Original file is located at
    https://colab.research.google.com/drive/1pf318xOdfipfbZ9UjIjrBBUQkwY0FbCF
"""

!pip uninstall d2l
!pip uninstall mxnet
!pip uninstall mxnet_cu100
!pip install d2l==0.10.3
!pip install mxnet==1.6.0b20190915
#!pip install https://apache-mxnet.s3-accelerate.amazonaws.com/dist/python/numpy/latest/mxnet-1.5.0-py2.py3-none-manylinux1_x86_64.whl
!pip install mxnet_cu100
!pip install matplotlib

ctx = d2l.try_gpu()
ctx

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import d2l
from mxnet import autograd, gluon, image, init, nd
from mxnet.gluon import nn
import matplotlib
import numpy as np

d2l.set_figsize((3.5, 2.5))
content_img = image.imread("lol.jpg")
print(content_img.shape)
d2l.plt.imshow(content_img.asnumpy())

style_img = image.imread("scream.jpg")

d2l.plt.imshow(style_img.asnumpy())
print(style_img.shape)

def rgb_mean_calc(img):
  mat = img.transpose((2, 0, 1))
  mat = mat.asnumpy()
  mean_rgb = []
  mean_std = []
  for i in range(3):
    mean_rgb.append(np.mean(mat[i,:,:])/255)
    mean_std.append(np.std(mat[i,:,:])/255)

  # print(nd.array(mean_rgb))
  # print(nd.array(mean_std))
  return nd.array(mean_rgb),nd.array(mean_std)
 

rgb_mean_calc(content_img)

#rgb_mean = nd.array([0.485, 0.456, 0.406])
#rgb_std = nd.array([0.229, 0.224, 0.225])

#def preprocess(img,image_shape):
#  img = image.imresize(img, *image_shape)
#  img = (img.astype('float32') / 255)
#  img = img.transpose((2,0,1)) #makes it channel first
#  return img.expand_dims(axis=0)

#def postprocess(img):
#  img = nd.squeeze(img , axis=0) 
#  img = img.transpose((1, 2, 0))
#  img = img*255
#  return img

rgb_mean = nd.array([0.485, 0.456, 0.406])
rgb_std = nd.array([0.229, 0.224, 0.225])

# rgb_mean, rgb_std = rgb_mean_calc(style_img)

def preprocess(img, image_shape):
  img = image.imresize(img, *image_shape)
  img = (img.astype('float32') / 255 - rgb_mean) / rgb_std 
  return img.transpose((2, 0, 1)).expand_dims(axis=0) #transforms it shape (1,3,200,200) and of (BRG)

def postprocess(img):
  img = img[0].as_in_context(rgb_std.context)
  return (img.transpose((1, 2, 0)) * rgb_std + rgb_mean).clip(0, 1) #reverts to (200,200,3)


image0 = preprocess(style_img,(200,200))
image1 = postprocess(image0)
d2l.plt.imshow(image1.asnumpy())

pretrained_net = gluon.model_zoo.vision.vgg19(pretrained=True)

print(pretrained_net)

style_layers, content_layers = [0, 5, 10], [18]
net = nn.Sequential()
for i in range(max(content_layers + style_layers) + 1):
    net.add(pretrained_net.features[i])

def extract_features(X,content_layers, style_layers):
  contents = []
  styles = []
  for i in range(len(net)):
    X = net[i](X)
    if i in style_layers:
      styles.append(X)
    if i in content_layers:
      contents.append(X)
  return contents, styles

def get_contents(image_shape, ctx):
  content_X = preprocess(content_img, image_shape).copyto(ctx)
  contents_Y, _ = extract_features(content_X, content_layers, style_layers)
  return content_X, contents_Y

def get_styles(image_shape, ctx):
  style_X = preprocess(style_img, image_shape).copyto(ctx)
  _, styles_Y = extract_features(style_X, content_layers, style_layers)
  return style_X, styles_Y

def content_loss(Y_hat, Y):
  return (Y_hat - Y).square().mean()

def gram(X):
  num_channels, n = X.shape[1], X.size // X.shape[1] 
  X = X.reshape((num_channels, n))
  return nd.dot(X, X.T) / (num_channels * n)

def style_loss(Y_hat, gram_Y):
  return (gram(Y_hat) - gram_Y).square().mean()

def tv_loss(Y_hat):
  return 0.5 * ((Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).abs().mean() +
                  (Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).abs().mean())

content_weight, style_weight, tv_weight = 1, 5000, 10
def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram): 
  # Calculate the content, style, and total variance losses respectively 
  contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
      contents_Y_hat, contents_Y)]
  styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
      styles_Y_hat, styles_Y_gram)]
  tv_l = tv_loss(X) * tv_weight
  # Add up all the losses
  l = nd.add_n(*styles_l) + nd.add_n(*contents_l) + tv_l
  return contents_l, styles_l, tv_l, l

class GeneratedImage(nn.Block):
  def __init__(self, img_shape, **kwargs):
          super(GeneratedImage, self).__init__(**kwargs)
          self.weight = self.params.get('weight', shape=img_shape)
  def forward(self):
    return self.weight.data()

def get_inits(X, ctx, lr, styles_Y):
  gen_img = GeneratedImage(X.shape)
  gen_img.initialize(init.Constant(X), ctx=ctx, force_reinit=True)
  trainer = gluon.Trainer(gen_img.collect_params(), 'adam',{'learning_rate': lr})
  styles_Y_gram = [gram(Y) for Y in styles_Y]
  return gen_img(), styles_Y_gram, trainer

def train(X, contents_Y, styles_Y, ctx, lr, num_epochs, lr_decay_epoch):
  X, styles_Y_gram, trainer = get_inits(X, ctx, lr, styles_Y)
  animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, num_epochs],
                            legend=['content', 'style', 'TV'],
                            ncols=2, figsize=(7,2.5))
  
  for epoch in range(1, num_epochs+1):
    with autograd.record():
            contents_Y_hat, styles_Y_hat = extract_features(
                X, content_layers, style_layers)
            contents_l, styles_l, tv_l, l = compute_loss(
                X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
            
    l.backward()
    trainer.step(1)
    nd.waitall()
    if epoch % lr_decay_epoch == 0:
      trainer.set_learning_rate(trainer.learning_rate * 0.1)
    # if epoch%10 == 0:
    #   print(epoch)
    if epoch % 10 == 0:
      # animator.axes[1].imshow(postprocess(X).asnumpy())
      animator.add(epoch, [nd.add_n(*contents_l).asscalar(),
                            nd.add_n(*styles_l).asscalar(), tv_l.asscalar()])
    if epoch % 100 == 0:
      d2l.plt.imsave('neural-style'+str(epoch)+'.png', postprocess(X).asnumpy())

    


  return X

d2l.try_gpu()

ctx, image_shape = d2l.try_gpu(), (120, 200)
net.collect_params().reset_ctx(ctx)
content_X, contents_Y = get_contents(image_shape, ctx)
_, styles_Y = get_styles(image_shape, ctx)
output = train(content_X, contents_Y, styles_Y, ctx, 0.01, 200, 200)
output1 = postprocess(output)
output1 = output1.asnumpy()
matplotlib.image.imsave('name1.png', output1)

ctx, image_shape = d2l.try_gpu(), (758, 948) 
_, content_Y = get_contents(image_shape, ctx)
_, style_Y = get_styles(image_shape, ctx)
X = preprocess(postprocess(output) * 255, image_shape)
output = train(X, content_Y, style_Y, ctx, 0.01, 300, 300)

d2l.plt.imsave('neural-style.png', postprocess(output).asnumpy())





net.collect_params().reset_ctx(d2l.try_gpu())

!pip install numpy

import numpy as np

!pip install matplotlib
import matplotlib



