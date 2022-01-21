import torch

from efficientnet_pytorch import EfficientNet

from wrappedefficientnet import *
from lrp_general6 import *

from getimagenetclasses import *
from heatmaphelpers import *
from dataset_imagenet2500 import dataset_imagenetvalpart_nolabels


import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import  DataLoader
from torchvision import transforms, utils


import numpy as np
import matplotlib.pyplot as plt

def run():



  model = EfficientNet.from_pretrained('efficientnet-b0')

  #model = EfficientNet_canonized.from_pretrained('efficientnet-b0')
  
  model_e = EfficientNet_canonized.from_name('efficientnet-b0')
  
  model_e.copyfromefficientnet( model, lrp_params, lrp_layer2method)
  
  for i, (nm,mod) in enumerate( model_e.named_modules()):
    print(i,nm,mod)


    
    
    
def test_model3(dataloader,  model, device):

  from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock

  model.train(False)
  for data in dataloader:
    # get the inputs
    #inputs, labels, filenames = data
    inputs=data['image']
    labels=data['label']    
    fns=data['filename']  

    inputs = inputs.to(device)
    labels = labels.to(device)

    print(inputs.shape)
    with torch.no_grad():
      outputs = model(inputs)
      print('shp ', outputs.shape)
      m=torch.mean(outputs)
      print(m.item() )
      print(fns)

      return m.item(), outputs, model.tmpfeats , torch.max(outputs)
      

def run2():

  #beta0
  skip = 21
  device = torch.device('cpu')
  
  lrp_params_def1={
    'conv2d_ignorebias': True, 
    'eltwise_eps': 1e-6,
    'linear_eps': 1e-6,
    'pooling_eps': 1e-6,
    'use_zbeta': False ,
  }

  lrp_layer2method={
    'Swish':          relu_wrapper_fct,
    'nn.BatchNorm2d':   relu_wrapper_fct,
    'nn.Conv2d':         Conv2dDynamicSamePadding_beta0_wrapper_fct,
    'nn.Linear':        linearlayer_eps_wrapper_fct,  
    'nn.AdaptiveAvgPool2d': adaptiveavgpool2d_wrapper_fct,
    'sum_stacked2': eltwisesum_stacked2_eps_wrapper_fct,
  }
  model = EfficientNet.from_pretrained('efficientnet-b0', image_size=None, dropout_rate= 0.0 , drop_connect_rate=0.0)
  model.set_swish( memory_efficient=False)

  #model._global_params.dro
  
  model_e = EfficientNet_canonized.from_pretrained('efficientnet-b0', image_size=None, dropout_rate= 0.0 , drop_connect_rate=0.0)
  #model_e = EfficientNet_canonized.from_name('efficientnet-b0', dropout_rate= 0.0 , drop_connect_rate=0.0 ,image_size=[224,224])
  model_e.set_swish( memory_efficient=False)
  
  model_e.copyfromefficientnet( model, lrp_params = lrp_params_def1 , lrp_layer2method = lrp_layer2method )
  
  for i, (nm,mod) in enumerate( model_e.named_modules()):
    print(i,nm,mod)
  #exit()

  data_transform = transforms.Compose([

          transforms.Resize(224),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) # if you do five crop, then you must change this part here, as it cannot be applied to 4 tensors

      ])

  root_dir='/home/binder/entwurf9/data/imagenetvalimg/'

  dset= dataset_imagenetvalpart_nolabels(root_dir, maxnum=1, transform=data_transform, skip= skip)
  dataloader =  torch.utils.data.DataLoader(dset, batch_size=1, shuffle=False) #, num_workers=1) 



  m1, outputs1, f1,mx1 = test_model3(dataloader,  model, device=device)
  print('here')
  m2, outputs2,f2, mx2 = test_model3(dataloader,  model_e, device=device)

  print('\n\n m1,m2',m1,m2  )
  print('diff of means: ', m1-m2)
  print('MAE diff of logits: ',  torch.mean(torch.abs(outputs1-outputs2)).item()  )
  print('MAE diff in largest logit: ',  (mx1-mx2).item() )
  print('MAE diff of ft: ',  torch.mean(torch.abs(f1-f2)).item()  )
  
  d= torch.abs(f1-f2)
  
  print(d[0,1,:,:]  )
  ax = plt.subplot()
  im = ax.imshow(d[0,1,:,:].numpy())
  from mpl_toolkits.axes_grid1 import make_axes_locatable
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)

  plt.colorbar(im, cax=cax)
  
  plt.show()

def test_model5(dataloader,  model, device):

  model.train(False)



  for data in dataloader:
    # get the inputs
    #inputs, labels, filenames = data
    inputs=data['image']
    labels=data['label']    
    fns=data['filename']  

    inputs = inputs.to(device).clone()
    labels = labels.to(device)

    inputs.requires_grad=True

    print(inputs.requires_grad)
    with torch.enable_grad():
      outputs = model(inputs)

    vals,cls = torch.max(outputs, dim=1)
    outputs[0,cls].backward()

    print(inputs.grad.shape)
    rel=inputs.grad.data
    print( torch.max(rel), torch.mean(rel) )

    clsss=get_classes()


    with torch.no_grad():

      print('shp ', outputs.shape)
      vals,cls = torch.max(outputs, dim=1)
      m=torch.mean(outputs)
      print(  vals.item(), clsss[cls.item()], m.item() )
      print(fns)

    imshow2(rel.to('cpu'),imgtensor = inputs.to('cpu'))


def run3(skip):

  #beta0
  
  device = torch.device('cpu')
  
  lrp_params_def1={
    'conv2d_ignorebias': True, 
    'eltwise_eps': 1e-6,
    'linear_eps': 1e-6,
    'pooling_eps': 1e-6,
    'use_zbeta': True ,
    'conv2d_beta': 1.0,
    'conv2d_maxbeta': 3.0,
  }

  lrp_layer2method={
    'Swish':          relu_wrapper_fct,
    'nn.BatchNorm2d':   relu_wrapper_fct,
    #'nn.Conv2d':         Conv2dDynamicSamePadding_beta0_wrapper_fct,
    'nn.Conv2d':         Conv2dDynamicSamePadding_betaany_wrapper_fct,
    #'nn.Conv2d':         Conv2dDynamicSamePadding_betaadaptive_wrapper_fct,
    'nn.Linear':        linearlayer_eps_wrapper_fct,  
    'nn.AdaptiveAvgPool2d': adaptiveavgpool2d_wrapper_fct,
    'sum_stacked2': eltwisesum_stacked2_eps_wrapper_fct,
  }
  model = EfficientNet.from_pretrained('efficientnet-b0', image_size=None, dropout_rate= 0.0 , drop_connect_rate=0.0)
  model.set_swish( memory_efficient=False)

  #model._global_params.dro
  
  model_e = EfficientNet_canonized.from_pretrained('efficientnet-b0', image_size=None, dropout_rate= 0.0 , drop_connect_rate=0.0)
  #model_e = EfficientNet_canonized.from_name('efficientnet-b0', dropout_rate= 0.0 , drop_connect_rate=0.0 ,image_size=[224,224])
  model_e.set_swish( memory_efficient=False)
  
  model_e.copyfromefficientnet( model, lrp_params = lrp_params_def1 , lrp_layer2method = lrp_layer2method )
  
  for i, (nm,mod) in enumerate( model_e.named_modules()):
    print(i,nm,mod)
  #exit()

  data_transform = transforms.Compose([

          transforms.Resize(224),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) # if you do five crop, then you must change this part here, as it cannot be applied to 4 tensors

      ])

  root_dir='/home/binder/entwurf9/data/imagenetvalimg/'

  dset= dataset_imagenetvalpart_nolabels(root_dir, maxnum=1, transform=data_transform, skip= skip)
  dataloader =  torch.utils.data.DataLoader(dset, batch_size=1, shuffle=False) #, num_workers=1) 

  test_model5(dataloader = dataloader,  model = model_e, device = device )
  
if __name__ == '__main__':
  run3(skip = 2)  # 16,20,24,21 # 68,81,90,30,99,126 (93) # 81 is lady and dog

