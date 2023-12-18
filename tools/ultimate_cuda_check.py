import torch

print(f'{torch.__version__=}')
print(f'{torch.version.cuda=}')
print(f'{torch.backends.cudnn.enabled=}')
print(f'{torch.cuda.is_available()=}')
print(f'{torch.cuda.device_count()=}')
print(f'{torch.cuda.get_device_name(0)=}')
print(f'{torch.cuda.get_device_capability(0)=}')
print(f'{torch.cuda.get_device_properties(0).total_memory=}')


try:
    import torchvision
except ImportError:
    print('torchvision not installed')
else:
    print(f'{torchvision.__version__=}')



