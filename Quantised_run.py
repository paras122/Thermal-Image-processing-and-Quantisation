import numpy as np
import torch
from model import UBDmDN
from PIL import Image
import torchvision as tv
import matplotlib.pyplot as plt


img=Image.open("/Users/parasjoshi/Desktop/IDEX/GitHub/IDEX/dataset/train/noisy/550.png").convert('L')

img_clean=Image.open("/Users/parasjoshi/Desktop/IDEX/GitHub/IDEX/dataset/train/clean/550.png").convert('L')

transform=tv.transforms.Compose([tv.transforms.ToTensor(),tv.transforms.Normalize((0.5), (0.5))])

device = 'mps'

model=UBDmDN().to(device)

model.load_state_dict(torch.load("/Users/parasjoshi/Desktop/IDEX/GitHub/IDEX/ubdmdn/trained/custom_train_logsoftmax.pt",map_location=device))

transformed_img=transform(img).to(device)
transformed_img_clean=transform(img_clean).to(device)

with torch.no_grad():
    output=model(transformed_img.unsqueeze(0))
plt.imshow(output[0].cpu().permute(1,2,0) + 1.0 /2.0,cmap='gray')
plt.show()
plt.imshow(img_clean,cmap='gray')
plt.show()
plt.imshow((transformed_img_clean-output[0]).cpu().permute(1,2,0) + 1.0/ 2.0,cmap='gray')
plt.show()
plt.imshow(img,cmap='gray')
plt.show()
