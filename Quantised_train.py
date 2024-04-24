import torch
from model import UBDmDN
import torch.utils.data as td
import os
from PIL import Image
import numpy as np
import torchvision as tv
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from model import SplitLoss

root_dir="/Users/parasjoshi/Desktop/IDEX/GitHub/IDEX/dataset"
class NoisyThermalDataset(td.Dataset):

    def __init__(self, root_dir, mode='train', image_size=(180, 180), sigma=30):
        super(NoisyThermalDataset, self).__init__()
        self.mode = mode
        self.image_size = image_size
        self.sigma = sigma
        self.images_dir_clean = os.path.join(os.path.join(root_dir, mode),"clean")
        self.images_dir_noisy=os.path.join(os.path.join(root_dir, mode),"noisy")

        self.files_clean = os.listdir(self.images_dir_clean)
        self.files_noisy=os.listdir(self.images_dir_noisy)

    def __len__(self):
        return len(self.files_clean)

    def __repr__(self):
        return "NoisyBSDSDataset(mode={}, image_size={}, sigma={})". \
            format(self.mode, self.image_size, self.sigma)

    def __getitem__(self, idx):
        img_path_clean = os.path.join(self.images_dir_clean, self.files_clean[idx])
        img_path_noisy = os.path.join(self.images_dir_noisy, self.files_noisy[idx])
        clean = Image.open(img_path_clean).convert('L')
        noisy = Image.open(img_path_noisy).convert('L')
        # random crop
        i = np.random.randint(clean.size[0] - self.image_size[0])
        j = np.random.randint(clean.size[1] - self.image_size[1])

        p = tv.transforms.Resize((180,180))
        clean = p(clean)
        noisy= p(noisy)
        #clean = clean.crop([i, j, i+self.image_size[0], j+self.image_size[1]])
        #noisy=noisy.crop([i,j,i+self.image_size[0], j+self.image_size[1]])


        transform_to_tensor = tv.transforms.Compose([
            # convert it to a tensor
            tv.transforms.ToTensor(),
            # normalize it to the range [−1, 1]
        ])

        transform_normalize=tv.transforms.Compose([
            tv.transforms.Normalize((0.5),(0.5))
        ])

        clean = transform_to_tensor(clean)
        noisy = transform_to_tensor(noisy)

        #noisy=noisy-clean
        #noisy=torch.clamp(noisy,0,255)
        clean=transform_normalize(clean)
        noisy=transform_normalize(noisy)

        #noisy = (noisy + 2 / 255 * self.sigma * torch.randn(clean.shape)) - clean

        #noisy=noisy-clean

        noisy=noisy.to('mps')
        clean=clean.to('mps')

        return noisy, clean

def train_one_epoch(epoch_index, tb_writer,training_loader,optimizer,model,loss_fn):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

        if i % 10 == 9:
            last_loss = running_loss / 50 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

def run():
    device='mps'

    model=UBDmDN().to(device)

    loss_fun= torch.nn.L1Loss()

    training_loader = torch.utils.data.DataLoader(NoisyThermalDataset(root_dir), batch_size=20, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(NoisyThermalDataset(root_dir,'test'), batch_size=10, shuffle=False)

    adam=torch.optim.Adam(model.parameters(),lr=0.001)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/ubdmdn_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 100

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer,training_loader,adam,model,loss_fun)

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = loss_fun(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch_number + 1)
        #writer.add_histogram('kernel weights',model.conv[0].weight,epoch_number+1)
        #writer.add_histogram('kernel weights', model.conv[1].weight, epoch_number + 1)
        #writer.add_histogram('kernel weights', model.conv[2].weight, epoch_number + 1)
        #writer.add_histogram('kernel weights', model.conv[3].weight, epoch_number + 1)
        #writer.add_histogram('kernel weights', model.conv[4].weight, epoch_number + 1)
        writer.add_histogram('kernel weights', model.conv[5].weight, epoch_number + 1)
        #writer.add_histogram('kernel weights', model.conv[6].weight, epoch_number + 1)
        #writer.add_histogram('kernel weights', model.conv[7].weight, epoch_number + 1)
        #writer.add_histogram('kernel weights', model.conv[8].weight, epoch_number + 1)
        #writer.add_histogram('kernel weights', model.conv[9].weight, epoch_number + 1)
        #writer.add_histogram('kernel weights', model.conv[10].weight, epoch_number + 1)
        #writer.add_histogram('kernel weights', model.conv[11].weight, epoch_number + 1)

        #writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1

        torch.save(model.state_dict(),"/Users/parasjoshi/Desktop/IDEX/GitHub/IDEX/ubdmdn/trained/custom_train_logsoftmax.pt")

run()

