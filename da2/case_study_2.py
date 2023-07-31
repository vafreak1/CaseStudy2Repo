import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import csv
import math
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
import pytorch_lightning as pl
import glob
import torchvision.transforms.functional as TF
import random


# First, we slice the train images into patches of 15 x 15 pixels with the ground truth in the middle. This seemed reasonable since 1 pixel is the equivalent of about 20 meters.


# Used to fetch and save files in load_data()
def ndigit(n, x):
    x = str(x)
    while(len(x) < n):
        x = "0" + x
    return x


# Function to load images, enrich them with moisture and vegetation index (i.e. increase channels from 10 to 12),
# extract ground truths from the masks, pad the images to work with ground truths close to the edges,
# slice images into 15 x 15 patches and save them.
def load_data(res, files = 20):
    j = 0
    path = ["02", "train"]
    res = int((res-1)/2)
    nan_values = 0

    # Load images and masks
    for p in path:
        for f in range(files):
            image = np.load(f"images_{p}/images/image_{ndigit(3, f)}.npy")
            mask = np.load(f"masks_{p}/masks/mask_{ndigit(3, f)}.npy")
            
            # In anticipation of toTensor() in transforms later which expects an array of H x W x C and converts it into C x H x W.
            image = np.transpose(image, (1,2,0))
            mask = np.transpose(mask, (1,2,0))
            
            nan_values_before = (np.count_nonzero(np.isnan(image)))
            
            # Extract spectral bands for calculating vegetation index
            channel8 = image[:, :, 6]
            channel4 = image[:, :, 2]

            # Calculate the vegetation index with small epsilon in order to prevent dividing through zero (which results in NaN values)
            vegetation_array = np.divide((np.subtract(channel8, channel4)), np.add(np.add(channel8, channel4), 1e-6))
            
            nan_values_vegetation = (np.count_nonzero(np.isnan(vegetation_array)))
            
            if(nan_values_vegetation > 0): 
                print("picture",f"images_{p}/images/image_{ndigit(3, f)}.npy had", nan_values_before, "before vegetation index")
                print("picture",f"images_{p}/images/image_{ndigit(3, f)}.npy has", nan_values_vegetation, "nan_values after adding vegetation")
            
            
            vegetation_array = np.nan_to_num(vegetation_array, nan=0.0)

            # Add vegetation index to the image as eleventh channel
            image_veg = np.concatenate((image, vegetation_array[:, :, np.newaxis]), axis=2)

            nan_values_before = 0
            
            # Extract spectral bands for calculating moisture index
            channel8a = image[:, :, 7]
            channel11 = image[:, :, 8]

            # Calculate the moisture index with small epsilon in order to prevent dividing through zero
            moisture_array = np.divide((np.subtract(channel8a, channel11)), np.add(np.add(channel8a, channel11), 1e-6))
            
            nan_values_moisture = (np.count_nonzero(np.isnan(moisture_array)))
            
            if(nan_values_moisture > 0): 
                print("picture",f"images_{p}/images/image_{ndigit(3, f)}.npy had", nan_values_before, "before moisture index")
                print("picture",f"images_{p}/images/image_{ndigit(3, f)}.npy has", nan_values_moisture, "nan_values after adding moisture")
            
            # Add moisture index to the image as twelfth channel
            image_veg_mois = np.concatenate((image_veg,moisture_array[:,:, np.newaxis]), axis = 2)

            nan_values_pic = np.count_nonzero(np.isnan(image_veg_mois))
            nan_values += nan_values_pic

            # Add padding to every image and mask edge in case there are ground truths which are too close to an edge
            padded_image = np.pad(image_veg_mois, ((res+1, res+1), (res+1, res+1), (0,0)), mode='constant')
            padded_mask = np.pad(mask, ((res+1, res+1), (res+1, res+1), (0,0)), mode='constant')

            # Extract ground truths
            ground_truths_pos = np.array(np.where(padded_mask != 0)).T
            
            # Slice and save patches around each ground truth
            for i in ground_truths_pos: 
                patch = (padded_image[i[0]-res : i[0]+res+1, i[1]-res : i[1]+res+1, :], padded_mask[i[0], i[1], 0])
                np.save(f"patches/train/patch_{p}_{ndigit(3, f)}_{ndigit(5, j)}.npy", np.array(patch, dtype="object"))                                 
                j += 1
    print("Added Vegetation (B8-B4)/(B8+B4)")
    print("Added Moisture (B8A-B11)/(B8A+B11)")
    print("Patched the pictures")
    print("NaN values:", nan_values)


res = 15 # Set resolution
load_data(res)


### Load, check and split the data

# Load patches into dataset
directory = 'patches/train'
file_paths = glob.glob(directory + '/*.npy')
dataset = [np.load(file_path, allow_pickle=True) for file_path in file_paths]

# Check some metrics on the dataset
l_t = len(dataset)
X,y = dataset[0]
X_s = X.shape
y_s = y.shape
print(f"Trainset contains {l_t} samples where each input shape is {X_s} and target shape is {y_s}.")

# Calculate the sizes of the training set and validation set
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# Split trainset into trainset and valset
trainset, valset = random_split(dataset, [train_size, val_size])
print(len(trainset), len(valset))

# Stack data for all train images in one array
inputs = np.stack([data[0] for data in trainset], axis=0)
inputs.shape


# Compute the mean, standard deviation and other metrics for each channel from all pictures 
# along heigth and width
channel_means = np.mean(inputs, axis=(0, 1, 2), keepdims=False)
channel_stds = np.std(inputs, axis=(0, 1, 2), keepdims=False)
channel_meds = np.median(inputs, axis=(0, 1, 2), keepdims=False)
channel_mins = np.min(inputs, axis=(0, 1, 2), keepdims=False)
channel_maxs = np.max(inputs, axis=(0, 1, 2), keepdims=False)
channel_10_quants = np.quantile(inputs, 0.1, axis=(0, 1, 2), keepdims=False)
channel_90_quants = np.quantile(inputs, 0.9, axis=(0, 1, 2), keepdims=False)

print(f"{len(channel_means)} means and {len(channel_stds)} standard deviations were computed.")

for i in range(inputs.shape[3]):
  print(f"Channel {i+1}")
  print(f"mean = {channel_means[i]}")
  print(f"std = {channel_stds[i]}")
  print(f"median = {channel_meds[i]}")
  print(f"min = {channel_mins[i]}")
  print(f"max = {channel_maxs[i]}")
  print(f"10 percent quantile = {channel_10_quants[i]}")
  print(f"90 percent quantile = {channel_90_quants[i]}")
  print()


### Apply transformations and augmentations


# Create custom dataset class in order to transform dataset and apply data augmentation
class CustomDataset(Dataset):
    def __init__(self, dataset, transform, augmentations):
        self.dataset = dataset
        self.transform = transform
        self.augment = augmentations

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, target = self.dataset[index]

        # Apply transformations
        if self.transform:
            data = self.transform(data)

        # Apply augmentations
        if self.augment:
           data = self.augment(data)

        return data, target


# Custom rotation transformation from the documentation in order to rotate at given angles,
# not select from range of angles.
class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, image):
        angle = random.choice(self.angles)
        return TF.rotate(image, angle)

# Custom elastic transformation which adds randomness. Originally, transforms.ElasticTransform transforms
# every image, but now only at given probability.
class RandomElasticTransform:
    def __init__(self, probability, alpha, sigma):
        self.probability = probability
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, image):
        if np.random.rand() < self.probability:
          elastic_transformer = transforms.ElasticTransform(self.alpha, self.sigma)
          return elastic_transformer(image)
        else:
          return image

# Custom normalization class. Only the first 10 channels need to be normalized, the last two are already in normalized form
# since their creation. The regular transforms.Normalize will throw an error, however, if less means/stds are passed than
# there are channels.
class MyNormalization:
    def __init__(self, means, stds):
        self.means = means
        self.stds = stds
        self.channels = [i for i in range(len(means))] # Channels to normalize

    def __call__(self, image):
        for i in self.channels:
            image[i, :, :] = (image[i, :, :] - self.means[i]) / self.stds[i]
        return image


# Define transformations
transform = transforms.Compose(
    [transforms.ToTensor(), # If input is 3D array then toTensor() switches dimensions from H x W x C to C x H x W
     transforms.ConvertImageDtype(torch.float64),
     #transforms.Lambda(lambda x : x / 3000),
     #transforms.Lambda(lambda x : torch.where(x > 1, 1, x)), # fix pixel values between 0 and 1
     #transforms.Normalize(mean=channel_means, std=channel_stds),
     MyNormalization(means=channel_means[0:10], # Apply normalization with means and stds of trainset
                          stds=channel_stds[0:10])
     ])
# Note: MyNormalization provided somewhat better performance than division by 3000, clipping between range of [0,1]
# and then using regular normalization.

# Define data augmentations
augmentations = transforms.Compose(
    [MyRotationTransform(angles=[0, 90, 180, 270, 0]), # Rotate input randomly at given angles
     transforms.RandomAffine(degrees=0, translate=(0.1,0.1), scale=(0.9,1.1)), # shift in both directions along 0.1 * height on y-axis and 0.1 * width on x-axis
                                                                               # scale in range 0.9 <= scale <= 1.1
     #transforms.ElasticTransform(alpha=50.0, sigma=3), # Displace pixels
     transforms.RandomHorizontalFlip(), # default p = 0.5
     transforms.RandomVerticalFlip()
    ])
# Note: tried various version and parameters of augmentations.


# Create the custom trainset and valset. In our final model, no augmentations were used for training as it
# proved to be most successful in most tries. Rotation was sometimes added, which was ok.
trainset_transformed = CustomDataset(trainset, transform=transform, augmentations=None)
valset_transformed = CustomDataset(valset, transform=transform, augmentations=None)

# Create data loaders for transformed training set and validation set
batch_size = 64
trainloader = DataLoader(trainset_transformed, batch_size=batch_size, shuffle=True, num_workers=2)
validloader = DataLoader(valset_transformed, batch_size=batch_size, shuffle=False, num_workers=2)


### Define and train the model


class MyCNNModel(pl.LightningModule):

    def __init__(self, *layers, classes=None):
        super().__init__()

        self.lr = 0.01  # Assign the learning rate
        self.classes = classes

        self.layers = nn.Sequential(*layers)  # Create a sequential model

    # Forward pass
    def forward(self, X):
        return self.layers(X)

    def predict(self, X):
        with torch.no_grad():
            y_hat = self(X).argmax(1)
        #if self.classes is not None:  # There are no classes here
        #    y_hat = [self.classes[i] for i in y_hat]
        return y_hat

    def training_step(self, batch, batch_idx, log_prefix='train'):
        X, y = batch
        y_hat = self(X) # calls forward(x)
        y_hat = y_hat.flatten()
        loss = nn.L1Loss() # Compute MAE
        loss = loss(y_hat, y)
        self.log(f"{log_prefix}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            return self.training_step(batch, batch_idx, log_prefix='valid')

    # Tried various optimizers, but Moritz's recommendations proved to be right :)
    def configure_optimizers(self):
        # Adam with Weight Decay = 0.01
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        #optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # ReduceLROnPlateau reduces the learning rate by 0.5 if the valid_loss has not decreased within the last 2 epochs.
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True),
            # Updates the scheduler after every epoch.
            "interval": "epoch",
            # Updates the learning rate with frequency 1
            "frequency": 1,
            # Metric to monitor for scheduler
            "monitor": "valid_loss",
            # Enforce that the value specified in 'monitor' is available when the scheduler is updated,
            # thus stopping training if not found.
            "strict": True,
            # No custom logged name
            "name": None,
        }
        return {"optimizer": optimizer, 'lr_scheduler': scheduler}


# We decided to follow Lang et al. (2019) using depthwise separable convolutions in order to
# be able to increase depth of the model while reducing trainable parameters hoping for equal/better
# performance (however, didn't check performance against regular convolutions).
# Own modifications are marked with comment "[M]".

# Implements entry layer to SepConv2d, see Lang et al. (2019), p. 6
# This is our own implementation as no source code has been published for the paper (yet)
class MyEntryLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.out_channels = out_channels

        # Called for residual learning in forward pass
        self.proj_out = nn.Conv2d(in_channels, out_channels[len(out_channels)-1], (1,1))

        # Appends a new entry block for each new out_channel to ModuleList()
        self.entry_blocks = nn.ModuleList()
        for i in range(len(out_channels)):
            self.entry_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels[i], (1, 1)), # In hindsight, could have been implemented as SepConv as well
                nn.BatchNorm2d(out_channels[i]),
                nn.PReLU(), # [M] Sets gradient for negative inputs as learnable parameter for the model
                #nn.PReLU(out_channels[i]), # [M] Sets gradient for negative inputs as learnable parameter for each channel individually
                nn.Dropout2d(p=0.5) # [M] Supported us in our constant struggle against overfitting
            ))
            in_channels = out_channels[i]  # Update in_channels for next iteration

    def forward(self, x):
        x_entry = x
        # Passes input through each block in entry_blocks as often as there are out_channels
        for i in range(len(self.out_channels)):
            x_entry = self.entry_blocks[i](x_entry)
        x = self.proj_out(x)
        return (x + x_entry) # Uses residual learning

# Implements SepConv2D, see Lang et al. (2019), p. 6
# This is our own implementation as no source code has been published for the paper (yet)
class MySepConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, **kwargs):
        super().__init__()

        # Called for residual learning in forward pass
        if in_channels == out_channels:
            self.proj_out = nn.Identity()
        else:
            self.proj_out = nn.Conv2d(in_channels, out_channels, (1,1), **kwargs)

        # Separable convolution block, divided in depthwise and pointwise separable convolution
        self.sep_conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel, groups=in_channels, **kwargs), # Depthwise SepConv
            nn.Conv2d(in_channels, out_channels, (1,1), **kwargs), # Pointwise SepConv
            nn.BatchNorm2d(out_channels),
            nn.PReLU(), # [M]
            nn.Dropout2d(p=0.5), # [M] Supported us in our constant struggle against overfitting
            nn.Conv2d(in_channels, in_channels, kernel, groups=in_channels, **kwargs), # Performs second SepConv, see Lang et al. (2019), p. 6
            nn.Conv2d(in_channels, out_channels, (1,1), **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(), # [M]
            nn.Dropout2d(p=0.5), # [M] Supported us in our constant struggle against overfitting
        )

    def forward(self, x):
        x_sep_conv = self.sep_conv_block(x)
        x = self.proj_out(x)
        return (x + x_sep_conv) # Uses residual learning

# Implementation of layers
tree_model = MyCNNModel(
    MyEntryLayer(12, [32, 64, 128]), # Increase number of channels to 128 (not 728 as in the paper)
    MySepConvLayer(128, 128, (5,5), padding='same'),
    MySepConvLayer(128, 128, (5,5), padding='same'),
    MySepConvLayer(128, 128, (5,5), padding='same'),
    nn.MaxPool2d(2,1),
    MySepConvLayer(128, 128, (5,5), padding='same'),
    MySepConvLayer(128, 128, (5,5), padding='same'),
    MySepConvLayer(128, 128, (5,5), padding='same'),
    MySepConvLayer(128, 128, (5,5), padding='same'),
    MySepConvLayer(128, 128, (5,5), padding='same'),
    MySepConvLayer(128, 128, (5,5), padding='same'),
    nn.MaxPool2d(2,2),
    MySepConvLayer(128, 128, (5,5), padding='same'),
    MySepConvLayer(128, 128, (5,5), padding='same'),
    MySepConvLayer(128, 128, (5,5), padding='same'),
    nn.AdaptiveMaxPool2d(1),
    nn.Flatten(1),
    nn.Linear(128, 1)
)

# Trainer class
# 30 Epochs were used and loss was still improving, however, it was too risky to increase number of epochs
# since Google Colab would occasionally withdraw GPU during training and delete runtime if used for too long.
from pytorch_lightning.callbacks import RichProgressBar, RichModelSummary
trainer1 = pl.Trainer(devices=1, accelerator="cpu", precision='64', max_epochs=30,
                      callbacks=[RichProgressBar(refresh_rate=1),
                                 RichModelSummary(3),
                                ])

# Train the model
trainer1.fit(tree_model, trainloader, validloader)


# Save the model
torch.save(tree_model, 'model07_1.zip')


### Evaluate performance of model 

# Printing of the predictions of the first batch and the corresponding ground truths to get an idea of how well the model actually predicts the data in a real batch


tree_model.eval()
tree_model = tree_model.float()
batch = next(iter(trainloader))
inputs = batch[0]
inputs = inputs.float()

print("Input shape:", inputs.shape)


with torch.no_grad():
    predictions = tree_model(inputs).flatten()

true_heights = batch[1]
print("Targets:", batch[1])
print("Target shape:", batch[1].shape)
print("Predictions:", predictions)
print("Prediction shape:", predictions.shape)
# expected batch size number of predictions for height !


# Here we split the predictions and ground truths into size classes to see whether the models performance differs based on the actual height of a ground truth. We print both the mae and mse per size class and the overall mse and mae. Numbers might differ from performance of all data since not all data points fall withing the range of 0-40 meters, that the size classes cover. 


tree_model.eval()
tree_model = tree_model.float()

num_classes = 10  # Number of size classes
class_intervals = 4  # Interval between size classes
# 10 classes of each 4 m 
class_thresholds = [i * class_intervals for i in range(1, num_classes+1)]

mse_total = [0.0] * num_classes
mae_total = [0.0] * num_classes
class_counts = [0] * num_classes

true_heights_total = 0.0
predictions_total = 0.0

with torch.no_grad():
    progress_bar = tqdm(validloader, desc="Evaluation")
    for batch in progress_bar:
        inputs, true_heights = batch[0].float(), batch[1].float()
        batch_size = inputs.size(0)

        predictions = tree_model(inputs)
        true_heights = true_heights.view(-1, 1)

        mse = F.mse_loss(predictions, true_heights, reduction='none').squeeze()
        mae = F.l1_loss(predictions, true_heights, reduction='none').squeeze()

        true_heights_total += true_heights.sum().item()
        predictions_total += predictions.sum().item()

        for i, threshold in enumerate(class_thresholds):
            indices = (true_heights <= threshold).squeeze(1)
            mse_total[i] += mse[indices].sum().item()
            mae_total[i] += mae[indices].sum().item()
            class_counts[i] += indices.sum().item()

        progress_bar.set_postfix({'Total MSE': mse_total[0] / class_counts[0], 'Total MAE': mae_total[0] / class_counts[0]})

mse_class_avg = [mse_total[i] / class_counts[i] if class_counts[i] != 0 else 0.0 for i in range(num_classes)]
mae_class_avg = [mae_total[i] / class_counts[i] if class_counts[i] != 0 else 0.0 for i in range(num_classes)]
average_true_height = true_heights_total / len(validloader.dataset)
average_prediction = predictions_total / len(validloader.dataset)
# Calculate overall MSE and MAE
overall_mse = sum(mse_total) / sum(class_counts) if sum(class_counts) != 0 else 0.0
overall_mae = sum(mae_total) / sum(class_counts) if sum(class_counts) != 0 else 0.0


# Print the evaluation metrics for each size class
for i, threshold in enumerate(class_thresholds):
    print(f"Size Class {i+1}:")
    print(f"MSE: {mse_class_avg[i]}")
    print(f"MAE: {mae_class_avg[i]}")

# Print the overall evaluation metrics
print("Average True Height:", average_true_height)
print("Average Prediction:", average_prediction)
print("Average MSE: ", overall_mse )
print("Average MAE:", overall_mae )


### Sliding window predictions

res = 15
res = int((res-1)/2)
model = torch.load("model07_1.zip") #Load the saved model
model.eval() #Put model in evaluation mode
for i in range(10): #Iterate over the 10 test images
    
    #Load the image and change the order HxWxC
    image = np.load(f"test_images/image_00{i}.npy")
    image = np.transpose(image, (1,2,0))
     
    nan_values_before = (np.count_nonzero(np.isnan(image)))
            
    channel8 = image[:, :, 6]
    channel4 = image[:, :, 2]
    channels = image.shape
    width = image[0].shape[0]
    height = image[0].shape[1]

    # add the vegetation array 
    vegetation_array = np.divide((np.subtract(channel8, channel4)), np.add(np.add(channel8, channel4), 1e-6))
            
    # Check for any nan values
    nan_values_vegetation = (np.count_nonzero(np.isnan(vegetation_array)))
            
    if(nan_values_vegetation > 0): 
        print("picture",f"test_images/image_00{i}.npy had", nan_values_before, "before vegetation index")
        print("picture",f"test_images/image_00{i}.npy has", nan_values_vegetation, "nan_values after adding vegetation")
                 
    vegetation_array = np.nan_to_num(vegetation_array, nan=0.0)
    image_transformed = np.concatenate((image, vegetation_array[:, :, np.newaxis]), axis=2)
       
    image = image_transformed
    nan_values_before = 0
    # add moisture index
    channel8a = image[:, :, 7]
    channel11 = image[:, :, 8]
    
    # add the moisture array 
    moisture_array = np.divide((np.subtract(channel8a, channel11)), np.add(np.add(channel8a, channel11), 1e-6))
            
    # Check for any nan values
    nan_values_moisture = (np.count_nonzero(np.isnan(moisture_array)))
            
    if(nan_values_moisture > 0): 
        print("picture",f"test_images/image_00{i}.npy had", nan_values_before, "before moisture index")
        print("picture",f"test_images/image_00{i}.npy has", nan_values_moisture, "nan_values after adding moisture")
                   
    image_transformed = np.concatenate((image,moisture_array[:,:, np.newaxis]), axis = 2)
    image = image_transformed
    
    # Add padding to every image edge in case there are ground truths which are too close to an edge
    padded_image = np.pad(image, ((res+1, res+1), (res+1, res+1), (0,0)), mode='constant')
    # Normalize the image with the means and stds from the trainset
    padded_image[:,:,:10] = (padded_image[:,:,:10] - channel_means[:10]) / channel_stds[:10]
    
    pred = np.zeros((1024, 1024))
    
    # iterate over rows and columns of the picture
    for p in range(res+1, 1024+res+1):
        for q in range(res+1, 1024+res+1):
            patch = padded_image[p-res : p+res+1, q-res : q+res+1, :] #Cut out the patch from the picture
            patch = torch.from_numpy(patch).float() #Convert to float tensor
            patch = torch.unsqueeze(patch, 0).permute(0,3,1,2) #Add extra empty dimension(batch size = 1) and change the order of the dimensions
            pred[p-(res+1), q-(res+1)] = model(patch) #Calculate the prediction for the patch and save it
            
    pred = np.expand_dims(pred, axis=0) #Add empty dimension in front (1x1024x1024)
    np.save(f"test_images/prediction_00{i}.npy", pred) #Save to file
    

