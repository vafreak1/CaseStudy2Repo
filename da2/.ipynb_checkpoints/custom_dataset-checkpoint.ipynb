{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "f1c4b4e0",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "from torch.utils.data import Dataset\n",
    "\n",
    "class CustomDataset(Dataset):\n",
    "    def __init__(self, trainset, transform, augmentations):\n",
    "        self.trainset = trainset\n",
    "        self.transform = transform\n",
    "        self.augmentations = augmentations\n",
    "\n",
    "    def __len__(self):\n",
    "        return len(self.trainset)\n",
    "\n",
    "    def __getitem__(self, index):\n",
    "        data, target = self.trainset[index]\n",
    "\n",
    "        # apply each transformation jointly to each input\n",
    "        if self.transform:\n",
    "            data = self.transform(data)\n",
    "\n",
    "        # apply each augmentation separately to each input\n",
    "        if self.augmentations:\n",
    "            for augmentation in self.augmentations:\n",
    "                data = augmentation(data)\n",
    "\n",
    "\n",
    "        return data, target\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
