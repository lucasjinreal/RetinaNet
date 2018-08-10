import time
import os
import copy
import argparse
import pdb
import collections
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision

import model
from anchors import Anchors
import losses
from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

import coco_eval



def main(args=None):

	parser = argparse.ArgumentParser(description="Evaluate network parameters pf RetinaNet")

	parser.add_argument("--dataset")
	parser.add_argument("--coco_path")
	parser.add_argument("--saved_weights")

	parser = parser.parse_args(args)


	if parser.dataset=="coco":

		dataset_val = CocoDataset(parser.coco_path, set_name='val2017', transform=transforms.Compose([Normalizer(), Resizer()]))

	#Load the network
	retinanet = torch.load(parser.saved_weights)
	retinanet.eval()


	#Evaluate the netwoek on coco
	coco_eval.evaluate_coco(dataset_val, retinanet)


if __name__=="__main__":

	main()


