import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse
import tqdm 

import sys
import cv2
import model
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from tensorboardX import SummaryWriter
from dataloader import scotty_dataset, UnNormalizer, Normalizer, Resizer

assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))



def color_id(label=None):

	color_info={0: [255,0,0], 1: [0,255,0], 2: [0,255,0], 3: [0,0,255], 4: [0,0,255], 5: [0,0,255], 6: [0,0,255], 7: [0,0,255], 
	8: [0,0,255], 9: [255,0,0]}

	return color_info.get(label,"Invalid ID")

class save_video(object):

	def __init__(self, vid_name):
		# Define the codec and create VideoWriter object
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		self.out = cv2.VideoWriter(vid_name, fourcc, 4.0, (1056, 544))

	def write(self, frame):
		# write the flipped frame
		self.out.write(frame)

	def close(self):
		self.out.release()


def main(args=None):

	parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
	parser.add_argument('--save_type', help='Saved model type is state_dict or model file')
	parser.add_argument('--model', help='Path to model (.pt) file.')
	parser.add_argument('--folder', help='Path to the evaluation images folder')
	parser.add_argument('--rec', type=bool)
	parser.add_argument('--video_file', help= "Name of the file to be saved")


	parser = parser.parse_args(args)
	if parser.rec:
		assert parser.video_file is not None
	dataset_val = scotty_dataset(parser.folder, transform=transforms.Compose([Normalizer(), Resizer()]))
	dataset_val_viz = scotty_dataset(parser.folder, transform=transforms.Compose([Resizer()]))
	dataloader_val = DataLoader(dataset_val, num_workers=1, shuffle=False)

	#retinanet = torch.load(parser.model)
	#retinanet = model.resnet18(num_classes=dataset_val.num_classes(),)
	#retinanet.load_state_dict(torch.load(parser.model))
	retinanet = torch.load(parser.model)

	use_gpu = True

	if use_gpu:
		retinanet = retinanet.cuda()

	retinanet.eval()

	unnormalize = UnNormalizer()

	def draw_caption(image, box, caption):

		b = np.array(box).astype(int)
		cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
		cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

	#initialize video wroter object
	if parser.rec:
		print("..Recording Video..")
		record = save_video(parser.video_file)

	pbar = tqdm.tqdm(range(len(dataloader_val)))

	alpha=0.55

	dummy_input = Variable(torch.rand(1, 3, 224, 224))

	with SummaryWriter(comment='resnet18') as w:
	    model = torchvision.models.resnet18()
	    w.add_graph(model, (dummy_input, ))



#-------------------------------------------------Initiate Training Loop----------------------------------------------------#
	for idx, data in enumerate(dataloader_val):

		with torch.no_grad():

			torch.cuda.synchronize()
			st = time.time()
			scores, classification, transformed_anchors = retinanet(data['img'].permute(0,3,1,2).float().cuda())
			#print ("Image shape: {} ".format(data['img'].permute(0,3,1,2).shape))
			pbar.write('Elapsed time: {}'.format(time.time()-st))
			torch.cuda.synchronize()
			idxs = np.where(scores>0.5)
			img = np.array(unnormalize(torch.squeeze(data["img"]).permute(2,1,0)).permute(2,1,0))
			img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2RGB)
			img = (255*img).astype(np.uint8)
			img_dup = img.copy()

			for j in range(idxs[0].shape[0]):
				bbox = transformed_anchors[idxs[0][j], :]
				x1 = int(bbox[0])
				y1 = int(bbox[1])
				x2 = int(bbox[2])
				y2 = int(bbox[3])
				label_name = dataset_val.labels[int(classification[idxs[0][j]])]
				color=color_id(int(classification[idxs[0][j]]))
				draw_caption(img, (x1, y1, x2, y2), label_name)
				cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
				#print(label_name)

			pbar.update()
			pbar.set_description("Images Processed : {}/{}".format(idx, len(dataloader_val)))
			pbar.set_postfix("")

			cv2.addWeighted(img, alpha, img_dup, 1 - alpha, 0, img)

			if parser.rec:
				record.write(img)

			cv2.imshow('img', img)
			cv2.waitKey(0)

	if parser.rec:
		record.close()
			



if __name__ == '__main__':

	# with torch.cuda.device(1):
	main()