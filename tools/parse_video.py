import sys
import argparse
from torchvision import models
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision.transforms as T


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--video_path', type=str, default=None)
	parser.add_argument('--frame_step', type=int, default=5)
	parser.add_argument('--dilate_kernel_size', type=int, default=5)
	parser.add_argument('--dilate_iterations', type=int, default=2)
	parser.add_argument('--fcn', action='store_true')
	parser.add_argument('--dlab', action='store_true')

	return parser.parse_args()


def decode_segmap(image, nc=21):
    label_colors = np.array([(255, 255, 255),  # 0=background
                             # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                             (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
                             # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                             (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
                             # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                             (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (0, 0, 0),
                             # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                             (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb


def _main():
	args = parse_args().video_path
	if args.fcn:
		mdoel = models.segmentation.fcn_resnet101(pretrained=True).eval()
	if args.dlab:
		mdoel = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

	trf = T.Compose([T.ToPILImage(),
		             T.Resize(256),
		             T.ToTensor(),
		             T.Normalize(mean=[0.485, 0.456, 0.406],
		                         std=[0.229, 0.224, 0.225])])
	cap = cv2.VideoCapture(argv[0])
	frame_number = 0
	C = 0
	while (cap.isOpened()):
		ret, img = cap.read()
		C += 1
		if C != args.frame_step:
			continue
		C=0
		if img is None:
			break
		inp = trf(img).unsqueeze(0)
		out = model(inp)['out']
		om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()

		mask = decode_segmap(om)
		kernel = np.ones((args.dilate_kernel_size, args.dilate_kernel_size), np.uint8)
		mask = 255 - cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
		mask = cv2.dilate(mask, kernel, iterations=args.dilate_iterations)

		img = cv2.resize(img, (mask.shape[1], mask.shape[0]))
		frame_name = '0'*(5 - len(str(frame_number))) + str(frame_number)
		cv2.imwrite('./masks/' + frame_name + '.png', mask)
		cv2.imwrite('./frames/' + frame_name + '.png', img)
		print(f'Write {frame_number}')
		frame_number += 1

	cap.release()


if __name__ == '__main__':
    _main()
