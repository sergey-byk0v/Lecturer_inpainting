import cv2
import sys
import os



def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--image_folder', type=str, default='./Inpaint_Res/inpaint_res')
	parser.add_argument('--video_name', type=str, default='video.avi')
	parser.add_argument('--fps', type=int, default=5)

	return parser.parse_args()


def _main():
	args = parse_args()
	images = [img for img in os.listdir(args.image_folder)]
	frame = cv2.imread(os.path.join(image_folder, images[0]))
	height, width, layers = frame.shape

	video = cv2.VideoWriter(args.video_name, 0, args.fps, (width,height))

	for image in sorted(images):
		video.write(cv2.imread(os.path.join(image_folder, image)))

	cv2.destroyAllWindows()
	video.release()


if __name__ == '__main__':
    _main()
