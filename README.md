
# Lecturer inpainting
In this project [Deep-Flow-Guided-Video-Inpainting](https://github.com/nbei/Deep-Flow-Guided-Video-Inpainting) was used.

## Result
![](./gif_results/result.gif)

## Original
![](./gif_results/original.gif)

## Install & Requirements

**To Install python packages**
```
pip install -r requirements.txt
```
**To Install flownet2 modules**
```
bash install_scripts.sh
```
## Usage
* First of all download pre-trained models from [here](https://drive.google.com/drive/folders/1a2FrHIQGExJTHXxSIibZOGMukNrypr_g?usp=sharing) and put them into `./pretrained_models`. 
* To prepare frames and masks for inpainting execute this code:
```
python tools/parse_video.py --video_path ./path/example.mp4 --frame_step 5 --dilate_kernel_size 5 --dilate_iterations 2 --fcn
```                                                                               
`--video_path` - path to source video file  
`--frame_step` - gap between frames (1 means that all frames are used)       
`--dilate_kernel_size` and `--dilate_iterations` - params for mask dilation    
`--fcn` or `dlab` - model for segmentation
* Launch inpainting model:  
```
python tools/video_inpaint.py --frame_dir ./frames --MASK_ROOT ./masks --img_size 512 832 --FlowNet2 --DFC --ResNet101 --Propagation 
```
* To make video from results:
```
python tools/video_from_frames --image_folder ./Inpaint_Res/inpaint_res --video_name result.avi --fps 25
```
### For more info look into `./description`
