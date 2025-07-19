I have cloned from this repository on github
https://github.com/WongKinYiu/PyTorch_YOLOv4
Special thanks to github user WongKinYiu for pytorch implementation of yolov4

pre-trained weights to detect drone can be downloaded from here.
https://drive.google.com/drive/folders/19KCfcrGNT4VFSM6iP9LkwlEdkcYGmHG5?usp=sharing

The three files are explained below.
drone.cfg: this file defines the architecture of the model along with hyper parameters such as learning rate, optimizer.
drone.data: this file is used to define path to the training and validation data which model wil use during training and testing
drone.names: this file contains the classes that will be detected by the model.

Steps to make conda environment
1. git clone https://github.com/WongKinYiu/PyTorch_YOLOv4.git -b master
2. conda create -n  yolo_drone python=3.7
3. conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
4. change directory to downloaded repo: cd PyTorch_YOLOv4
5. place drone.cfg file in cfg/ folder.
6. place drone.data and drone.names files in data/ folder
7. place best_yolov4-one_class.pt weights file in weights/ folder
8. install required libraries using pip install -r requirements.txt
9. for inference use detect.py the details are given below. (use 0 in --source argument for webcam or attached camera)

Another way to is to download my cloned repository using this link which already has weights and config files in it
https://drive.google.com/file/d/1X77yNXXOi5n8IBO8fsCqnOlO3aGI2loz/view?usp=sharing
1. Unzip the folder and cd into folder
2. create conda environment like before.
3. install required libraries using pip install -r requirements.txt
4. for inference use detect.py the details are given below. (use 0 in --source argument for webcam or attached camera)

------------------------------------------FOR DETECTION-----------------------------------------------------
python detect.py --names 'data/drone.names' --cfg drone.cfg --weights 'weights/best_yolov4-one_class.pt'  --source 'test_video/00_01_52_to_00_01_58.mp4' --img-size 1024

    parser.add_argument('--cfg', type=str, default='cfg/yolov4-pacsp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/yolov4-pacsp.pt', help='weights path')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')


------------------------------------------FOR TRAINING-----------------------------------------------------
python train.py --data drone.data --cfg drone.cfg --weights 'weights/yolov4-pacsp-x.pt' --name yolov4-one_class --img 1024 1024 1024

(ARGUMENTS)
    parser.add_argument('--epochs', type=int, default=300)  # 500200 batches at bs 16, 117263 COCO images = 273 epochs
    parser.add_argument('--batch-size', type=int, default=16)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--cfg', type=str, default='cfg/yolov4-pacsp.cfg', help='*.cfg path')
    parser.add_argument('--data', type=str, default='data/coco2017.data', help='*.data path')
    parser.add_argument('--multi-scale', action='store_true', help='adjust (67%% - 150%%) img_size every 10 batches')
    parser.add_argument('--img-size', nargs='+', type=int, default=[320, 640], help='[min_train, max-train, test]')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')


------------------------------------------FOR TESTING-----------------------------------------------------
THE TEST SCRIPT RUNS TEST ON DATA THAT IS GIVEN IN VALIDATION PATH IN THE FILE drone.data

python test.py --cfg drone.cfg  --data drone.data --weights weights/best_yolov4-one_class.pt  --img 1024 --conf 0.5 --batch-size 8
    parser.add_argument('--cfg', type=str, default='cfg/yolov4-pacsp.cfg', help='*.cfg path')
    parser.add_argument('--data', type=str, default='data/coco2017.data', help='*.data path')
    parser.add_argument('--weights', type=str, default='weights/yolov4-pacsp.pt', help='weights path')
    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--task', default='test', help="'test', 'study', 'benchmark'")
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
