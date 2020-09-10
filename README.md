- [Human Pose estimation](#human-pose-estimation)
- [Face Recognition](#face-recognition)
- [Classfying flying objects](#classfying-flying-objects)
- [Detecting drones using YOLOV3](#detecting-drones-using-yolov3)
- [Monocular depth estimation and Backgroud substraction](#monocular-depth-estimation-and-backgroud-substraction)
- [Classify attributes of a person in given picture](#classify-attributes-of-a-person-in-given-picture)

<!-- # List of projects -->

<!-- ## Working on GAN -->

## Human Pose estimation
- Implemented [Simple Baseline for HPE and tracking](https://arxiv.org/pdf/1804.06208.pdf) 
- Used [pretrained weights](https://onedrive.live.com/?authkey=%21AFkTgCsr3CT9%2D%5FA&id=56B9F9C97F261712%2110709&cid=56B9F9C97F261712) opensourced by [authors](https://github.com/Microsoft/human-pose-estimation.pytorch)
- Converted pytorch model to [onxx](https://onnx.ai/) model to make it framework independent.
- Quantized the onnx model to reduce the size and deployed on AWS lambda
  - size reduced by around 40% 
  - accuracy reduced by around 5% 
- Key points mapped and skeleton drawn on the image
- Try it out [here](https://thetensorclan-web.herokuapp.com/human-pose-estimation)

Demo
![Pose Estimation Demo](images/hpe_demo.gif)


## Face Recognition 

- Used dlib's library to detect 68 points on the face, used to align the face, since the face in the image might be skewed or not facing directly in front.
- Trained model on Labeled Faces in the Wild(LFW) dataset along with some Indian face images. Notebooks available in notebook folder
- Try it out [here](https://thetensorclan-web.herokuapp.com/indian-face-recognizer)

- Face Swap
  - Create a mask for each of the faces, i.e. by creating a convex hull out of the detected 68 points from dlib model
  - Created [delaunay traingles](https://en.wikipedia.org/wiki/Delaunay_triangulation), so we can simply swap these triangles between the faces, or apply the triangles from one face to the other !
  - Try it [here](https://thetensorclan-web.herokuapp.com/face-swap)

Demo
![Face recog Demo](images/face_recog_demo.gif)
![face swap Demo](images/face_demo.gif)


## Classfying flying objects
<!-- Sqeeuzenet, Used MobileNet -->
- Complete end to end pipeline from data collection, cleaning, training to model deployment on AWS Lambda using serverless
- Read about it [here](https://github.com/hemanth346/EVA4_Phase2/blob/master/Session2/README.md)
- Try it out [here](https://thetensorclan-web.herokuapp.com/classifiers)
## Detecting drones using YOLOV3
- Tranied YoloV3 on custom dataset with 500 images of drones scrapped and labelled using custom Annotation tool
- Used pre-trained weights from a model trained on COCO dataset 
- Trained 70:30 split for 300 epochs and achieved train mAP@0.5 - 0.8 and Val avg. precision of 0.3 
- Ran inference on random youtube video and identified drones 80% of the time.
- Read about it [here](https://github.com/hemanth346/YoloV3_CustomData)

## Monocular depth estimation and Backgroud substraction
 
- created **_synthetic dataset of around 1.2 million_** 2D images
- generated depth map ground truth using existing state of the art model
- generate ground truths masks for background substraction using opencv and aplha channel thresholding
- Designed custom architecture, inspired by [Deep Residual U-Net](https://arxiv.org/pdf/1711.10684.pdf) but uses less param for but works good for both the tasks  
- Multiple loss functions are [explored](https://github.com/hemanth346/mde_bs#3-loss-functions).
  - SoftDiceLoss is used for mask over BCEloss(Binary Cross entropy) 
  - MSEloss(Mean Squared Error Loss) used for Depth maps
- DiceScore is used as metric, [details here](https://github.com/hemanth346/mde_bs#4-metrics-results-and-observations) 
- Dataset creation to training is done entirely on colab.
- TensorBoard(a profiler and visualization tool) is used to keep track of the progress

## Classify attributes of a person in given picture 
- Worked on labeling and annotations of images to inference
- Custom built Resnet V2 architecture for both base tower and Label towers
- Fully Convolutional Network, accounted for different receptive fields of the classes
- Created Custom LR scheduler obtained from OneCyclic LR




<!-- # Todo

## Ad click predection

## Key pair document extraction

## ML Incubator project

## Donors choose EDA ? -->
