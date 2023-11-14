1.I already created new virtual env called ocr in my system,choose python 3.9 when creation of env

1.1 install pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
1.2 pip install ultralytics dill pytorch-lightning==1.9.5 timm nltk

2.open yolo.ipynb in jupiter ,check that yolo jupiter file in ocr env
3.install matplot lib,
4.give yolov5 path to that sys path ex.sys.path.append('C:/Users/shan/engraved_ocr/yolov5/')
5.instll scikit-image,pandas
6.thats it run all lines in yolo.ipynb

(above for jupyter file)
<------------------------------------------------------------------------------------->
(following for docker)
1.install docker
2.go to the particular folder where the original dockerfile is there.
3.create a temporary image
ex.docker build -t client-ocr .

4.then run
docker run -p 8501:8501 client-ocr

5.if you will get any browser error then use localhost:8501 instaed 0f 0.0.0.1:8501.



<----------------------------------------------------------->
check git ignore file