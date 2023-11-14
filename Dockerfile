FROM python:3.9

RUN apt-get update
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install streamlit

COPY best.pt data.yaml epoch=16-step=2086-val_accuracy=100.0000-val_NED=100.0000.ckpt ParseqPretrained.pth tokenizer.pkl ./app/
ADD app.py __init__.py custom_utils.py ./app/

# Clone the parseq repository
RUN apt-get update && \
    apt-get install -y git && \
    git clone https://github.com/baudm/parseq /app/parseq

# Clone the yolov5 repository
RUN git clone https://github.com/ultralytics/yolov5 /app/yolov5

# Set the working directory
WORKDIR /app/parseq

# Set platform variable
#ENV platform=cpu

# Generate requirements files for specified PyTorch platform
#RUN make torch-${platform}

# Install the project and core + train + test dependencies
RUN pip install -r requirements/train.txt

# Install pip-tools and generate requirements files
RUN pip install pip-tools && \
    cd /app/parseq && \
    make clean-reqs reqs

RUN pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install ultralytics dill pytorch-lightning==1.9.5 timm nltk

EXPOSE 8501

WORKDIR /app

ENTRYPOINT ["streamlit", "run", "/app/app.py"]
