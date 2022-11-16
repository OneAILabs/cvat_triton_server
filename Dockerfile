FROM nvcr.io/nvidia/tritonserver:21.03-py3
COPY nvidia_entrypoint.sh .
RUN chmod 775 /opt/tritonserver/nvidia_entrypoint.sh
COPY tritonclient-2.8.0-py3-none-manylinux1_x86_64.whl .
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN apt-get update && apt-get install -y python3-dev \
         libsm6 \
         libxext6 \ 
         libxrender-dev \
         protobuf-compiler \
         libprotoc-dev

RUN pip3 install cmake \
         pybind11[global] \
         Pillow \
         waitress \
         flask_restful \
         jsonschema \
         pynvml \
         jsonpickle \
         opencv_python==3.4.8.29 \
         torch==1.8.1 \
         jupyter \
         attrdict \
         tritonclient-2.8.0-py3-none-manylinux1_x86_64.whl[all] \
         pytest-cov \
         torchvision==0.9.1 \
         Cython \
         scipy \
         keras==2.2.4 \
         nvidia-pyindex \
         scikit-image

RUN pip3 install onnx==1.4.1 \
         nvidia-tensorflow[horovod]
RUN pip3 install AIMakerMonitor==1.0.5

RUN pip3 install pandas \
         tqdm \
         matplotlib \
         seaborn


ADD yolo ./yolo
ADD siammask ./siammask
ADD fbrs_interactive_segmentation ./fbrs_interactive_segmentation
ADD mask_rcnn ./mask_rcnn
ADD reid ./reid
ADD models_warmup ./models_warmup
ADD webapp ./webapp
COPY run.sh .
RUN chmod 775 /opt/tritonserver/run.sh
COPY build_config.py .
COPY check_env.py .
COPY check_env.sh .
RUN chmod 775 /opt/tritonserver/check_env.sh
COPY yolov3.ipynb .
ADD yolov7 ./yolov7