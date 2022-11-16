#!/bin/bash

GPU_AVAILABLE=$(python3 check_env.py)

export ROOT=$(pwd)
export WARM_UP=enable

if [[ -v GPU_AVAILABLE ]] && [[ $GPU_AVAILABLE == 'True' ]]
then
    export INFERENCE_DEVICE=GPU
else
    export INFERENCE_DEVICE=CPU
fi

if [ -v MODEL ] && [ $MODEL = 'YOLO' ]
then
    if [ -v YOLO_VER ] && [ $YOLO_VER = 'V7' ]
    then
        model_name="yolov7"
        cd /opt/tritonserver/yolov7/yolov7/
        python3 yolo_to_torch.py
        if [ ! -f "/models/$model_name/1/model.pt" ]
            then
                echo "Failed to convert model to TorchScript"
                exit 1
        fi
        cd /opt/tritonserver/
        python3 build_config.py -m  $model_name  --max-batch-size 0  --inputs INPUT__0:1,3,$HEIGHT,$WIDTH:fp32 --outputs OUTPUT__0:1,-1,-1:fp32  -f pytorch_libtorch -d $INFERENCE_DEVICE
        cd /opt/tritonserver/yolov7/yolov7/utils/
        python3 setup.py install
        cd /opt/tritonserver/yolov7/
        python3 setup.py install
    else
        if [ -v YOLO_VER ] && [ $YOLO_VER = 'V4' ]
        then
            model_name="yolov4"
        else
            model_name="yolov3"
        fi
        python3 yolo/yolo_onnx/yolo_to_onnx.py
        if [ ! -f "/models/$model_name/1/model.onnx" ]
        then
            echo "Failed to convert model to onnx format"
            exit 1
        fi
        anchors=$(python3 yolo/yolo_onnx/get_anchors.py)
        export YOLO_ANCHORS=$anchors
        python3 build_config.py -m $model_name --max-batch-size 0 -f onnxruntime_onnx -d $INFERENCE_DEVICE -w models_warmup/yolo/yolo.pbtxt
        cd yolo
        python3 setup.py install
    fi


elif [ -v MODEL ] && [ $MODEL = 'SiamMask' ]
then
    if [ -v MODEL_FILE_PATH ]
    then
	python3 siammask/conversion/convert.py --resume $MODEL_FILE_PATH --config siammask/config_davis.json
	temp_dir=$ROOT/siammask_models
        mkdir /models
	mkdir /models/feature_extractor
	mkdir /models/feature_extractor/1
	cp $temp_dir/feature_extractor.pt /models/feature_extractor/1/model.pt
	python3 build_config.py -m feature_extractor --inputs INPUT__0:-1,3,-1,-1:fp32 --outputs OUTPUT__0:-1,64,-1,-1:fp32 OUTPUT__1:-1,256,-1,-1:fp32 OUTPUT__2:-1,512,-1,-1:fp32 OUTPUT__3:-1,1024,-1,-1:fp32 --max-batch-size 0 -f pytorch_libtorch -d $INFERENCE_DEVICE -w models_warmup/siammask/feature_extractor.pbtxt
	mkdir /models/feature_downsampler
        mkdir /models/feature_downsampler/1
        cp $temp_dir/feature_downsampler.pt /models/feature_downsampler/1/model.pt
        python3 build_config.py -m feature_downsampler --inputs INPUT__0:-1,1024,-1,-1:fp32 --outputs OUTPUT__0:-1,256,-1,-1:fp32 --max-batch-size 0 -f pytorch_libtorch -d $INFERENCE_DEVICE -w models_warmup/siammask/feature_downsampler.pbtxt       
	mkdir /models/rpn_model
        mkdir /models/rpn_model/1
        cp $temp_dir/rpn_model.pt /models/rpn_model/1/model.pt
        python3 build_config.py -m rpn_model --inputs INPUT__0:-1,256,7,7:fp32 INPUT__1:-1,256,31,31:fp32 --outputs OUTPUT__0:-1,10,25,25:fp32 OUTPUT__1:-1,20,25,25:fp32 --max-batch-size 0 -f pytorch_libtorch -d $INFERENCE_DEVICE -w models_warmup/siammask/rpn_model.pbtxt
        mkdir /models/mask_conv_kernel
        mkdir /models/mask_conv_kernel/1
        cp $temp_dir/mask_conv_kernel.pt /models/mask_conv_kernel/1/model.pt
        python3 build_config.py -m mask_conv_kernel --inputs INPUT__0:-1,256,7,7:fp32 --outputs OUTPUT__0:-1,256,5,5:fp32 --max-batch-size 0 -f pytorch_libtorch -d $INFERENCE_DEVICE -w models_warmup/siammask/mask_conv_kernel.pbtxt
	mkdir /models/mask_conv_search
        mkdir /models/mask_conv_search/1
        cp $temp_dir/mask_conv_search.pt /models/mask_conv_search/1/model.pt
        python3 build_config.py -m mask_conv_search --inputs INPUT__0:-1,256,31,31:fp32 --outputs OUTPUT__0:-1,256,29,29:fp32 --max-batch-size 0 -f pytorch_libtorch -d $INFERENCE_DEVICE -w models_warmup/siammask/mask_conv_search.pbtxt
	mkdir /models/mask_depthwise_conv
        mkdir /models/mask_depthwise_conv/1
        cp $temp_dir/mask_depthwise_conv.pt /models/mask_depthwise_conv/1/model.pt
        python3 build_config.py -m mask_depthwise_conv --inputs INPUT__0:-1,256,29,29:fp32 INPUT__1:-1,256,5,5:fp32 --outputs OUTPUT__0:-1,256,25,25:fp32 --max-batch-size 0 -f pytorch_libtorch -d $INFERENCE_DEVICE -w models_warmup/siammask/mask_depthwise_conv.pbtxt
	mkdir /models/mask_head
        mkdir /models/mask_head/1
        cp $temp_dir/mask_head.pt /models/mask_head/1/model.pt
        python3 build_config.py -m mask_head --inputs INPUT__0:-1,256,25,25:fp32 --outputs OUTPUT__0:-1,3969,25,25:fp32 --max-batch-size 0 -f pytorch_libtorch -d $INFERENCE_DEVICE -w models_warmup/siammask/mask_head.pbtxt
	mkdir /models/refine_model
        mkdir /models/refine_model/1
        cp $temp_dir/refine_model.pt /models/refine_model/1/model.pt
        python3 build_config.py -m refine_model --inputs INPUT__0:-1,64,125,125:fp32 INPUT__1:-1,256,63,63:fp32 INPUT__2:-1,512,31,31:fp32 INPUT__3:-1,256,25,25:fp32 INPUT__4:2:int64 --outputs OUTPUT__0:-1,16129:fp32 --max-batch-size 0 -f pytorch_libtorch -d $INFERENCE_DEVICE -w models_warmup/siammask/refine_model.pbtxt
	rm -rf $temp_dir
	cd siammask
	python3 setup.py install
    fi
fi

if [ -v MODEL ] && [ $MODEL = 'f-BRS' ]
then
    if [ -v MODEL_FILE_PATH ]
    then
	temp_dir=$ROOT/fbrs_interactive_segmentation/output_models
	cd $ROOT/fbrs_interactive_segmentation
	python3 convert_onnx.py --checkpoint $MODEL_FILE_PATH --outdir $temp_dir
	python3 create_models.py $temp_dir
	rm -rf $temp_dir
	python3 setup.py install
    fi
fi

if [ -v MODEL ] && [ $MODEL = 'MaskRCNN' ]
then
    if [ -v MODEL_FILE_PATH ]
    then
        cd $ROOT/mask_rcnn/conversion
        python3 convert_to_pb.py -w $MODEL_FILE_PATH
	if [ ! -f "temp.pb" ]
        then
            echo "Failed to convert model to tensorflow graphdef"
            exit 1
        fi
	mkdir /models
	mkdir /models/mask_rcnn
	mkdir /models/mask_rcnn/1
	cp temp.pb /models/mask_rcnn/1/model.graphdef
	cd $ROOT
	python3 build_config.py -m mask_rcnn --inputs input_anchors:-1,4:fp32 input_image:-1,-1,3:fp32 input_image_meta:-1:fp32 --outputs mrcnn_detection/Reshape_1:-1,6:fp32 mrcnn_mask/Reshape_1:-1,28,28,-1:fp32  --max-batch-size 1 -f tensorflow_graphdef -d $INFERENCE_DEVICE -w models_warmup/mask_rcnn/mask_rcnn.pbtxt
        cd $ROOT/mask_rcnn
	python3 setup.py install
    fi
fi

if [ -v MODEL ] && [ $MODEL = 'Person-reid' ]
then
    if [ -v MODEL_DIR ]
    then
	mkdir /models
	mkdir /models/reid
	mkdir /models/reid/1
	cp $MODEL_DIR/person-reidentification-retail-0300.xml /models/reid/1/model.xml
	cp $MODEL_DIR/person-reidentification-retail-0300.bin /models/reid/1/model.bin
        python3 build_config.py -m reid --inputs data:1,3,256,128:fp32 --outputs reid_embedding:1,512:fp32 --max-batch-size 0 -b openvino -d CPU
	cd $ROOT/reid
	python3 setup.py install
    fi
fi


cd $ROOT/webapp
python3 setup.py install
tritonserver --model-repository=/models --strict-model-config=false &
until [[ $(curl -s -w "%{http_code}\n" http://localhost:8000/v2/models/stats -o /dev/null) = "200" ]]; do
    sleep 1
done
waitress-serve --call --port 9999 "webapp:create_app"

