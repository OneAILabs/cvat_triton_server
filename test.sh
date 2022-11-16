#!/bin/bash

while getopts "m:d:a" opt
do
   case "$opt" in
      m ) test_model="$OPTARG";;
      a ) test_model="all";;
      d ) project_dir="$OPTARG";;
   esac
done

export PROJECT_DIR=$project_dir
export WARM_UP=enable

GPU_AVAILABLE=$(python3 check_env.py)

if [[ -v GPU_AVAILABLE ]] && [[ $GPU_AVAILABLE == 'Ture' ]]
then
    export INFERENCE_DEVICE=GPU
else
    export INFERENCE_DEVICE=CPU
fi

# yolov3 test
if [[ $test_model == "yolov3" ]] || [[ $test_model == "all" ]]
then
	cd $project_dir
	export CONFIG_FILE_PATH=$project_dir/tests/yolov3.cfg
	export WEIGHTS_FILE_PATH=$project_dir/tests/yolov3.weights
	export LABEL_FILE_PATH=$project_dir/tests/labels.txt
	export YOLO_VER='V3'
	export CONFIDENCE_THRESH=0.3
	export NMS_THRESH=0.5

	model_name="yolov3"
	python3 yolo/yolo_onnx/yolo_to_onnx.py
	if [ ! -f "/models/$model_name/1/model.onnx" ]
	then
	    echo "Failed to convert model to onnx format"
	    exit 1
	fi
	anchors=$(python3 yolo/yolo_onnx/get_anchors.py)
        export YOLO_ANCHORS=$anchors
	python3 build_config.py -m $model_name --max-batch-size 0 -f onnxruntime_onnx -d $INFERENCE_DEVICE
	tritonserver --model-repository=/models --log-verbose=0 --strict-model-config=false &
	until [[ $(curl -s -w "%{http_code}\n" http://localhost:8000/v2/models/stats -o /dev/null) = "200" ]]; do
            sleep 1
        done
	pytest -rP tests/test_yolo.py --cov yolo --cov-report xml:$project_dir/coverage/yolo-coverage.xml --cov-report html:$project_dir/coverage/yolo-coverage.html --junitxml=$project_dir/test-results/yolo.xml

	kill -9 $(pgrep tritonserver)
	rm -rf /models
	sleep 5
fi

# yolov4 test
if [[ $test_model == "yolov4" ]] || [[ $test_model == "all" ]]
then
        cd $project_dir
        export CONFIG_FILE_PATH=$project_dir/tests/yolov4.cfg
        export WEIGHTS_FILE_PATH=$project_dir/tests/yolov4.weights
        export LABEL_FILE_PATH=$project_dir/tests/labels.txt
        export YOLO_VER='V4'
	export CONFIDENCE_THRESH=0.3
        export NMS_THRESH=0.5

        model_name="yolov4"
        python3 yolo/yolo_onnx/yolo_to_onnx.py
        if [ ! -f "/models/$model_name/1/model.onnx" ]
        then
            echo "Failed to convert model to onnx format"
            exit 1
        fi
	anchors=$(python3 yolo/yolo_onnx/get_anchors.py)
        export YOLO_ANCHORS=$anchors
        python3 build_config.py -m $model_name --max-batch-size 0 -f onnxruntime_onnx -d $INFERENCE_DEVICE
        tritonserver --model-repository=/models --log-verbose=0 --strict-model-config=false &
        until [[ $(curl -s -w "%{http_code}\n" http://localhost:8000/v2/models/stats -o /dev/null) = "200" ]]; do
            sleep 1
        done
        pytest -rP tests/test_yolo.py --cov yolo --cov-report xml:$project_dir/coverage/yolo-coverage.xml --cov-report html:$project_dir/coverage/yolo-coverage.html --junitxml=$project_dir/test-results/yolo.xml

        kill -9 $(pgrep tritonserver)
        rm -rf /models
	sleep 5
fi


#siammask test

if [[ $test_model == "siammask" ]] || [[ $test_model == "all" ]]
then
	cd $project_dir
	export MODEL_FILE_PATH=$project_dir/tests/SiamMask_DAVIS.pth

	python3 siammask/conversion/convert.py --resume $MODEL_FILE_PATH --config siammask/config_davis.json

	temp_dir=$project_dir/siammask_models

	mkdir /models
	mkdir /models/feature_extractor
	mkdir /models/feature_extractor/1
	cp $temp_dir/feature_extractor.pt /models/feature_extractor/1/model.pt
	python3 build_config.py -m feature_extractor --inputs INPUT__0:-1,3,-1,-1:fp32 --outputs OUTPUT__0:-1,64,-1,-1:fp32 OUTPUT__1:-1,256,-1,-1:fp32 OUTPUT__2:-1,512,-1,-1:fp32 OUTPUT__3:-1,1024,-1,-1:fp32 --max-batch-size 0 -f pytorch_libtorch -d $INFERENCE_DEVICE
	
	mkdir /models/feature_downsampler
	mkdir /models/feature_downsampler/1
	cp $temp_dir/feature_downsampler.pt /models/feature_downsampler/1/model.pt
	python3 build_config.py -m feature_downsampler --inputs INPUT__0:-1,1024,-1,-1:fp32 --outputs OUTPUT__0:-1,256,-1,-1:fp32 --max-batch-size 0 -f pytorch_libtorch -d $INFERENCE_DEVICE       
	
	mkdir /models/rpn_model
	mkdir /models/rpn_model/1
	cp $temp_dir/rpn_model.pt /models/rpn_model/1/model.pt
	python3 build_config.py -m rpn_model --inputs INPUT__0:-1,256,7,7:fp32 INPUT__1:-1,256,31,31:fp32 --outputs OUTPUT__0:-1,10,25,25:fp32 OUTPUT__1:-1,20,25,25:fp32 --max-batch-size 0 -f pytorch_libtorch -d $INFERENCE_DEVICE
	
	mkdir /models/mask_conv_kernel
	mkdir /models/mask_conv_kernel/1
	cp $temp_dir/mask_conv_kernel.pt /models/mask_conv_kernel/1/model.pt
	python3 build_config.py -m mask_conv_kernel --inputs INPUT__0:-1,256,7,7:fp32 --outputs OUTPUT__0:-1,256,5,5:fp32 --max-batch-size 0 -f pytorch_libtorch -d $INFERENCE_DEVICE
	
	mkdir /models/mask_conv_search
	mkdir /models/mask_conv_search/1
	cp $temp_dir/mask_conv_search.pt /models/mask_conv_search/1/model.pt
	python3 build_config.py -m mask_conv_search --inputs INPUT__0:-1,256,31,31:fp32 --outputs OUTPUT__0:-1,256,29,29:fp32 --max-batch-size 0 -f pytorch_libtorch -d $INFERENCE_DEVICE
	
	mkdir /models/mask_depthwise_conv
	mkdir /models/mask_depthwise_conv/1
	cp $temp_dir/mask_depthwise_conv.pt /models/mask_depthwise_conv/1/model.pt
	python3 build_config.py -m mask_depthwise_conv --inputs INPUT__0:-1,256,29,29:fp32 INPUT__1:-1,256,5,5:fp32 --outputs OUTPUT__0:-1,256,25,25:fp32 --max-batch-size 0 -f pytorch_libtorch -d $INFERENCE_DEVICE
	
	mkdir /models/mask_head
	mkdir /models/mask_head/1
	cp $temp_dir/mask_head.pt /models/mask_head/1/model.pt
	python3 build_config.py -m mask_head --inputs INPUT__0:-1,256,25,25:fp32 --outputs OUTPUT__0:-1,3969,25,25:fp32 --max-batch-size 0 -f pytorch_libtorch -d $INFERENCE_DEVICE
		
	mkdir /models/refine_model
	mkdir /models/refine_model/1
	cp $temp_dir/refine_model.pt /models/refine_model/1/model.pt
	python3 build_config.py -m refine_model --inputs INPUT__0:-1,64,125,125:fp32 INPUT__1:-1,256,63,63:fp32 INPUT__2:-1,512,31,31:fp32 INPUT__3:-1,256,25,25:fp32 INPUT__4:2:int64 --outputs OUTPUT__0:-1,16129:fp32 --max-batch-size 0 -f pytorch_libtorch -d $INFERENCE_DEVICE
	tritonserver --model-repository=/models --log-verbose=0 --strict-model-config=false &
	
	until [[ $(curl -s -w "%{http_code}\n" http://localhost:8000/v2/models/stats -o /dev/null) = "200" ]]; do
            sleep 1
        done
        pytest -rP tests/test_siammask.py --cov siammask --cov-report xml:$project_dir/coverage/siammask-coverage.xml --cov-report html:$project_dir/coverage/siammask-coverage.html --junitxml=$project_dir/test-results/siammask.xml

	kill -9 $(pgrep tritonserver)
	rm -rf /models
	rm -rf $temp_dir
	sleep 5
fi

if [[ $test_model == "f-BRS" ]] || [[ $test_model == "all" ]]
then
	
	temp_dir=$project_dir/fbrs_interactive_segmentation/output_models
	
	#test resnet weights
	cd $project_dir/fbrs_interactive_segmentation
	export MODEL_FILE_PATH=$project_dir/tests/resnet34_dh128_sbd.pth
	python3 convert_onnx.py --checkpoint $MODEL_FILE_PATH --outdir $temp_dir
	python3 create_models.py $temp_dir
	tritonserver --model-repository=/models --log-verbose=0 --strict-model-config=false &

	until [[ $(curl -s -w "%{http_code}\n" http://localhost:8000/v2/models/stats -o /dev/null) = "200" ]]; do
            sleep 1
        done
        cd $project_dir
	pytest -rP tests/test_fbrs_resnet.py --cov fbrs_interactive_segmentation --cov-report xml:$project_dir/coverage/fbrs-resnet-coverage.xml --cov-report html:$project_dir/coverage/fbrs-resnet-coverage.html --junitxml=$project_dir/test-results/fbrs-resnet.xml

	kill -9 $(pgrep tritonserver)
	rm -rf /models
	rm -rf $temp_dir
	rm -rf $project_dir/fbrs_interactive_segmentation/*.json
	sleep 5

	#test hrnet weights
	cd $project_dir/fbrs_interactive_segmentation
        export MODEL_FILE_PATH=$project_dir/tests/hrnet18_ocr64_sbd.pth
        python3 convert_onnx.py --checkpoint $MODEL_FILE_PATH --outdir $temp_dir
        python3 create_models.py $temp_dir
        tritonserver --model-repository=/models --log-verbose=0 --strict-model-config=false &

        until [[ $(curl -s -w "%{http_code}\n" http://localhost:8000/v2/models/stats -o /dev/null) = "200" ]]; do
            sleep 1
        done
        cd $project_dir
        pytest -rP tests/test_fbrs_hrnet.py --cov fbrs_interactive_segmentation --cov-report xml:$project_dir/coverage/fbrs-hrnet-coverage.xml --cov-report html:$project_dir/coverage/fbrs-hrnet-coverage.html --junitxml=$project_dir/test-results/fbrs-hrnet.xml

        kill -9 $(pgrep tritonserver)
        rm -rf /models
        rm -rf $temp_dir
        rm -rf $project_dir/fbrs_interactive_segmentation/*.json
        sleep 5

fi

if [[ $test_model == "mask_rcnn" ]] || [[ $test_model == "all" ]]
then
	export MODEL_FILE_PATH=$project_dir/tests/mask_rcnn_coco.h5
	export LABEL_FILE_PATH=$project_dir/tests/labels.txt
	cd $project_dir/mask_rcnn/conversion
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
	rm temp.pb
        cd $project_dir
	python3 build_config.py -m mask_rcnn --inputs input_anchors:-1,4:fp32 input_image:-1,-1,3:fp32 input_image_meta:-1:fp32 --outputs mrcnn_detection/Reshape_1:-1,6:fp32 mrcnn_mask/Reshape_1:-1,28,28,-1:fp32  --max-batch-size 1 -f tensorflow_graphdef -d $INFERENCE_DEVICE
	tritonserver --model-repository=/models --log-verbose=0 --strict-model-config=false &

	until [[ $(curl -s -w "%{http_code}\n" http://localhost:8000/v2/models/stats -o /dev/null) = "200" ]]; do
            sleep 1
        done
        pytest -rP tests/test_mrcnn.py --cov mask_rcnn --cov-report xml:$project_dir/coverage/mask_rcnn-coverage.xml --cov-report html:$project_dir/coverage/mask_rcnn-coverage.html --junitxml=$project_dir/test-results/mask_rcnn.xml

        kill -9 $(pgrep tritonserver)
        rm -rf /models
	sleep 5
fi

if [[ $test_model == "reid" ]] || [[ $test_model == "all" ]]
then
	mkdir /models
        mkdir /models/reid
        mkdir /models/reid/1
        cp $project_dir/tests/person-reidentification-retail-0300.xml /models/reid/1/model.xml
        cp $project_dir/tests/person-reidentification-retail-0300.bin /models/reid/1/model.bin
        python3 build_config.py -m reid --inputs data:1,3,256,128:fp32 --outputs reid_embedding:1,512:fp32 --max-batch-size 0 -b openvino -d CPU
	tritonserver --model-repository=/models --log-verbose=0 --strict-model-config=false &

	until [[ $(curl -s -w "%{http_code}\n" http://localhost:8000/v2/models/stats -o /dev/null) = "200" ]]; do
            sleep 1
        done
        pytest -rP tests/test_reid.py --cov reid --cov-report xml:$project_dir/coverage/reid-coverage.xml --cov-report html:$project_dir/coverage/reid-coverage.html --junitxml=$project_dir/test-results/reid.xml

        kill -9 $(pgrep tritonserver)
        rm -rf /models
        sleep 5
fi
