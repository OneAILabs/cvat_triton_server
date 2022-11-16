import sys
import numpy as np
from attrdict import AttrDict
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from tritonclient.utils import triton_to_np_dtype

url = "localhost:8000"
protocol = "http"


def parse_model(model_metadata, model_config):

    inputs_metadata = sorted(model_metadata.inputs, key=lambda x:x.name)
    #inputs_config = model_config.input
    outputs_metadata = sorted(model_metadata.outputs, key=lambda x:x.name)
    
    input_names = [input.name for input in inputs_metadata]
    output_names = [output.name for output in outputs_metadata]
    input_shapes = [input.shape for input in inputs_metadata]
    output_shapes = [output.shape for output in outputs_metadata]
    input_dtypes = [input.datatype for input in inputs_metadata]

    return (input_names, output_names, input_shapes, output_shapes, input_dtypes)

def convert_http_metadata_config(_metadata, _config):
    _model_metadata = AttrDict(_metadata)
    _model_config = AttrDict(_config)

    return _model_metadata, _model_config

def detect(inputs, model_name, model_version='1'):
    myinputs = []
    myoutputs = []
    try:
        if protocol.lower() == "grpc":
            # Create gRPC client for communicating with the server
            triton_client = grpcclient.InferenceServerClient(
                url=url, verbose=False)
        else:
            # Specify large enough concurrency to handle the
            # the number of requests.
            triton_client = httpclient.InferenceServerClient(
                url=url, verbose=False, concurrency=1)
    except Exception as e:
        print("client creation failed: " + str(e))
        sys.exit(1)
    try:
        model_metadata = triton_client.get_model_metadata(
            model_name=model_name, model_version=model_version)
    except InferenceServerException as e:
        print("failed to retrieve the metadata: " + str(e))
        sys.exit(1)

    try:
        model_config = triton_client.get_model_config(
            model_name=model_name, model_version=model_version)
    except InferenceServerException as e:
        print("failed to retrieve the config: " + str(e))
        sys.exit(1)
    if protocol.lower() == "grpc":
        model_config = model_config.config
    else:
        model_metadata, model_config = convert_http_metadata_config(
            model_metadata, model_config)
    input_names, output_names, input_shapes, output_shapes, input_dtypes = parse_model(model_metadata, model_config)
    if protocol == "grpc":
        client = grpcclient
    else:
        client = httpclient
    for input, input_name, input_shape, input_dtype in zip(inputs, input_names, input_shapes, input_dtypes):
        myinputs.append(client.InferInput(input_name, input.shape, "FP32"))
        myinputs[-1].set_data_from_numpy(input.astype(np.float32))
    for output_name in output_names:
        myoutputs.append(client.InferRequestedOutput(output_name))
    
    results = triton_client.infer(
        model_name=model_name,
        inputs=myinputs,
        outputs=None)
    
    myresults = [results.as_numpy(output_name) for output_name in output_names]
    return myresults


