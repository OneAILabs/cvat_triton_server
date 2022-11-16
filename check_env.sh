#!/bin/bash

if [[ "$(find -L /usr -name libcuda.so.1 | grep -v "compat") " == " " || "$(ls /dev/nvidiactl 2>/dev/null) " == " " ]]; then
  echo
  echo "WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available."
  echo "   Use Docker with NVIDIA Container Toolkit to start this container; see"
  echo "   https://github.com/NVIDIA/nvidia-docker."
  #ln -s `find / -name libnvidia-ml.so -print -quit` /opt/tritonserver/lib/libnvidia-ml.so.1
  export TRITON_SERVER_CPU_ONLY=1
else
  ( /usr/local/bin/checkSMVER.sh )
  DRIVER_VERSION=$(sed -n 's/^NVRM.*Kernel Module *\([0-9.]*\).*$/\1/p' /proc/driver/nvidia/version 2>/dev/null || true)
  if [[ ! "$DRIVER_VERSION" =~ ^[0-9]*.[0-9]*(.[0-9]*)?$ ]]; then
    echo "Failed to detect NVIDIA driver version."
  elif [[ "${DRIVER_VERSION%%.*}" -lt "${CUDA_DRIVER_VERSION%%.*}" ]]; then
    if [[ "${_CUDA_COMPAT_STATUS}" == "CUDA Driver OK" ]]; then
      echo
      echo "NOTE: Legacy NVIDIA Driver detected.  Compatibility mode ENABLED."
    else
      echo
      echo "ERROR: This container was built for NVIDIA Driver Release ${CUDA_DRIVER_VERSION%.*} or later, but"
      echo "       version ${DRIVER_VERSION} was detected and compatibility mode is UNAVAILABLE."
      echo
      echo "       [[${_CUDA_COMPAT_STATUS}]]"
      export TRITON_SERVER_CPU_ONLY=1
      sleep 2
    fi
  fi
fi
