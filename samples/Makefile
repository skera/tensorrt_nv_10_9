# SPDX-FileCopyrightText: Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

SHELL = /bin/bash -o pipefail
TARGET ?= $(shell uname -m)
LIBDIR ?= lib
VERBOSE ?= 0
ifeq ($(VERBOSE), 1)
  AT =
else
  AT = @
endif

CUDA_TRIPLE = x86_64-linux
DLSW_TRIPLE = x86_64-linux-gnu
ifeq ($(TARGET), aarch64)
  ifeq ($(ARMSERVER), 1)
    CUDA_TRIPLE = sbsa-linux
  else
    CUDA_TRIPLE = aarch64-linux
  endif
  DLSW_TRIPLE = aarch64-linux-gnu
endif
ifeq ($(TARGET), qnx)
  CUDA_TRIPLE = aarch64-qnx
  DLSW_TRIPLE = aarch64-unknown-nto-qnx
endif
ifeq ($(TARGET), ppc64le)
  CUDA_TRIPLE = ppc64le-linux
  DLSW_TRIPLE = ppc64le-linux
endif
export TARGET
export LIBDIR
export VERBOSE
export CUDA_TRIPLE
export DLSW_TRIPLE

ifeq ($(SAFE_PDK), 1)
  # Only dlaSafetyRuntime is currently able to execute with safety pdk.
  samples := dlaSafetyRuntime
else
  samples := \
          trtexec \
          sampleDemo
  # TRTS-9861 TRTS-10013 add trtSafeExec and sampleSafeMNIST respectively
  ifeq ($(SAFETY_SAMPLE_BUILD), 1)
    ifeq ($(TARGET),x86_64)
      samples += trtSafeExec
    endif
  endif

  ifeq ($(ENABLE_DLA), 1)
    samples += sampleCudla
  endif

  # skip std samples since they have non-safety certified dependencies.
  # TRTS-9861 TRTS-10013 add trtSafeExec and sampleSafeMNIST respectively
  ifeq ($(QNX_SAFE_BUILD), 1)
    samples =
  endif
endif

.PHONY: all clean help
all:
	$(AT)$(foreach sample, $(samples), $(MAKE) -C $(sample) &&) :

clean:
	$(AT)$(foreach sample, $(samples), $(MAKE) clean -C $(sample) &&) :

help:
	$(AT)echo "Sample building help menu."
	$(AT)echo "Samples:"
	$(AT)$(foreach sample, $(samples), echo -e "\t$(sample)" &&) :
	$(AT)echo -e "\nCommands:"
	$(AT)echo -e "\tall - build all samples."
	$(AT)echo -e "\tclean - clean all samples."
	$(AT)echo -e "\nVariables:"
	$(AT)echo -e "\tTARGET - Specify the target to build for."
	$(AT)echo -e "\tVERBOSE - Specify verbose output."
	$(AT)echo -e "\tCUDA_INSTALL_DIR - Directory where cuda installs to."