#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Downloading aws-ofi-nccl-1.17.2..."
wget https://github.com/aws/aws-ofi-nccl/releases/download/v1.17.2/aws-ofi-nccl-1.17.2.tar.gz

echo "Extracting tarball..."
tar -xzvf aws-ofi-nccl-1.17.2.tar.gz

echo "Navigating to directory..."
cd aws-ofi-nccl-1.17.2

echo "Configuring build..."
CC=gcc CXX=g++ ac_cv_header_limits_h=yes ./configure \
    --with-libfabric=/opt/cray/libfabric/2.2.0rc1 \
    --with-cuda=/opt/nvidia/hpc_sdk/Linux_aarch64/25.5/cuda/12.9 \
    --enable-trace \
    --prefix=/lustre/blizzard/stf218/scratch/emin/aws-ofi-nccl-1.17.2 \
    --disable-tests

echo "Compiling..."
make

echo "Installing..."
make install

echo "Build and installation complete!"