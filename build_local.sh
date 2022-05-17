#!/bin/sh

# example: sudo ./train_local.sh unet

image=$1

mkdir -p test_dir/model
mkdir -p test_dir/output

# rm -rf test_dir/model/*
rm -rf test_dir/output/*

docker build -t ${image} .