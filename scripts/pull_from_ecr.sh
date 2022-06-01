#!/usr/bin/env bash

account="763104351884"

region=$(aws configure get region)
region=${region:-us-west-2}

aws ecr get-login-password --region "${region}" | docker login --username AWS --password-stdin "${account}".dkr.ecr."${region}".amazonaws.com

docker pull "${account}".dkr.ecr."${region}".amazonaws.com/pytorch-inference:1.11.0-cpu-py38-ubuntu20.04-sagemaker-v1.0