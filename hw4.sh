#!/bin/bash

source /apps/profiles/modules_asax.sh.dyn

module load cuda

./life 5120 5000 /scratch/$USER

./life 5120 5000 /scratch/$USER

./life 5120 5000 /scratch/$USER
