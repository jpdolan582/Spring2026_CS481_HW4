This repository contains all relevant files for CS481 Homework 4

life and life.cu are the executable and program file for this project
life.cu contains the CUDA modified code of the instructor's life.c

output.5120.5000.x files each contain the output board from each parallel section of the GPU
these were all found to be identical boards, signifying a correct parallel implementation

hw4shGPU.o3663x files contain specific information from the GPU during each running instance

cuda_life_output.txt contains the actual output from the program, containing time taken and number of iterations reached
cuda_life_output.txt was overwritten after each run, resulting in timing averages not being directly available
each time used in the average calculation was recorded by hand prior to cuda_life_output.txt being overwritten upon next run
