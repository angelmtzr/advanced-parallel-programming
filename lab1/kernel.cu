
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void print_hello_cuda()
{
	printf("[DEVICE] ThreadIdx.x: %d\n", threadIdx.x);
}

int main() 
{
	print_hello_cuda << <2, 8 >> > ();
	return 0;
}