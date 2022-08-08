#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <cstdlib>

#include <cuda_runtime.h>

using namespace std;

void stopTimer(std::chrono::time_point<std::chrono::system_clock> start) {
	auto stop = std::chrono::system_clock::now();
	auto elapsed = stop - start;
	auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
	auto seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed);
	printf("time elapsed: %lld milliseconds (%lld seconds)\n", milliseconds.count(), seconds.count());
}

__global__ void polyEvalKern(int* buckets, uint64_t* primes, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	int c = 0;
	if (i < n) {
		for (uint64_t j = 1; j < primes[i]; j++) {
			if (((j * j * j -3*j -1) % primes[i]) == 0)
				c++;
		}
		 
		atomicAdd(&buckets[(c<2)?c:2], 1);
		//__syncthreads();
		//buckets[c]++;
	}

}

void polynomialEval(int* buckets, uint64_t* primes, int n) {
	int size = n * sizeof(uint64_t);
	uint64_t* d_primes;
	cudaError_t code0 = cudaMalloc((void**)&d_primes, size);
	if (code0 != cudaSuccess)
		printf("0 the memory allocation was unsuccessful\n%s\n", cudaGetErrorString(code0));

	cudaError_t code1 = cudaMemcpy(d_primes, primes, size, cudaMemcpyHostToDevice);
	if (code1 != cudaSuccess) {
		printf("1 the copy was unsucessful\n %s\n", cudaGetErrorString(code1));
	}

	int* d_buckets = nullptr;
	cudaMalloc((void**)&d_buckets,(size_t) 3 * sizeof(int));
	cudaMemset(d_buckets, 0, (size_t)3 * sizeof(int));
	//cudaError_t code2 = cudaMemcpy(d_buckets, buckets, 3 * sizeof(int), cudaMemcpyHostToDevice);
	//if (code2 != cudaSuccess) {
	//	printf("2 the copy was unsucessful\n %s\n", cudaGetErrorString(code2));
	//}

	polyEvalKern << <ceil(n / 256.0), 256 >> > (d_buckets, d_primes, n);
	cudaDeviceSynchronize();

	cudaError_t code3 = cudaMemcpy(buckets, d_buckets, 3 * sizeof(int), cudaMemcpyDeviceToHost);
	if (code3 != cudaSuccess) {
		printf("3 the copy was unsucessful\n%s\n", cudaGetErrorString(code3));
	}


	cudaFree(d_buckets); cudaFree(d_primes);
}

int main(int argc, char* argv[])
{
	//int n;
	uint64_t* selectPrimes;
	int buckets[3] = {0};


	if (argc != 2) {
		printf("usage: %s primes.txt", argv[0]);
		return 1;
	}

	std::ifstream file(argv[1]);
	if (!file.good()) {
		printf("failed to open %s\n", argv[1]);
	}

	//cout << "put in the number of primes you want to test on" << endl;
	//cin >> n;


	std::string line;
	std::vector<uint64_t> primes;
	while (std::getline(file, line)) {
		primes.push_back(std::stoull(line));
	}

	printf("lines: %zu\n", primes.size());
//  primes.resize(n);

	auto start = std::chrono::system_clock::now();

	polynomialEval(buckets, primes.data(), primes.size());

	stopTimer(start);
	for (auto& b : buckets)
		cout << b << " ";
	cout << endl;


}