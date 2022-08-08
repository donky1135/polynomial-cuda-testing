#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <chrono>
#include <cmath>

using namespace std;

__global__ void polyKernel(int64_t a, int64_t b, int prime_size, uint64_t* primes, int* bucket){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int c = 0;
    for(uint64_t i = 1; i <= prime_size; i++)
        if((i*(i*i + a) + b) % primes[index] == 0)
            c++;
    atomicAdd(&bucket[(c<2)?c:2],1);
};

void polyEval(int startL, int endL, vector<uint64_t> &h_primes, vector<vector<int>> &h_buckets){
    uint64_t *d_primes;
    int *d_buckets;

    for(int i = startL + 3; i <= endL + 3){
        polyEval(-i, i, )
    }

}

int main(int argc, char* argv[]){
    if(argc != 2){
        printf("%s requires primes.txt", argv[0]);
        return 1;
    }

    ifstream file(argv[1]);
    if(!file.good()){
        printf("%s is invalid", argv[1]);
        return 1;
    }

    string line;
    vector<uint64_t> primes;
    while(getline(file, line))
        primes.push_back(stoull(line));

    printf("\# of primes read in: %zu\n", primes.size());
    
    int n, startL, endL;
    cout << "put in the number of primes you want to test on" << endl;
	cin >> n;

    cout << "what number polynomial do you want to start eval on?";
    cin >> startL;
    cout << "what number polynomial do you want to end eval on?";

    if(n > primes.size()){
        printf("%d is bigger than %zu primes read in", n, primes.size());
        return 1;
    }

    if(startL < 1){
        printf("%d is an invalid place to evaluate the polynomial on");
        return 1;
    }

    if(endl < startL){
        printf("can't evaluate on a negative number of polynomials!");
        return 1;
    }
    vector<vector<int>> buckets(endL-startL, vector<int>(3));
    primes.resize(n);

    polyEval(startL, endL, primes, buckets);
}


