/*
 ============================================================================
 Name        : main.cu
 Author      : imtsuki
 Version     : 0.1.0
 Copyright   : imtsuki <me@qjx.app>
 Description : Flexible Job Shop Scheduling Problem
 ============================================================================
 */

#include <iostream>
#include <fstream>
#include <numeric>
#include <cstdlib>
#include <climits>
#include <stdexcept>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/tuple.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

#include <curand.h>
#include <curand_kernel.h>

static void CheckCudaErrorAux(const char *, unsigned, const char *,
        cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

const int MAX_OPERATIONS_PER_STEP = 5;
const int MAX_STEPS_PER_JOB = 20;
const int MAX_JOBS = 20;
const int MAX_MACHINES = 20;

int POPULATION_SIZE = 2000;
int INDIVIDUAL_LEN = 20; // TODO
const int SIZE_PARENT_POOL = 7;

int TOTALTHREADS = 2048;
int BLOCKSIZE = 1024;

int total_jobs, total_machines, max_operations;

struct Operation {
    int id_machine;
    int processing_time;
};

struct Step {
    int len;
    Operation candidates[MAX_OPERATIONS_PER_STEP];
};

struct Job {
    int len;
    Step steps[MAX_STEPS_PER_JOB];
};

Job input_data[MAX_JOBS];

struct Gene {
    int id_job;
    int id_step;
    // Make sure update them both.
    int id_machine;
    int id_operation;
};

std::ostream &operator<<(std::ostream &os, const Gene &gene) {
    os << "[" << gene.id_job << ", " << gene.id_step << ", "
            << gene.id_operation << "]";
    return os;
}

void parse_input(const char *path) {
    auto input = std::ifstream();

    input.exceptions(std::ifstream::failbit);

    input.open(path);

    input >> total_jobs >> total_machines >> max_operations;

    if (total_jobs > MAX_JOBS) {
        throw std::runtime_error("Too many jobs");
    }

    if (total_machines > MAX_MACHINES) {
        throw std::runtime_error("Too many machines");
    }

    INDIVIDUAL_LEN = 0;

    for (int id_job = 0; id_job < total_jobs; id_job++) {
        int number_steps;
        input >> number_steps;

        if (number_steps > MAX_STEPS_PER_JOB) {
            throw std::runtime_error("Too many steps");
        }

        input_data[id_job].len = number_steps;

        for (int id_step = 0; id_step < number_steps; id_step++) {
            int number_operations;
            input >> number_operations;

            if (number_operations > MAX_OPERATIONS_PER_STEP) {
                throw std::runtime_error("Too many operations");
            }

            input_data[id_job].steps[id_step].len = number_operations;

            for (int id_operation = 0; id_operation < number_operations;
                    id_operation++) {
                int id_machine;
                int processing_time;
                input >> id_machine >> processing_time;
                input_data[id_job].steps[id_step].candidates[id_operation].id_machine =
                        id_machine - 1;
                input_data[id_job].steps[id_step].candidates[id_operation].processing_time =
                        processing_time;
            }
            INDIVIDUAL_LEN++;
        }
    }
}

__global__ void init_rand_kernel(curandState_t *states, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void fill_rand_kernel(int *numbers, int len, int max_value,
        curandState_t *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    numbers[idx] = curand(&states[idx]) % max_value;
}

__global__ void init_population_kernel(Gene *population, int population_size,
        int individual_len, Job *jobs, int total_jobs,
        curandState_t *rand_states) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int next_step[MAX_JOBS];

    if (index < population_size) {
        for (int i = index; i < population_size; i += stride) {
            int cursor = 0;
            Gene *me = population + i * individual_len;
            memset(next_step, 0, sizeof(next_step));
            while (cursor < individual_len) {
                int id_job = curand(&rand_states[i]) % total_jobs;
                if (next_step[id_job] < jobs[id_job].len) {
                    me[cursor].id_job = id_job;
                    me[cursor].id_step = next_step[id_job];
                    next_step[id_job]++;
                    cursor++;
                }
            }
        }
    }

}

__global__ void pick_parents_kernel(int *parents, int *parent_candidates,
        int *scores, int population_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if (index < population_size) {
        for (int i = index; i < population_size; i += stride) {
            int best_score = INT_MAX;
            int best_index = -1;

            for (int j = 0; j < SIZE_PARENT_POOL; j++) {
                int k = parent_candidates[i * SIZE_PARENT_POOL + j];
                if (scores[k] < best_score) {
                    best_score = scores[k];
                    best_index = k;
                }
            }

            parents[i] = best_index;
        }
    }
}

__device__ void assignment_crossover(Gene *child, Gene *parent_a,
        Gene *parent_b, int individual_len, Job *jobs) {
    int reverse_index[MAX_JOBS][MAX_STEPS_PER_JOB];

    for (int s = 0; s < individual_len; s++) {
        int id_job = parent_b[s].id_job;
        int id_step = parent_b[s].id_step;
        reverse_index[id_job][id_step] = s;
    }

    for (int s = 0; s < individual_len; s++) {
        int id_job = parent_a[s].id_job;
        int id_step = parent_a[s].id_step;
        int i = reverse_index[id_job][id_step];

        child[s] = parent_a[s];
        child[s].id_operation = parent_b[i].id_operation;
        child[s].id_machine = parent_b[i].id_machine;
    }
}

__device__ void sequencing_crossover(Gene *child, Gene *parent_a,
        Gene *parent_b, int individual_len, Job *jobs,
        curandState_t *rand_state) {
    int crossover_point = curand(rand_state) % individual_len;

    int last_step[MAX_JOBS];

    for (int i = 0; i < MAX_JOBS; i++) {
        last_step[i] = -1;
    }

    for (int s = 0; s < crossover_point; s++) {
        int id_job = parent_b[s].id_job;
        int id_step = parent_b[s].id_step;

        child[s] = parent_b[s];
        last_step[id_job] = id_step;
    }

    int cursor = crossover_point;

    for (int s = 0; s < individual_len; s++) {
        int id_job = parent_a[s].id_job;

        if (last_step[id_job] < parent_a[s].id_step) {
            child[cursor] = parent_a[s];
            cursor++;
        }
    }
}

__device__ void assignment_mutation(Gene *individual, int individual_len,
        Job *jobs, curandState_t *rand_state) {
    int count = 5;
    while (count--) {
        int mutation_point = curand(rand_state) % individual_len;
        int id_job = individual[mutation_point].id_job;
        int id_step = individual[mutation_point].id_step;
        int len = jobs[id_job].steps[id_step].len;

        int id_operation = curand(rand_state) % len;

        individual[mutation_point].id_operation = id_operation;
        individual[mutation_point].id_machine =
                jobs[id_job].steps[id_step].candidates[id_operation].id_machine;
    }

}

__device__ void swapping_mutation(Gene *individual, int individual_len,
        Job *jobs, curandState_t *rand_state) {
    int count = 5;
    while (count--) {
        int mutation_point = curand(rand_state) % (individual_len - 1);

        if (individual[mutation_point].id_job
                != individual[mutation_point + 1].id_job) {
            thrust::swap(individual[mutation_point],
                    individual[mutation_point + 1]);
        }
    }

}

__global__ void stage_1_breed_kernel(int *parents, Gene *population,
        Gene *new_population, int population_size, int individual_len,
        Job *jobs, curandState_t *rand_states) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if (index < population_size) {
        for (int i = index; i < population_size; i += stride) {
            if (i < population_size * 8 / 10) {
                sequencing_crossover(&new_population[i * individual_len],
                        &population[parents[i] * individual_len],
                        &population[parents[i + 1] * individual_len],
                        individual_len, jobs, &rand_states[i]);
            } else {
                for (int s = 0; s < individual_len; s++) {
                    new_population[i * individual_len + s] =
                            population[parents[i] * individual_len + s];
                }

                swapping_mutation(&new_population[i * individual_len],
                        individual_len, jobs, &rand_states[i]);
            }
        }
    }
}

__global__ void stage_2_breed_kernel(int *parents, Gene *population,
        Gene *new_population, int population_size, int individual_len,
        Job *jobs, curandState_t *rand_states) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if (index < population_size) {
        for (int i = index; i < population_size; i += stride) {
            if (i < population_size * 4 / 10) {
                assignment_crossover(&new_population[i * individual_len],
                        &population[parents[i] * individual_len],
                        &population[parents[i + 1] * individual_len],
                        individual_len, jobs);
            } else if (i < population_size * 8 / 10) {
                sequencing_crossover(&new_population[i * individual_len],
                        &population[parents[i] * individual_len],
                        &population[parents[i + 1] * individual_len],
                        individual_len, jobs, &rand_states[i]);
            } else {
                for (int s = 0; s < individual_len; s++) {
                    new_population[i * individual_len + s] =
                            population[parents[i] * individual_len + s];
                }
                if (i < population_size * 9 / 10) {
                    assignment_mutation(&new_population[i * individual_len],
                            individual_len, jobs, &rand_states[i]);
                } else {
                    swapping_mutation(&new_population[i * individual_len],
                            individual_len, jobs, &rand_states[i]);

                }
            }
        }
    }
}

__global__ void stage_1_evaluate_kernel(int *scores, Gene *population,
        int population_size, int individual_len, Job *jobs) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int value;
    int machines[MAX_MACHINES];
    int last_step_id_machine[MAX_JOBS];

    if (index < population_size) {
        for (int i = index; i < population_size; i += stride) {
            value = 0;
            memset(machines, 0, sizeof(machines));
            Gene *me = population + i * individual_len;

            for (int s = 0; s < individual_len; s++) {
                int id_job = me[s].id_job;
                int id_step = me[s].id_step;
                int len = jobs[id_job].steps[id_step].len;
                int best_end_time = INT_MAX;
                int best_id_operation = -1;
                int best_id_machine = -1;

                // Greedy search to find best operation in this step
                for (int id_operation = 0; id_operation < len; id_operation++) {
                    int processing_time =
                            jobs[id_job].steps[id_step].candidates[id_operation].processing_time;
                    int id_machine =
                            jobs[id_job].steps[id_step].candidates[id_operation].id_machine;

                    int machine_end_time = machines[id_machine];

                    if (id_step > 0) {
                        int previous_id_machine = last_step_id_machine[id_job];
                        if (machine_end_time < machines[previous_id_machine]) {
                            machine_end_time = machines[previous_id_machine];
                        }
                    }

                    machine_end_time += processing_time;

                    if (machine_end_time < best_end_time) {
                        best_end_time = machine_end_time;
                        best_id_operation = id_operation;
                        best_id_machine = id_machine;
                    }
                }
                me[s].id_operation = best_id_operation;
                me[s].id_machine = best_id_machine;
                machines[best_id_machine] = best_end_time;
                last_step_id_machine[id_job] = best_id_machine;
                if (best_end_time > value) {
                    value = best_end_time;
                }
            }

            scores[i] = value;
        }
    }
}

__global__ void stage_2_evaluate_kernel(int *scores, Gene *population,
        int population_size, int individual_len, Job *jobs) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int value;
    int machines[MAX_MACHINES];
    int last_step_id_machine[MAX_JOBS];

    if (index < population_size) {
        for (int i = index; i < population_size; i += stride) {
            value = 0;
            memset(machines, 0, sizeof(machines));
            Gene *me = population + i * individual_len;

            for (int s = 0; s < individual_len; s++) {
                int id_job = me[s].id_job;
                int id_step = me[s].id_step;
                int id_machine = me[s].id_machine;
                int id_operation = me[s].id_operation;

                int processing_time =
                        jobs[id_job].steps[id_step].candidates[id_operation].processing_time;

                int previous_id_machine = last_step_id_machine[id_job];

                machines[id_machine] =
                        (id_step > 0
                                && machines[id_machine]
                                        < machines[previous_id_machine]) ?
                                machines[previous_id_machine] :
                                machines[id_machine];

                machines[id_machine] += processing_time;

                value = machines[id_machine] > value ?
                        machines[id_machine] : value;

                last_step_id_machine[id_job] = id_machine;
            }

            scores[i] = value;
        }
    }
}

int main(int argc, const char *argv[]) {
    cudaDeviceProp prop;
    CUDA_CHECK_RETURN(cudaGetDeviceProperties(&prop, 0));

    std::cout << "GPU device: " << prop.name << std::endl;
    std::cout << "Number of SM: " << prop.multiProcessorCount << std::endl;
    std::cout << "Shared memory per block: " << prop.sharedMemPerBlock / 1024.0
            << " KB" << std::endl;
    std::cout << "Max Threads per block: " << prop.maxThreadsPerBlock
            << std::endl;
    std::cout << "Max Threads per SM: " << prop.maxThreadsPerMultiProcessor
            << std::endl;

    const char *path = "./data/mk01.fjs";

    if (argc >= 2) {
        path = argv[1];
    }
    parse_input(path);

    std::cout << "total_jobs: " << total_jobs << "\n";
    std::cout << "total_machines: " << total_machines << "\n";
    std::cout << "INDIVIDUAL_LEN: " << INDIVIDUAL_LEN << "\n";

    std::cout << "input data:\n";

    for (int id_job = 0; id_job < total_jobs; id_job++) {
        std::cout << "[Job " << id_job << "] ";
        for (int id_step = 0; id_step < input_data[id_job].len; id_step++) {
            std::cout << id_step << ": ";
            for (int id_operation = 0;
                    id_operation < input_data[id_job].steps[id_step].len;
                    id_operation++) {
                std::cout << "("
                        << input_data[id_job].steps[id_step].candidates[id_operation].id_machine
                        << ", "
                        << input_data[id_job].steps[id_step].candidates[id_operation].processing_time
                        << ") ";
            }
        }
        std::cout << "\n";
    }

    Job *jobs;
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&jobs, MAX_JOBS * sizeof(Job)));
    CUDA_CHECK_RETURN(
            cudaMemcpy(jobs, input_data, MAX_JOBS * sizeof(Job),
                    cudaMemcpyHostToDevice));

    thrust::device_vector<Gene> population(POPULATION_SIZE * INDIVIDUAL_LEN);
    thrust::device_vector<int> scores(POPULATION_SIZE);
    thrust::device_vector<Gene> new_population(
            POPULATION_SIZE * INDIVIDUAL_LEN);

    Gene *pop_ptr = thrust::raw_pointer_cast(&population[0]);
    Gene *new_pop_ptr = thrust::raw_pointer_cast(&new_population[0]);
    int *scores_ptr = thrust::raw_pointer_cast(&scores[0]);

    curandState_t *parent_candidates_states;
    CUDA_CHECK_RETURN(
            cudaMalloc((void ** )&parent_candidates_states,
                    POPULATION_SIZE * SIZE_PARENT_POOL
                            * sizeof(curandState_t)));

    curandState_t *population_states;
    CUDA_CHECK_RETURN(
            cudaMalloc((void ** )&population_states,
                    POPULATION_SIZE * sizeof(curandState_t)));

    // Parent candidate indexes
    int *parent_candidates;
    CUDA_CHECK_RETURN(
            cudaMalloc((void ** )&parent_candidates,
                    POPULATION_SIZE * SIZE_PARENT_POOL * sizeof(int)));

    // Picked parent indexes
    int *parents;
    CUDA_CHECK_RETURN(
            cudaMalloc((void ** )&parents, POPULATION_SIZE * sizeof(int)));

    init_rand_kernel<<<POPULATION_SIZE, 1>>>(population_states, time(0));
    CUDA_CHECK_RETURN(cudaPeekAtLastError());

    init_rand_kernel<<<POPULATION_SIZE * SIZE_PARENT_POOL, 1>>>(
            parent_candidates_states, time(0));
    CUDA_CHECK_RETURN(cudaPeekAtLastError());

    init_population_kernel<<<TOTALTHREADS, BLOCKSIZE>>>(pop_ptr,
            POPULATION_SIZE, INDIVIDUAL_LEN, jobs, total_jobs,
            population_states);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());

    stage_1_evaluate_kernel<<<TOTALTHREADS, BLOCKSIZE>>>(scores_ptr, pop_ptr,
            POPULATION_SIZE, INDIVIDUAL_LEN, jobs);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());

    int stage_1 = 3000;

    while (stage_1--) {

        fill_rand_kernel<<<POPULATION_SIZE * SIZE_PARENT_POOL, 1>>>(
                parent_candidates, POPULATION_SIZE * SIZE_PARENT_POOL,
                POPULATION_SIZE, parent_candidates_states);
        CUDA_CHECK_RETURN(cudaPeekAtLastError());

        pick_parents_kernel<<<TOTALTHREADS, BLOCKSIZE>>>(parents,
                parent_candidates, scores_ptr, POPULATION_SIZE);
        CUDA_CHECK_RETURN(cudaPeekAtLastError());

        stage_1_breed_kernel<<<TOTALTHREADS, BLOCKSIZE>>>(parents, pop_ptr,
                new_pop_ptr, POPULATION_SIZE, INDIVIDUAL_LEN, jobs,
                population_states);
        CUDA_CHECK_RETURN(cudaPeekAtLastError());

        thrust::copy(thrust::device, new_population.begin(),
                new_population.end(), population.begin());

        stage_1_evaluate_kernel<<<TOTALTHREADS, BLOCKSIZE>>>(scores_ptr,
                pop_ptr, POPULATION_SIZE, INDIVIDUAL_LEN, jobs);
        CUDA_CHECK_RETURN(cudaPeekAtLastError());

        if (stage_1 % 100 == 0) {
            int min_score = *thrust::min_element(scores.begin(), scores.end());
            std::cout << "stage_1: " << stage_1 << " score: " << min_score
                    << std::endl;
        }
    }

    int stage_2 = 2000;

    while (stage_2--) {
        fill_rand_kernel<<<POPULATION_SIZE * SIZE_PARENT_POOL, 1>>>(
                parent_candidates, POPULATION_SIZE * SIZE_PARENT_POOL,
                POPULATION_SIZE, parent_candidates_states);
        CUDA_CHECK_RETURN(cudaPeekAtLastError());

        pick_parents_kernel<<<TOTALTHREADS, BLOCKSIZE>>>(parents,
                parent_candidates, scores_ptr, POPULATION_SIZE);
        CUDA_CHECK_RETURN(cudaPeekAtLastError());

        stage_2_breed_kernel<<<TOTALTHREADS, BLOCKSIZE>>>(parents, pop_ptr,
                new_pop_ptr, POPULATION_SIZE, INDIVIDUAL_LEN, jobs,
                population_states);
        CUDA_CHECK_RETURN(cudaPeekAtLastError());

        thrust::copy(thrust::device, new_population.begin(),
                new_population.end(), population.begin());

        stage_2_evaluate_kernel<<<TOTALTHREADS, BLOCKSIZE>>>(scores_ptr,
                pop_ptr, POPULATION_SIZE, INDIVIDUAL_LEN, jobs);
        CUDA_CHECK_RETURN(cudaPeekAtLastError());

        if (stage_2 % 100 == 0) {
            int min_score = *thrust::min_element(scores.begin(), scores.end());
            std::cout << "stage_2: " << stage_2 << " score: " << min_score
                    << std::endl;
        }
    }

    auto min_iter = thrust::min_element(scores.begin(), scores.end());

    int index = min_iter - scores.begin();

    std::cout << "Done" << std::endl;

    std::cout << "Best solution score: " << scores[index] << std::endl;

    for (int i = 0; i < INDIVIDUAL_LEN; i++) {
        std::cout << population[index * INDIVIDUAL_LEN + i] << " ";
    }
    std::cout << std::endl;

    CUDA_CHECK_RETURN(cudaFree(parent_candidates_states));
    CUDA_CHECK_RETURN(cudaFree(population_states));
    CUDA_CHECK_RETURN(cudaFree(parent_candidates));
    CUDA_CHECK_RETURN(cudaFree(parents));
    CUDA_CHECK_RETURN(cudaFree(jobs));

    return 0;
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux(const char *file, unsigned line,
        const char *statement, cudaError_t err) {
    if (err == cudaSuccess)
        return;
    std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
            << err << ") at " << file << ":" << line << std::endl;
    exit(1);
}
