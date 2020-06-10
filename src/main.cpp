/*
 ============================================================================
 Name        : main.cpp
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
#include <vector>
#include <cstring>
#include <algorithm>

const int MAX_OPERATIONS_PER_STEP = 5;
const int MAX_STEPS_PER_JOB = 20;
const int MAX_JOBS = 20;
const int MAX_MACHINES = 20;

int POPULATION_SIZE = 4000;
int INDIVIDUAL_LEN = 20; // TODO
const int SIZE_PARENT_POOL = 7;

int TOTALTHREADS = 2048;
int BLOCKSIZE = 256;

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

void fill_rand_kernel(int *numbers, int len, int max_value, int idx) {
    numbers[idx] = rand() % max_value;
}

void init_population_kernel(Gene *population, int population_size,
        int individual_len, Job *jobs, int total_jobs, int index) {

    int next_step[MAX_JOBS];
    int i = index;

    int cursor = 0;
    Gene *me = population + i * individual_len;
    memset(next_step, 0, sizeof(next_step));
    while (cursor < individual_len) {
        int id_job = rand() % total_jobs;
        if (next_step[id_job] < jobs[id_job].len) {
            me[cursor].id_job = id_job;
            me[cursor].id_step = next_step[id_job];
            next_step[id_job]++;
            cursor++;
        }
    }

}

void pick_parents_kernel(int *parents, int *parent_candidates, int *scores,
        int population_size, int index) {

    int i = index;
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

void assignment_crossover(Gene *child, Gene *parent_a, Gene *parent_b,
        int individual_len, Job *jobs) {
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

void sequencing_crossover(Gene *child, Gene *parent_a, Gene *parent_b,
        int individual_len, Job *jobs) {
    int crossover_point = rand() % individual_len;

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

void assignment_mutation(Gene *individual, int individual_len, Job *jobs) {
    int count = 5;
    while (count--) {
        int mutation_point = rand() % individual_len;
        int id_job = individual[mutation_point].id_job;
        int id_step = individual[mutation_point].id_step;
        int len = jobs[id_job].steps[id_step].len;

        int id_operation = rand() % len;

        individual[mutation_point].id_operation = id_operation;
        individual[mutation_point].id_machine =
                jobs[id_job].steps[id_step].candidates[id_operation].id_machine;
    }

}

void swapping_mutation(Gene *individual, int individual_len, Job *jobs) {
    int count = 5;
    while (count--) {
        int mutation_point = rand() % (individual_len - 1);

        if (individual[mutation_point].id_job
                != individual[mutation_point + 1].id_job) {
            std::swap(individual[mutation_point],
                    individual[mutation_point + 1]);
        }
    }
}

void stage_1_breed_kernel(int *parents, Gene *population, Gene *new_population,
        int population_size, int individual_len, Job *jobs, int index) {

    int i = index;
    if (i < population_size * 8 / 10) {
        sequencing_crossover(&new_population[i * individual_len],
                &population[parents[i] * individual_len],
                &population[parents[i + 1] * individual_len], individual_len,
                jobs);
    } else {
        for (int s = 0; s < individual_len; s++) {
            new_population[i * individual_len + s] = population[parents[i]
                    * individual_len + s];
        }

        swapping_mutation(&new_population[i * individual_len], individual_len,
                jobs);
    }

}

void stage_2_breed_kernel(int *parents, Gene *population, Gene *new_population,
        int population_size, int individual_len, Job *jobs, int index) {
    int i = index;
    if (i < population_size * 4 / 10) {
        assignment_crossover(&new_population[i * individual_len],
                &population[parents[i] * individual_len],
                &population[parents[i + 1] * individual_len], individual_len,
                jobs);
    } else if (i < population_size * 8 / 10) {
        sequencing_crossover(&new_population[i * individual_len],
                &population[parents[i] * individual_len],
                &population[parents[i + 1] * individual_len], individual_len,
                jobs);
    } else {
        for (int s = 0; s < individual_len; s++) {
            new_population[i * individual_len + s] = population[parents[i]
                    * individual_len + s];
        }
        if (i < population_size * 9 / 10) {
            assignment_mutation(&new_population[i * individual_len],
                    individual_len, jobs);
        } else {
            swapping_mutation(&new_population[i * individual_len],
                    individual_len, jobs);

        }
    }
}

void stage_1_evaluate_kernel(int *scores, Gene *population, int population_size,
        int individual_len, Job *jobs, int index) {
    int value;
    int machines[MAX_MACHINES];
    int last_step_id_machine[MAX_JOBS];

    if (index < population_size) {
        int i = index;
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

void stage_2_evaluate_kernel(int *scores, Gene *population, int population_size,
        int individual_len, Job *jobs, int index) {
    int value;
    int machines[MAX_MACHINES];
    int last_step_id_machine[MAX_JOBS];

    if (index < population_size) {
        int i = index;
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

            if (id_step > 0) {
                int previous_id_machine = last_step_id_machine[id_job];
                if (machines[id_machine] < machines[previous_id_machine]) {
                    machines[id_machine] = machines[previous_id_machine];
                }
            }

            machines[id_machine] += processing_time;
            if (machines[id_machine] > value) {
                value = machines[id_machine];
            }

            last_step_id_machine[id_job] = id_machine;
        }

        scores[i] = value;
    }
}

int main(int argc, const char *argv[]) {
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

    Job *jobs = input_data;

    std::vector<Gene> population(POPULATION_SIZE * INDIVIDUAL_LEN);
    std::vector<int> scores(POPULATION_SIZE);
    std::vector<Gene> new_population(POPULATION_SIZE * INDIVIDUAL_LEN);

    Gene *pop_ptr = &population[0];
    Gene *new_pop_ptr = &new_population[0];
    int *scores_ptr = &scores[0];

    // Parent candidate indexes
    int *parent_candidates = new int[POPULATION_SIZE * SIZE_PARENT_POOL];

    // Picked parent indexes
    int *parents = new int[POPULATION_SIZE];

    for (int i = 0; i < POPULATION_SIZE; i++) {
        init_population_kernel(pop_ptr, POPULATION_SIZE, INDIVIDUAL_LEN, jobs,
                total_jobs, i);
    }

    for (int i = 0; i < POPULATION_SIZE; i++) {
        stage_1_evaluate_kernel(scores_ptr, pop_ptr, POPULATION_SIZE,
                INDIVIDUAL_LEN, jobs, i);
    }

    int stage_1 = 3000;

    while (stage_1--) {

        for (int i = 0; i < POPULATION_SIZE * SIZE_PARENT_POOL; i++) {
            fill_rand_kernel(parent_candidates,
                    POPULATION_SIZE * SIZE_PARENT_POOL, POPULATION_SIZE, i);
        }

        for (int i = 0; i < POPULATION_SIZE; i++) {
            pick_parents_kernel(parents, parent_candidates, scores_ptr,
                    POPULATION_SIZE, i);
        }

        for (int i = 0; i < POPULATION_SIZE; i++) {
            stage_1_breed_kernel(parents, pop_ptr, new_pop_ptr, POPULATION_SIZE,
                    INDIVIDUAL_LEN, jobs, i);
        }

        std::copy(new_population.begin(), new_population.end(),
                population.begin());

        for (int i = 0; i < POPULATION_SIZE; i++) {
            stage_1_evaluate_kernel(scores_ptr, pop_ptr, POPULATION_SIZE,
                    INDIVIDUAL_LEN, jobs, i);
        }

        if (stage_1 % 100 == 0) {
            int min_score = *std::min_element(scores.begin(), scores.end());
            std::cout << "stage_1: " << stage_1 << " score: " << min_score
                    << std::endl;
        }
    }

    int stage_2 = 2000;

    while (stage_2--) {
        for (int i = 0; i < POPULATION_SIZE * SIZE_PARENT_POOL; i++) {
            fill_rand_kernel(parent_candidates,
                    POPULATION_SIZE * SIZE_PARENT_POOL, POPULATION_SIZE, i);
        }

        for (int i = 0; i < POPULATION_SIZE; i++) {
            pick_parents_kernel(parents, parent_candidates, scores_ptr,
                    POPULATION_SIZE, i);
        }

        for (int i = 0; i < POPULATION_SIZE; i++) {
            stage_2_breed_kernel(parents, pop_ptr, new_pop_ptr, POPULATION_SIZE,
                    INDIVIDUAL_LEN, jobs, i);
        }

        std::copy(new_population.begin(), new_population.end(),
                population.begin());

        for (int i = 0; i < POPULATION_SIZE; i++) {
            stage_2_evaluate_kernel(scores_ptr, pop_ptr, POPULATION_SIZE,
                    INDIVIDUAL_LEN, jobs, i);
        }

        if (stage_2 % 100 == 0) {
            int min_score = *std::min_element(scores.begin(), scores.end());
            std::cout << "stage_2: " << stage_2 << " score: " << min_score
                    << std::endl;
        }
    }

    auto min_iter = std::min_element(scores.begin(), scores.end());

    int index = min_iter - scores.begin();

    std::cout << "Done" << std::endl;

    std::cout << "Best solution score: " << scores[index] << std::endl;

    for (int i = 0; i < INDIVIDUAL_LEN; i++) {
        std::cout << population[index * INDIVIDUAL_LEN + i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
