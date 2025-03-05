#include <iostream>
#include <csignal>
#include <algorithm>
#include <random>
#include <chrono>
#include <queue>

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "log.h"
#include "config.h"

using namespace std;

perms_config cfg;

#define SIGN(x) (x < 0 ? (-1) : 1)
#define ENFORCE_SAME(a, b, x, y) if (SIGN(perm[b] - perm[a]) != SIGN(cfg.pattern[y] - cfg.pattern[x])) continue
struct organism {
    int score = 0;
    ENTRY *perm;

    organism() {
        perm = new ENTRY[cfg.permutation_length];

        iota(&perm[0], &perm[cfg.permutation_length], 0);
        shuffle(&perm[0], &perm[cfg.permutation_length], cfg.gen);

        get_score();
    }
    organism(const organism *parentA, const organism *parentB) {
        perm = new ENTRY[cfg.permutation_length];
        bool *scratch = new bool[cfg.permutation_length];

        int n = cfg.rand() % N_CROSSOVER_FUNCS;
        while (!cfg.crossover_funcs[n]) {
            n = cfg.rand() % N_CROSSOVER_FUNCS;
        }

        switch (n) {
            // Cut-and-crossfill
            case 0: {
                for (int j = 0; j < cfg.permutation_length; j++) scratch[j] = false;

                int from = cfg.rand() % cfg.permutation_length;
                int to = cfg.rand() % cfg.permutation_length;
                while (from != to) {
                    ENTRY v = parentA->perm[from];
                    perm[from] = v;
                    scratch[v] = true;
                    from = (from + 1) % cfg.permutation_length;
                }

                for (int j = 0; j < cfg.permutation_length; j++) {
                    ENTRY v = parentB->perm[j];
                    if (scratch[v]) continue;
                    perm[from] = v;
                    scratch[v] = true;

                    from = (from + 1) % cfg.permutation_length;
                }
                break;
            }
            // Flip-and-scan
            case 1: {
                for (int j = 0; j < cfg.permutation_length; j++) scratch[j] = false;
                for (int j = 0; j < cfg.permutation_length; j++) {
                    const organism *parent = (cfg.rand() % 2) == 1 ? parentA : parentB;

                    int k = j;
                    while (scratch[parent->perm[k]]) k = (k + 1) % cfg.permutation_length;

                    ENTRY v = parent->perm[k];
                    perm[j] = v;
                    scratch[v] = true;
                }
                break;
            }
            // Flip-and-shift
            case 2: {
                pair<float, ENTRY> *pairs = new pair<float, ENTRY>[cfg.permutation_length];
                for (int j = 0; j < cfg.permutation_length; j++) {
                    if ((cfg.rand() % 2) == 1) pairs[j] = make_pair(
                        static_cast<float>(parentA->perm[j]) + 0.25f, j);
                    else pairs[j] = make_pair(
                        static_cast<float>(parentB->perm[j]) - 0.25f, j);
                }
                sort(&pairs[0], &pairs[cfg.permutation_length]);
                for (int j = 0; j < cfg.permutation_length; j++) {
                    perm[j] = pairs[j].second;
                }
                delete[] pairs;
                break;
            }
        }
        delete[] scratch;

        // Mutate
        if (cfg.max_mutations > 0) {
            /*
            int mutationCount = cfg.rand() % cfg.max_mutations;
            for (int j = 0; j < mutationCount; j++) {
                int a = cfg.rand() % cfg.permutation_length;
                int b = cfg.rand() % cfg.permutation_length;
                while (b == a) b = cfg.rand() % cfg.permutation_length;

                ENTRY tmp = perm[a];
                perm[a] = perm[b];
                perm[b] = tmp;
            }
            */

            int T = cfg.rand() % cfg.max_mutations;
            vector<pair<ENTRY, ENTRY>> output(cfg.permutation_length);
            for (int i = 0; i < cfg.permutation_length; i++) {
                int r1 = cfg.rand() % 10;
                if (r1 <= 1) {
                    int r2 = 5 - (cfg.rand() % 10);
                    output[i] = { (10 * perm[i]) + T * r2, i };
                } else {
                    output[i] = { 10 * perm[i], i };
                }
            }
            std::sort(
                output.begin(),
                output.end(),
                [](const pair<ENTRY, ENTRY> &a, const pair<ENTRY, ENTRY> &b) {
                    return a.first < b.first;
                }
            );
            for (int i = 0; i < cfg.permutation_length; i++) {
                perm[i] = output[i].second;
            }
        }
        get_score();
    }
    ~organism() {
        delete[] perm;
    }

    /*
    void get_score() {
        score = 0;
#pragma omp parallel for reduction(+:score)
        for (int j = 0; j < cfg.permutation_length; j++) {
            for (int k = j + 1; k < cfg.permutation_length; k++) {
                ENFORCE_SAME(j, k, 0, 1);
                if (cfg.pattern_length == 2) {
                    score++;
                    continue;
                }

                for (int l = k + 1; l < cfg.permutation_length; l++) {
                    ENFORCE_SAME(k, l, 1, 2);
                    ENFORCE_SAME(j, l, 0, 2);
                    if (cfg.pattern_length == 3) {
                        score++;
                        continue;
                    }

                    for (int m = l + 1; m < cfg.permutation_length; m++) {
                        ENFORCE_SAME(l, m, 2, 3);
                        ENFORCE_SAME(k, m, 1, 3);
                        ENFORCE_SAME(j, m, 0, 3);
                        score++;
                    }
                }
            }
        }
    }
    */

    void get_score() {
        score = 0;
        int n = cfg.permutation_length;
        vector<vector<ENTRY>> GT, LT;
        
        for (int i = 0; i < n; i++) {
            GT.push_back(vector<ENTRY>());
            LT.push_back(vector<ENTRY>());
        }

        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                if (perm[i] < perm[j]) {
                    GT[perm[i]].push_back(perm[j]);
                } else {
                    LT[perm[i]].push_back(perm[j]);
                }
            }
        }

        for (int i = 0; i < n - 1; i++) {
            for (const ENTRY &g : GT[perm[i]]) {
                int curr = 0,
                    total = 0;
                if (LT[perm[i]].size() <= 0) continue;

                int cnt = LT[perm[i]].size() - 1;
                for (int v = 0; v < LT[g].size(); v++) {
                    int w = LT[g][LT[g].size() - v - 1];
                    if (std::find(
                            GT[LT[perm[i]][cnt]].begin(),
                            GT[LT[perm[i]][cnt]].end(),
                            w
                        ) != GT[LT[perm[i]][cnt]].end()
                    ) {
                        curr++;
                    } else if (w == LT[perm[i]][cnt]) {
                        total += curr;
                        cnt--;
                    }
                }
                score += total;
            }
        }
    }

    void print() {
        for (int j = 0; j < cfg.permutation_length; j++) {
            printf("%i ", perm[j]);
        }
    }
};

void stop(int sig) {
    printf("\x1b[0m\x1b[?25h\nCtrl-C received, shutting down.\n");
    exit(0);
}

int main(int argc, char** argv) {
    std::signal(SIGINT, stop);
    std::signal(SIGTERM, stop);
    std::signal(SIGKILL, stop);

    if (argc < 2) {
        Log::die("Usage: perms [config file]");
    }
    cfg = perms_config(argv[1]);
    Log::info(string("Crossover functions:") +
        string(cfg.crossover_funcs[0] ? " cut-and-crossfill" : "") +
        string(cfg.crossover_funcs[1] ? " flip-and-scan" : "") +
        string(cfg.crossover_funcs[2] ? " flip-and-shift" : "")
    );
    Log::info("Pattern: " + to_string(cfg.raw));
    Log::info("Permutation length: " + to_string(cfg.permutation_length));
    Log::info("Fitness evals: " + (cfg.fitness_evals == INT_MAX ? "(infinite)" : to_string(cfg.fitness_evals)));
    Log::info("Max mutation count: " + to_string(cfg.max_mutations));
    if (cfg.view == FULL) printf("\x1b[2J\x1b[?25l");
    else if (cfg.view == SIMPLE) printf("\x1b[?25l");
    setvbuf(stdout, NULL, _IOFBF, 4096);

    chrono::time_point<chrono::system_clock> start, end;
    start = chrono::system_clock::now();
    vector<organism*> population(cfg.population_size, nullptr);
#pragma omp parallel for
    for (int i = 0; i < cfg.population_size; i++) {
        population[i] = new organism();
    }

    sort(population.begin(), population.end(), [](const organism *a, const organism *b) {
        return a->score > b->score;
    });
#pragma omp parallel for
    for (int i = cfg.parent_pool; i < cfg.population_size; i++) {
        delete population[i];
    }
    population.erase(
        population.begin() + cfg.parent_pool,
        population.end()
    );
    end = chrono::system_clock::now();

    chrono::duration<double> elapsed = end - start;
    printf("\x1b[36m[INFO] \x1b[0mGenerated initial parent population in %.2fs.\x1b[0m\n\n", elapsed.count());
    fflush(stdout);

    start = chrono::system_clock::now();
    int i = 0;
#pragma omp parallel
    {
        while (i < cfg.fitness_evals) {
            vector<organism*> local_population;
            for (int j = 0; j < cfg.batch_size; j++) {
                organism *A = population[cfg.rand() % cfg.parent_pool];
                organism *B = population[cfg.rand() % cfg.parent_pool];
                organism *C = new organism(A, B);
                local_population.push_back(C);
            }

#pragma omp critical
            {
                population.insert(population.end(), local_population.begin(), local_population.end());
                sort(population.begin(), population.end(), [](const organism *a, const organism *b) {
                    return a->score > b->score;
                });
                for (int j = cfg.population_size; j < population.size(); j++) {
                    delete population.back();
                    population.pop_back();
                }

                i += cfg.batch_size;
            }

            if (omp_get_thread_num() == 0) {
                end = chrono::system_clock::now();
                elapsed = end - start;
                double rate = static_cast<double>(i) / elapsed.count();
                organism *best = population[0];
                switch (cfg.view) {
                    case FULL: {
                        printf("\x1b[38;2;255;255;255m");
                        for (int j = 0; j < cfg.permutation_length; j++) {
                            printf("\x1b[%i;%iH\x1b[2K██", (cfg.permutation_length - best->perm[j]) + 1, (2 * j) + 1);
                        }
                        printf(
                            "\x1b[0m\x1b[%i;1H\x1b[34mPattern: \x1b[33m%i\x1b[0K\n\x1b[34mEvals: \x1b[33m%i\x1b[0K\n\x1b[34mFitness: \x1b[33m%i\x1b[0K\n\x1b[34mRate: \x1b[33m%.2f evals/sec\x1b[0K\n\x1b[34mPermutation: \x1b[33m",
                            cfg.permutation_length + 3,
                            cfg.raw,
                            i + 1,
                            best->score,
                            rate
                        );
                        best->print();
                        printf("\x1b[0K\n\x1b[34mOther scores: \x1b[33m");
                        for (int j = 1; j < cfg.parent_pool; j++) {
                            printf("%i ", population[j]->score);
                        }
                        printf("\x1b[0m\x1b[0K\n");
                        fflush(stdout);
                        break;
                    }
                    case SIMPLE: {
                        printf("\x1b[1E\x1b[2ABest fitness after \x1b[33m%i/%i\x1b[0m evals: \x1b[34m%i\x1b[35m (%.2f evals/sec)\x1b[0m\x1b[0K\n", i + 1, cfg.fitness_evals, best->score, rate);
                        fflush(stdout);
                        break;
                    }
                    case NONE: break;
                }
            }
        }
    }

    organism *best = population[0];
    printf("\x1b[36m[INFO] \x1b[0mBest fitness after \x1b[33m%i\x1b[0m evals: \x1b[34m%i\n\x1b[36m[INFO] \x1b[0mPermutation: \x1b[33m", cfg.fitness_evals, best->score);
    best->print();
    printf("\x1b[0m\x1b[?25h\n");
    fflush(stdout);

    for (organism *o : population) delete o;

    return 0;
}
