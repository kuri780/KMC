#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <mpi.h>

// Constants and Parameters
#define LATTICE_SIZE 300
#define N_CATEGORIES 5
#define MAX_STEPS 10000000
#define MAX_TIME 100.0


// Structure to represent a pair of indices (i, j)
typedef struct {
    int i;
    int j;
} pair_t;

// Structure for dynamic category lists
typedef struct {
    pair_t *pairs;      // Dynamic array of pairs
    int size;           // Current number of elements
    int capacity;       // Current capacity of the array
} category_t;

// Structure to hold process information
typedef struct {
    int start_row;
    int end_row;
    int rank;
    int size;
    double local_time;
    double next_event_time;
} ProcessInfo;

// Function Prototypes
void initialize_categories(category_t categories[]);
void free_categories(category_t categories[]);
void add_to_category(int n, int i, int j, category_t categories[], int category_indices[][LATTICE_SIZE]);
void remove_from_category(int n, int i, int j, category_t categories[], int category_indices[][LATTICE_SIZE]);
int count_neighbors(int i, int j, int lattice[][LATTICE_SIZE]);
double desorption_rate(int n, double k_d0, double E_d, double E_n, double k_B, double T);
double migration_rate(int n, double k_m0, double E_s, double E_n, double k_B, double T);
double rand_uniform();
int select_random_event(double desorption, double migration, double adsorption, double total_rate);
int main(int argc, char *argv[]) {
    // 添加时间变量
    double start_time, end_time;
    double temp_start, temp_end;      // 临时计时变量
    
    // MPI initialization
    int rank, size;
    MPI_Init(&argc, &argv);
    start_time = MPI_Wtime();
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Ensure the program is run with exactly 4 processes
    if (size != 4) {
        if (rank == 0) {
            printf("This program must be run with exactly 4 processes.\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Set process information
    ProcessInfo proc_info;
    proc_info.rank = rank;
    proc_info.size = size;
    proc_info.start_row = (LATTICE_SIZE / size) * rank;
    proc_info.end_row = (rank == size - 1) ? LATTICE_SIZE : (LATTICE_SIZE / size) * (rank + 1);
    proc_info.local_time = 0.0;
    proc_info.next_event_time = 0.0;

    // Seed the random number generator
    srand((unsigned int)time(NULL) + rank);

    // Parameters
    const double T = 610.0;  // Temperature (K)
    const double k_B = 8.617e-5;  // Boltzmann constant in eV/K
    const double E_d = 1.8;   // Desorption energy (eV)
    const double E_s = 1.58;  // Migration energy (eV)
    const double E_n = 0.28;  // Interaction energy (eV)
    const double h = 4.135667696e-15;  // Planck constant (eV*s)

    // Rate constants
    const double k_d0 = 2.0 * k_B * T / h;
    const double k_m0 = 2.0 * k_B * T / h; 

    // Lattice Initialization
    int lattice[LATTICE_SIZE][LATTICE_SIZE];
    memset(lattice, 0, sizeof(lattice));

    // Category Management
    category_t categories[N_CATEGORIES];
    initialize_categories(categories);

    // Auxiliary arrays to keep track of each site's category and its index within the category
    int site_categories[LATTICE_SIZE][LATTICE_SIZE];
    int category_indices[LATTICE_SIZE][LATTICE_SIZE];
    memset(site_categories, -1, sizeof(site_categories));
    memset(category_indices, -1, sizeof(category_indices));

    // Initialize categories based on initial lattice
    for (int i = proc_info.start_row; i < proc_info.end_row; i++) {
        for (int j = 0; j < LATTICE_SIZE; j++) {
            lattice[i][j] = 0;
        }
    }

    for (int i = proc_info.start_row; i < proc_info.end_row; i++) {
        for (int j = 0; j < LATTICE_SIZE; j++) {
            int n_neighbors = count_neighbors(i, j, lattice);
            if (n_neighbors >= 0 && n_neighbors < N_CATEGORIES) {
                add_to_category(n_neighbors, i, j, categories, category_indices);
                site_categories[i][j] = n_neighbors;
            }
        }
    }

    // Precompute desorption and migration rates for each category
    double desorption_rates[N_CATEGORIES];
    double migration_rates[N_CATEGORIES];
    for (int n = 0; n < N_CATEGORIES; n++) {
        desorption_rates[n] = desorption_rate(n, k_d0, E_d, E_n, k_B, T);
        migration_rates[n] = migration_rate(n, k_m0, E_s, E_n, k_B, T);
    }

    // Simulation Variables
    unsigned long long step;

    // 在主循环外定义总计时变量
    double total_simulation_time = 0.0;

    // 在main函数开始处添加计数器
    int boundary_changes = 0;  // 记录边界变化次数
    int communication_interval = 10;  // 设置通信间隔

    // Simulation Loop
    for (step = 0; step < MAX_STEPS; step++) {
        // 仿真部分开始
        double sim_start = MPI_Wtime();

        // Calculate event rates
        double total_desorption = 0.0;
        double total_migration = 0.0;
        for (int n = 0; n < N_CATEGORIES; n++) {
            total_desorption += desorption_rates[n] * categories[n].size;
            total_migration += migration_rates[n] * categories[n].size;
        }

        
        double adsorption_rate = (double)((proc_info.end_row - proc_info.start_row) * LATTICE_SIZE);
        double total_rate = total_desorption + total_migration + adsorption_rate;

        // 事件选择
        char event_type[20];
        double r = rand_uniform() * total_rate;
        if (r < total_desorption) {
            strcpy(event_type, "desorption");
        } else if (r < total_desorption + total_migration) {
            strcpy(event_type, "migration");
        } else {
            strcpy(event_type, "adsorption");
        }

        // 时间同步
        double local_time_increment = -log(rand_uniform()) / total_rate;
        proc_info.local_time += local_time_increment;

        // 标记是否需要通信（边界是否发生变化）
        int need_communication = 0; // 默认不通信
        // 记录事件发生的位置
        int event_i = -1, event_j = -1;
        int affected_neighbor_i = -1; // 记录可能受影响的邻居行

        // 事件选择和执行逻辑
        if (strcmp(event_type, "adsorption") == 0) {
            // Adsorption Event
            // Select a random site for adsorption
            int i = proc_info.start_row + (rand() % (proc_info.end_row - proc_info.start_row));
            int j = rand() % LATTICE_SIZE;
            
            event_i = i;
            event_j = j;

            // Perform adsorption
            lattice[i][j]++;

            // Update category of the affected site
            int old_category = site_categories[i][j];
            int new_n_neighbors = count_neighbors(i, j, lattice);
            if (new_n_neighbors >= 0 && new_n_neighbors < N_CATEGORIES) {
                if (old_category != new_n_neighbors) {
                    if (old_category != -1) {
                        remove_from_category(old_category, i, j, categories, category_indices);
                    }
                    add_to_category(new_n_neighbors, i, j, categories, category_indices);
                    site_categories[i][j] = new_n_neighbors;
                }
            }

            // Update categories of neighboring sites
            int directions[4][2] = { {-1,0}, {1,0}, {0,-1}, {0,1} };
            for (int d = 0; d < 4; d++) {
                int ni = (i + directions[d][0] + LATTICE_SIZE) % LATTICE_SIZE;
                int nj = (j + directions[d][1] + LATTICE_SIZE) % LATTICE_SIZE;
                
                // 记录可能受影响的邻居行
                if (directions[d][0] != 0) { // 只关注上下方向的邻居
                    affected_neighbor_i = ni;
                }
                
                int old_cat = site_categories[ni][nj];
                int new_cat = count_neighbors(ni, nj, lattice);
                if (new_cat >= 0 && new_cat < N_CATEGORIES) {
                    if (old_cat != new_cat) {
                        if (old_cat != -1) {
                            remove_from_category(old_cat, ni, nj, categories, category_indices);
                        }
                        add_to_category(new_cat, ni, nj, categories, category_indices);
                        site_categories[ni][nj] = new_cat;
                    }
                }
            }
        }
        else {
            // Desorption or Migration Event
            int event_category = -1;
            if (strcmp(event_type, "desorption") == 0) {
                // Select category based on desorption rates
                double cumulative = 0.0;
                double target = rand_uniform() * total_desorption;
                for (int n = 0; n < N_CATEGORIES; n++) {
                    cumulative += desorption_rates[n] * categories[n].size;
                    if (cumulative >= target) {
                        event_category = n;
                        break;
                    }
                }
                if (event_category == -1) event_category = N_CATEGORIES - 1;
            }
            else if (strcmp(event_type, "migration") == 0) {
                // Select category based on migration rates
                double cumulative = 0.0;
                double target = rand_uniform() * total_migration;
                for (int n = 0; n < N_CATEGORIES; n++) {
                    cumulative += migration_rates[n] * categories[n].size;
                    if (cumulative >= target) {
                        event_category = n;
                        break;
                    }
                }
                if (event_category == -1) event_category = N_CATEGORIES - 1;
            }

            if (event_category != -1 && categories[event_category].size > 0) {
                // Select a random site from the chosen category
                int index = rand() % categories[event_category].size;
                pair_t selected = categories[event_category].pairs[index];
                int i = selected.i;
                int j = selected.j;

                event_i = i;
                event_j = j;
                
                // 记录受影响的位置
                affected_neighbor_i = i;

                if (strcmp(event_type, "desorption") == 0 && lattice[i][j] > 0) {
                    // Perform desorption
                    lattice[i][j]--;

                    // Update category of the affected site
                    int old_category = site_categories[i][j];
                    int new_n_neighbors = count_neighbors(i, j, lattice);
                    if (new_n_neighbors >= 0 && new_n_neighbors < N_CATEGORIES) {
                        if (old_category != new_n_neighbors) {
                            if (old_category != -1) {
                                remove_from_category(old_category, i, j, categories, category_indices);
                            }
                            add_to_category(new_n_neighbors, i, j, categories, category_indices);
                            site_categories[i][j] = new_n_neighbors;
                        }
                    }

                    // Update categories of neighboring sites
                    int directions[4][2] = { {-1,0}, {1,0}, {0,-1}, {0,1} };
                    for (int d = 0; d < 4; d++) {
                        int ni = (i + directions[d][0] + LATTICE_SIZE) % LATTICE_SIZE;
                        int nj = (j + directions[d][1] + LATTICE_SIZE) % LATTICE_SIZE;
                        int old_cat = site_categories[ni][nj];
                        int new_cat = count_neighbors(ni, nj, lattice);
                        if (new_cat >= 0 && new_cat < N_CATEGORIES) {
                            if (old_cat != new_cat) {
                                if (old_cat != -1) {
                                    remove_from_category(old_cat, ni, nj, categories, category_indices);
                                }
                                add_to_category(new_cat, ni, nj, categories, category_indices);
                                site_categories[ni][nj] = new_cat;
                            }
                        }
                    }
                }
                else if (strcmp(event_type, "migration") == 0 && lattice[i][j] > 0) {
                    // Determine possible migration directions (to lower height)
                    int directions[4][2] = { {-1,0}, {1,0}, {0,-1}, {0,1} };
                    pair_t possible_dirs[4];
                    int dir_count = 0;
                    for (int d = 0; d < 4; d++) {
                        int ni = (i + directions[d][0] + LATTICE_SIZE) % LATTICE_SIZE;
                        int nj = (j + directions[d][1] + LATTICE_SIZE) % LATTICE_SIZE;
                        if (lattice[ni][nj] < lattice[i][j]) {
                            possible_dirs[dir_count].i = directions[d][0];
                            possible_dirs[dir_count].j = directions[d][1];
                            dir_count++;
                        }
                    }

                    if (dir_count > 0) {
                        // Select a random migration direction
                        int selected_dir = rand() % dir_count;
                        int di = possible_dirs[selected_dir].i;
                        int dj = possible_dirs[selected_dir].j;
                        int ni = (i + di + LATTICE_SIZE) % LATTICE_SIZE;
                        int nj = (j + dj + LATTICE_SIZE) % LATTICE_SIZE;

                        // Move the particle
                        lattice[i][j]--;
                        lattice[ni][nj]++;

                        // Update category of the original site
                        int old_category_orig = site_categories[i][j];
                        int new_n_neighbors_orig = count_neighbors(i, j, lattice);
                        if (new_n_neighbors_orig >= 0 && new_n_neighbors_orig < N_CATEGORIES) {
                            if (old_category_orig != new_n_neighbors_orig) {
                                if (old_category_orig != -1) {
                                    remove_from_category(old_category_orig, i, j, categories, category_indices);
                                }
                                add_to_category(new_n_neighbors_orig, i, j, categories, category_indices);
                                site_categories[i][j] = new_n_neighbors_orig;
                            }
                        }

                        // Update category of the destination site
                        int old_category_dest = site_categories[ni][nj];
                        int new_n_neighbors_dest = count_neighbors(ni, nj, lattice);
                        if (new_n_neighbors_dest >= 0 && new_n_neighbors_dest < N_CATEGORIES) {
                            if (old_category_dest != new_n_neighbors_dest) {
                                if (old_category_dest != -1) {
                                    remove_from_category(old_category_dest, ni, nj, categories, category_indices);
                                }
                                add_to_category(new_n_neighbors_dest, ni, nj, categories, category_indices);
                                site_categories[ni][nj] = new_n_neighbors_dest;
                            }
                        }

                        // Update categories of neighboring sites for both original and destination
                        int migration_affected_sites[8][2] = {
                            {(i - 1 + LATTICE_SIZE) % LATTICE_SIZE, j},
                            {(i + 1) % LATTICE_SIZE, j},
                            {i, (j - 1 + LATTICE_SIZE) % LATTICE_SIZE},
                            {i, (j + 1) % LATTICE_SIZE},
                            {(ni - 1 + LATTICE_SIZE) % LATTICE_SIZE, nj},
                            {(ni + 1) % LATTICE_SIZE, nj},
                            {ni, (nj - 1 + LATTICE_SIZE) % LATTICE_SIZE},
                            {ni, (nj + 1) % LATTICE_SIZE}
                        };
                        for (int a = 0; a < 8; a++) {
                            int ai = migration_affected_sites[a][0];
                            int aj = migration_affected_sites[a][1];
                            int old_cat = site_categories[ai][aj];
                            int new_cat = count_neighbors(ai, aj, lattice);
                            if (new_cat >= 0 && new_cat < N_CATEGORIES) {
                                if (old_cat != new_cat) {
                                    if (old_cat != -1) {
                                        remove_from_category(old_cat, ai, aj, categories, category_indices);
                                    }
                                    add_to_category(new_cat, ai, aj, categories, category_indices);
                                    site_categories[ai][aj] = new_cat;
                                }
                            }
                            
                            // 检查数组是否已满
                            if (affected_neighbor_i == -1) {
                                affected_neighbor_i = ai;
                            }
                        }
                    }
                }
            }
        }

        // 简化的边界检查
        if (event_i == proc_info.start_row || event_i == proc_info.start_row + 1 || 
            event_i == proc_info.end_row - 1 || event_i == proc_info.end_row - 2) {
            need_communication = 1;
        }
        // 检查受影响的邻居是否在边界附近
        if (affected_neighbor_i == proc_info.start_row - 1 || affected_neighbor_i == proc_info.end_row) {
            need_communication = 1;
        }

        // 如果需要通信，增加计数器
        if (need_communication) {
            boundary_changes++;
        }

        // 只有当累积的边界变化达到阈值时才实际进行通信
        need_communication = (boundary_changes >= communication_interval);

        // 如果决定通信，重置计数器
        if (need_communication) {
            boundary_changes = 0;
        }

        // 仿真部分结束
        double sim_end = MPI_Wtime();
        total_simulation_time += (sim_end - sim_start);
        
        // 时间同步 - 这部分应该在仿真时间计时之外
        if (rank == 0) {
            double root_time = proc_info.local_time;
            MPI_Bcast(&root_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        } else {
            double root_time;
            MPI_Bcast(&root_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            proc_info.local_time = root_time;
        }

        // 只在本地需要通信时才参与全局同步
        int global_need_communication = 0;
        if (need_communication) {
            // 使用MPI_Allreduce来确定是否有任何进程需要通信
            MPI_Allreduce(&need_communication, &global_need_communication, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        } else {
            // 如果本地不需要通信，仍然需要参与集体操作
            // 但使用MPI_Allreduce的MPI_IN_PLACE选项减少数据传输
            MPI_Allreduce(MPI_IN_PLACE, &global_need_communication, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        }

        // 只在需要时进行通信
        if (global_need_communication) {
            // 同步边界数据
            if (rank > 0) {
                MPI_Request requests[2];
                MPI_Isend(&lattice[proc_info.start_row][0], LATTICE_SIZE, MPI_INT, rank-1, 0, MPI_COMM_WORLD, &requests[0]);
                MPI_Irecv(&lattice[proc_info.start_row-1][0], LATTICE_SIZE, MPI_INT, rank-1, 0, MPI_COMM_WORLD, &requests[1]);
                MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
            }
            
            if (rank < size-1) {
                MPI_Recv(&lattice[proc_info.end_row][0], LATTICE_SIZE, MPI_INT, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(&lattice[proc_info.end_row-1][0], LATTICE_SIZE, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }

        // 检查终止条件
        if (proc_info.local_time >= MAX_TIME) {
            break;
        }
    }

    // Final Calculations
    // 首先计算每个进程负责部分的统计数据
    double local_sum = 0.0;
    for (int i = proc_info.start_row; i < proc_info.end_row; i++) {
        for (int j = 0; j < LATTICE_SIZE; j++) {
            local_sum += lattice[i][j];
        }
    }

    // 收集所有进程的总和
    double global_sum;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // 计算全局平均高度
    double global_mean_height = 0.0;
    if (rank == 0) {
        global_mean_height = global_sum / (LATTICE_SIZE * LATTICE_SIZE);
    }

    // 广播全局平均高度给所有进程
    MPI_Bcast(&global_mean_height, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 计算局部RMS粗糙度
    double local_roughness_sum = 0.0;
    for (int i = proc_info.start_row; i < proc_info.end_row; i++) {
        for (int j = 0; j < LATTICE_SIZE; j++) {
            double diff = lattice[i][j] - global_mean_height;
            local_roughness_sum += diff * diff;
        }
    }

    // 收集所有进程的RMS粗糙度
    double global_roughness_sum;
    MPI_Reduce(&local_roughness_sum, &global_roughness_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // 只在rank 0进程输出最终结果
    if (rank == 0) {
        double global_rms_roughness = sqrt(global_roughness_sum / (LATTICE_SIZE * LATTICE_SIZE));
        
        printf("Simulation finished.\n");
        printf("Total simulation time: %.6f\n", proc_info.local_time);
        printf("Mean Height: %.6f\n", global_mean_height);
        printf("RMS Roughness: %.6f\n", global_rms_roughness);
    }

    // Free allocated memory
    free_categories(categories);

    // 每个进程计算自己的总时间和仿真时间
    end_time = MPI_Wtime();
    double proc_total_time = end_time - start_time;

    // 收集所有进程的总时间和仿真时间
    double all_times[2] = {proc_total_time, total_simulation_time};
    double max_times[2];
    MPI_Reduce(all_times, max_times, 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // 在rank 0进程上处理时间统计
    if (rank == 0) {
        double global_total_time = max_times[0];
        double global_sim_time = max_times[1];
        
        // 计算通信和其他开销的总和
        double global_comm_other_time = global_total_time - global_sim_time;
        if (global_comm_other_time < 0) global_comm_other_time = 0;
        
        // 计算百分比
        double sim_percent = (global_sim_time / global_total_time) * 100.0;
        double comm_other_percent = (global_comm_other_time / global_total_time) * 100.0;
        
        printf("\n程序性能统计:\n");
        printf("总运行时间: %.2f 秒\n", global_total_time);
        printf("KMC仿真时间: %.2f 秒 (%.1f%%)\n", global_sim_time, sim_percent);
        printf("通信和其他开销: %.2f 秒 (%.1f%%)\n", global_comm_other_time, comm_other_percent);
    }

    MPI_Finalize();
    return 0;
}

// Function to initialize categories
void initialize_categories(category_t categories[]) {
    for (int n = 0; n < N_CATEGORIES; n++) {
        categories[n].size = 0;
        categories[n].capacity = 1000;  // Initial capacity
        categories[n].pairs = (pair_t *)malloc(categories[n].capacity * sizeof(pair_t));
        if (categories[n].pairs == NULL) {
            fprintf(stderr, "Memory allocation failed for category %d\n", n);
            exit(EXIT_FAILURE);
        }
    }
}

// Function to free allocated memory for categories
void free_categories(category_t categories[]) {
    for (int n = 0; n < N_CATEGORIES; n++) {
        free(categories[n].pairs);
    }
}

// Function to add a site to a category
void add_to_category(int n, int i, int j, category_t categories[], int category_indices[][LATTICE_SIZE]) {
    if (n < 0 || n >= N_CATEGORIES) return;

    // Expand capacity if needed
    if (categories[n].size >= categories[n].capacity) {
        categories[n].capacity *= 2;
        pair_t *new_pairs = (pair_t *)realloc(categories[n].pairs, categories[n].capacity * sizeof(pair_t));
        if (new_pairs == NULL) {
            fprintf(stderr, "Memory reallocation failed for category %d\n", n);
            exit(EXIT_FAILURE);
        }
        categories[n].pairs = new_pairs;
    }

    // Add the new site
    categories[n].pairs[categories[n].size].i = i;
    categories[n].pairs[categories[n].size].j = j;
    category_indices[i][j] = categories[n].size;
    categories[n].size++;
}

// Function to remove a site from a category
void remove_from_category(int n, int i, int j, category_t categories[], int category_indices[][LATTICE_SIZE]) {
    if (n < 0 || n >= N_CATEGORIES) return;

    int index = category_indices[i][j];
    if (index == -1 || index >= categories[n].size) return;

    // Replace the removed element with the last element to maintain continuity
    categories[n].pairs[index] = categories[n].pairs[categories[n].size - 1];
    int swapped_i = categories[n].pairs[index].i;
    int swapped_j = categories[n].pairs[index].j;
    category_indices[swapped_i][swapped_j] = index;

    // Decrease the size and reset the index of the removed site
    categories[n].size--;
    category_indices[i][j] = -1;
}

// Function to count the number of neighbors with height >= current site
int count_neighbors(int i, int j, int lattice[][LATTICE_SIZE]) {
    int current_height = lattice[i][j];
    int count = 0;
    int directions[4][2] = { {-1,0}, {1,0}, {0,-1}, {0,1} };
    for (int d = 0; d < 4; d++) {
        int ni = (i + directions[d][0] + LATTICE_SIZE) % LATTICE_SIZE;
        int nj = (j + directions[d][1] + LATTICE_SIZE) % LATTICE_SIZE;
        if (lattice[ni][nj] >= current_height) {
            count++;
        }
    }
    return count;
}

// Function to calculate desorption rate
double desorption_rate(int n, double k_d0, double E_d, double E_n, double k_B, double T) {
    return k_d0 * exp(-(E_d + n * E_n) / (k_B * T));
}

// Function to calculate migration rate
double migration_rate(int n, double k_m0, double E_s, double E_n, double k_B, double T) {
    return k_m0 * exp(-(E_s + n * E_n) / (k_B * T));
}

// Function to generate a uniform random number between 0 and 1
double rand_uniform() {
    return ((double)rand()) / ((double)RAND_MAX);
}
