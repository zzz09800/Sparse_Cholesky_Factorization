#include <stdio.h>
#include <stdlib.h>
#include <mpi/mpi.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define MATRIX_SIZE 1000

/* Timing Guidelines:
 *  1. Small matrix (~600x600)
 *  2. Sparse enough (rand coefficienct = [3.5,4.5])
 *  3. Slow CPU, the slower the better.
 */

/*double mat_in[MATRIX_SIZE][MATRIX_SIZE] = {{  49,    0,    0,    0,    0,    0,  203,    0,    0,    0,    0,  105,    0,    0,    0,    0,   63,  189,    0,   63,    0,    0,    0,    0,    0,    0,    0,   21,   21,    0, },
                                           {   0,  841,  638,    0,    0,    0,  783,    0,    0,    0,    0,  319,  406,    0,    0,    0,    0,    0,    0,  551,    0,  290,    0,    0,    0,    0,    0,    0,    0,    0, },
                                           {   0,  638,  845,    0,  532,    0,  594,    0,  190,  380,  418,  755,  308,  342,    0,    0,    0,    0,    0,  418,    0,  220,    0,    0,  532,    0,    0,    0,    0,    0, },
                                           {   0,    0,    0,  625,    0,  350,  100,    0,    0,  475,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, },
                                           {   0,    0,  532,    0, 1625,    0,    0,    0,  280,  560,  616,  756,    0,  504,    0,    0,    0,    0,    0,   29,    0,    0,  464,    0,  784,    0,  783,   29,    0,    0, },
                                           {   0,    0,    0,  350,    0,  260,   56,    0,    0,  266,   16,    0,   64,    0,    0,    0,    0,  144,    0,    0,    0,  152,    0,    0,    0,    0,    0,    0,    0,  112, },
                                           { 203,  783,  594,  100,    0,   56, 2162,    0,    0,   76,  552, 1092,  378,    0,    0,    0,  261,  783,    0,  774,    0,  966,    0,    0,    0,    0,    0,   87,   87,    0, },
                                           {   0,    0,    0,    0,    0,    0,    0,  196,    0,    0,    0,    0,    0,    0,    0,  406,    0,    0,    0,    0,  238,  140,    0,    0,    0,    0,    0,    0,  294,   28, },
                                           {   0,    0,  190,    0,  280,    0,    0,    0,  541,  683,  220,  270,  441,  180,  126,  462,    0,  504,  567,    0,  252,    0,    0,    0,  322,    0,    0,    0,    0,    0, },
                                           {   0,    0,  380,  475,  560,  266,   76,    0,  683, 1434,  440,  540,  483,  360,  138,  506,    0,  552,  753,    0,  276,    0,    0,    0,  606,  264,    0,  336,  276,    0, },
                                           {   0,    0,  418,    0,  616,   16,  552,    0,  220,  440, 1081,  939,   16,  396,    0,   80,    0,   36,    0,    0,    0,  705,  128,    0,  616,   64,    0,    0,  152,   28, },
                                           { 105,  319,  755,    0,  756,    0, 1092,    0,  270,  540,  939, 1469,  492,  486,    0,    0,  135,  405,    0,  344,    0,  545,    0,    0,  756,    0,    0,   45,   45,  260, },
                                           {   0,  406,  308,    0,    0,   64,  378,    0,  441,  483,   16,  492, 1381,    0,  126,  462,   42,  648,  619,  270,  252,  292,    0,    0,   42,   24,    4,    0,    4,  632, },
                                           {   0,    0,  342,    0,  504,    0,    0,    0,  180,  360,  396,  486,    0,  808,    0,  110,    0,    0,    0,    0,    0,    0,  242,  220, 1142,  132,    0,    0,    0,    0, },
                                           {   0,    0,    0,    0,    0,    0,    0,    0,  126,  138,    0,    0,  126,    0,  820,  132,    0,  144,  162,    0,   72,    0,    0,    0,  600,    0,    0,    0,    0,    0, },
                                           {   0,    0,    0,    0,    0,    0,    0,  406,  462,  506,   80,    0,  462,  110,  132, 2179,    0,  528,  594,    0,  757,  479,  215,   50,  945,  272,    0,    0,  826,   58, },
                                           {  63,    0,    0,    0,    0,    0,  261,    0,    0,    0,    0,  135,   42,    0,    0,    0,  603,  243,  546,  123,    0,  252,    0,    0,    0,  252,  150,   27,   69,    0, },
                                           { 189,    0,    0,    0,    0,  144,  783,    0,  504,  552,   36,  405,  648,    0,  144,  528,  243, 1773,  648,  567,  288,  390,  192,    0,   48,    0,    0,   81,  417,  252, },
                                           {   0,    0,    0,    0,    0,    0,    0,    0,  567,  753,    0,    0,  619,    0,  162,  594,  546,  648, 2102,   52,  324,    0,    0,    0,   54,  818,   52,  356,  305,  576, },
                                           {  63,  551,  418,    0,   29,    0,  774,    0,    0,    0,    0,  344,  270,    0,    0,    0,  123,  567,   52, 1212,    0,  298,  592,    0,    0,   72,   31,   28,  787,  126, },
                                           {   0,    0,    0,    0,    0,    0,    0,  238,  252,  276,    0,    0,  252,    0,   72,  757,    0,  288,  324,    0,  577,  170,    0,    0,   24,    0,  204,    0,  357,   34, },
                                           {   0,  290,  220,    0,    0,  152,  966,  140,    0,    0,  705,  545,  292,    0,    0,  479,  252,  390,    0,  298,  170, 2395,  340,    0,  196,   42,  336,    0,  329,  286, },
                                           {   0,    0,    0,    0,  464,    0,    0,    0,    0,    0,  128,    0,    0,  242,    0,  215,    0,  192,    0,  592,    0,  340, 2318,  110,  319,  386,  882,   16,  968,  504, },
                                           {   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,  220,    0,   50,    0,    0,    0,    0,    0,    0,  110,  725,  290,   60,    0,  275,  425,    0, },
                                           {   0,    0,  532,    0,  784,    0,    0,    0,  322,  606,  616,  756,   42, 1142,  600,  945,    0,   48,   54,    0,   24,  196,  319,  290, 3583,  450,    0,  702,   28,    0, },
                                           {   0,    0,    0,    0,    0,    0,    0,    0,    0,  264,   64,    0,   24,  132,    0,  272,  252,    0,  818,   72,    0,   42,  386,   60,  450, 1109,   24,  802,  688,  432, },
                                           {   0,    0,    0,    0,  783,    0,    0,    0,    0,    0,    0,    0,    4,    0,    0,    0,  150,    0,   52,   31,  204,  336,  882,    0,    0,   24, 1987,   27,  640,    0, },
                                           {  21,    0,    0,    0,   29,    0,   87,    0,    0,  336,    0,   45,    0,    0,    0,    0,   27,   81,  356,   28,    0,    0,   16,  275,  702,  802,   27, 1629,  840,  108, },
                                           {  21,    0,    0,    0,    0,    0,   87,  294,    0,  276,  152,   45,    4,    0,    0,  826,   69,  417,  305,  787,  357,  329,  968,  425,   28,  688,  640,  840, 3763,  742, },
                                           {   0,    0,    0,    0,    0,  112,    0,   28,    0,    0,   28,  260,  632,    0,    0,   58,    0,  252,  576,  126,   34,  286,  504,    0,    0,  432,    0,  108,  742, 2970, }};
*/
/*
  3   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
  6   8   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
  0   7   9   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
 15   9  13   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
 13   2   0   5  14   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
 15  10   0   5   4   8   0   0   0   0   0   0   0   0   0   0   0   0   0   0
  6   2   4   4   0  19  17   0   0   0   0   0   0   0   0   0   0   0   0   0
 12   3  17   7  10  12  12  17   0   0   0   0   0   0   0   0   0   0   0   0
  9   7   0  16   7   6   0  11   3   0   0   0   0   0   0   0   0   0   0   0
  1  15   4   5   8   6  16   1  19  15   0   0   0   0   0   0   0   0   0   0
  2   0  15   6   0  14   1  15   4   9   8   0   0   0   0   0   0   0   0   0
  7   2  10   9  18  19   6   9   0  16   4   7   0   0   0   0   0   0   0   0
 10   2  13  13   8  15   1   7  11   3   8  16  11   0   0   0   0   0   0   0
 19  18   6  17   4  10   5   7  12  18   1   3  17  15   0   0   0   0   0   0
  3   9  11   4   3  14   9  16   9   5  10  11   8   9   3   0   0   0   0   0
  6   7  16  15  11  18   4  17   3   6  13   7  15  17  11   3   0   0   0   0
  0  13  13   2  14   7  19  13   0  11  14   5   7  15   8   3   4   0   0   0
 12  16   2   5   0   1   9   6  10   1   4  19  14  13   9  13  18   9   0   0
 16  11   0  14  12  17  11   5   8  15   4  16   1  18  16  18   8  17  16   0
 11   2  16   7  16   0  17   6   8  19   0  12   0   8   4   6   8  17   4  14
*/

double **mat_in;
int **mat_res_verifier;

struct node_info {
	int col_no;
	int tier_level;
	int dependency_count;
	int dependency_col[MATRIX_SIZE];
};

struct tier_map {
	int col_no;
	int tier_level_origin;
};

struct send_map {
	int col_no;
	int target_procs[MATRIX_SIZE];
	int target_count;
};

//int tiers[MATRIX_SIZE][MATRIX_SIZE];
int **tiers;
int current_tier;
int current_tier_size;
int zero_tier_size;

struct node_info *all_columns;
struct node_info *all_columns_orig;
struct tier_map *all_columns_sortmap;
struct send_map *self_send_map;
int **self_recv_map;

int **rank_col_map;
int *iteration_per_rank;
int rank_id, num_proc;

int has_node_left(struct node_info all_nodes[]);

int check_sat(struct node_info node, int tier);

void dependency_checker(struct tier_map *all_nodes_sorted, int num_proc);

void quick_sort(struct tier_map *all_nodes_sorted, int low, int high);

int partition(struct tier_map *all_nodes_sortmap, int low, int high);

void cdiv(double **matrix, int col_num_i);

void cmod(double **matrix, int col_num_j, int col_num_k);


int main(int argc, char *argv[]) {
	MPI_Init(&argc, &argv);

	int i, j, k;
	double start_timer,end_timer,timer_period;

	//rank_id=3;
	//num_proc=4;
	MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
	MPI_Request request;

	mat_in = malloc(sizeof(double*)*MATRIX_SIZE);
	mat_res_verifier = malloc(sizeof(int*)*MATRIX_SIZE);
	tiers = malloc(sizeof(int*)*MATRIX_SIZE);
	for(i=0;i<MATRIX_SIZE;i++)
	{
		mat_in[i]=malloc(sizeof(double)*MATRIX_SIZE);
		mat_res_verifier[i] = malloc(sizeof(int)*MATRIX_SIZE);
		tiers[i] = malloc(sizeof(int)*MATRIX_SIZE);
	}

	if(rank_id==0)
	{
		start_timer = clock();
	}

	FILE *fp1 = fopen("/home/andrew/CLionProjects/Belphegor-Generator/cmake-build-debug/res.txt","r");
	//FILE *fp1 = fopen(argv[1],"r");
	for(i=0;i<MATRIX_SIZE;i++)
	{
		for(j=0;j<MATRIX_SIZE;j++)
		{
			fscanf(fp1,"%lf",&mat_in[i][j]);
		}
	}
	fclose(fp1);

	fp1 = fopen("/home/andrew/CLionProjects/Belphegor-Generator/cmake-build-debug/lower.txt","r");
	//fp1 = fopen(argv[2],"r");
	for(i=0;i<MATRIX_SIZE;i++)
	{
		for(j=0;j<MATRIX_SIZE;j++)
		{
			fscanf(fp1,"%d",&mat_res_verifier[i][j]);
		}
	}
	fclose(fp1);

	zero_tier_size = 0;

	all_columns = malloc(sizeof(struct node_info) * MATRIX_SIZE);
	all_columns_orig = malloc(sizeof(struct node_info) * MATRIX_SIZE);
	struct node_info *current_node;

	all_columns_sortmap = malloc(sizeof(struct tier_map) * MATRIX_SIZE);

	//zero_tier_size is already set here.
	dependency_checker(all_columns_sortmap, num_proc);

	double **buffer_mat = (double **) malloc(sizeof(double *) * MATRIX_SIZE);
	for (i = 0; i < MATRIX_SIZE; i++) {
		buffer_mat[i] = (double *) malloc(sizeof(double *) * MATRIX_SIZE);
	}

	printf("Pre-processing complete\n");
	if(rank_id==0)
	{
		end_timer=clock();
		timer_period = (end_timer-start_timer)/CLOCKS_PER_SEC;
		printf("Pre-processing spent: %lf seconds\n", timer_period);
		start_timer = clock();
	}

	MPI_Barrier(MPI_COMM_WORLD);

/*    if(rank_id==0) {
        for (i = 0; i < MATRIX_SIZE; i++) {
            printf("i:%d col:%d\n", i,all_columns[i].col_no);
        }
    }*/

	float recv[MATRIX_SIZE][MATRIX_SIZE] = {0};
	int available[MATRIX_SIZE] = {0};

	//==================== construct iteration_per_rank ====================//
	int total_iteration = MATRIX_SIZE / num_proc;
	if (rank_id < MATRIX_SIZE % num_proc) {
		total_iteration++;
	}
	//printf("%d proc has %d iterations\n",rank_id, total_iteration);
	self_send_map=malloc(sizeof(struct send_map)*total_iteration);
	self_recv_map=malloc(sizeof(int*)*total_iteration);
	for(i=0;i<total_iteration;i++)
	{
		self_recv_map[i]=malloc(sizeof(int)*MATRIX_SIZE);
	}

	if (rank_id == 0) {
		iteration_per_rank = (int *) malloc(sizeof(int) * num_proc);

		iteration_per_rank[0] = total_iteration;
		for (i = 1; i < num_proc; i++) {
			MPI_Recv(&iteration_per_rank[i], 1, MPI_INT, i, i, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
		}

	} else {
		MPI_Send(&total_iteration, 1, MPI_INT, 0, rank_id, MPI_COMM_WORLD);
	}

	//==================== construct rank_col_map ====================//
	int temp;
	int *self_cols;
	if (rank_id == 0) {
		rank_col_map = (int **) malloc(sizeof(int *) * num_proc);
		for (temp = 0; temp < num_proc; temp++) {
			rank_col_map[temp] = (int *) malloc(sizeof(int) * iteration_per_rank[temp]);
		}
		self_cols = (int *) malloc(sizeof(int) * total_iteration);
		j = 0;
		for (i = 0; i < MATRIX_SIZE; i++) {
			if (i % num_proc == 0) {
				self_cols[j] = all_columns[i].col_no;
				j++;
			}
		}

		memcpy(rank_col_map[0], self_cols, sizeof(int) * total_iteration);
		for (temp = 1; temp < num_proc; temp++) {
			MPI_Recv(rank_col_map[temp], iteration_per_rank[temp], MPI_INT, temp, temp, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
		}
	} else {
		self_cols = (int *) malloc(sizeof(int) * total_iteration);
		j = 0;
		for (i = 0; i < MATRIX_SIZE; i++) {
			if (i % num_proc == rank_id) {
				self_cols[j] = all_columns[i].col_no;
				j++;
			}
		}
		MPI_Isend(self_cols, total_iteration, MPI_INT, 0, rank_id, MPI_COMM_WORLD, &request);
	}
	//MPI_Wait(&request, MPI_STATUSES_IGNORE);

	//test allocate cols to processors
/*    if(rank_id==0) {
        for(i=0; i<num_proc;i++) {
            printf("%d rank: ",i);
            for (j = 0; j < iteration_per_rank[i]; j++) {
                printf("%d col  ", rank_col_map[i][j]);
            }
            printf("\n");
        }
    }*/

	//==================== computation ====================//
	int iter;
	int recv_col;
	int send_count;
	int send_rank[MATRIX_SIZE];

	/*if(rank_id==3){
		for (iter = 0; iter < total_iteration; iter++) {
			current_node = &all_columns_orig[self_cols[iter]];
			printf("Col %d: ",current_node->col_no);
			for(i=0;i<current_node->dependency_count;i++)
				printf("%d ",current_node->dependency_col[i]);
			printf("\n");
		}
	}*/


	for (iter = 0; iter < total_iteration; iter++) {
		k = 0;
		current_node = &all_columns_orig[self_cols[iter]];
		for (i = current_node->col_no; i < MATRIX_SIZE; i++) {
			for (j = 0; j < all_columns[i].dependency_count; j++) {
				if (all_columns[i].dependency_col[j] == current_node->col_no) {
					if (rank_id == i % num_proc) {
						continue;
					}
					send_rank[k] = i % num_proc;
					self_send_map[iter].target_procs[k]=send_rank[k];
					//printf("rank %d computing %d send to rank %d to compute %d col\n",rank_id,current_node->col_no,send_rank[k],all_columns[i].col_no);
					k++;
					break;
				}
			}
		}
		send_count = k;
		self_send_map[iter].target_count=send_count;

		//Unlimited Debug Works
		for (i=0; i<self_send_map[iter].target_count; i++) {
			j=i+1;
			while(j<self_send_map[iter].target_count) {
				if (self_send_map[iter].target_procs[j] == self_send_map[iter].target_procs[i]) {
					for (k=j; k<self_send_map[iter].target_count; k++) {
						self_send_map[iter].target_procs[k] = self_send_map[iter].target_procs[k + 1];
					}
					self_send_map[iter].target_count--;
				} else
					j++;
			}
		}

	}

	/*for(iter=0;iter<total_iteration; iter++)
	{
		current_node = &all_columns_orig[self_cols[iter]];
		printf("Col %d going to: ",current_node->col_no);
		for(j=0;j<self_send_map[iter].target_count;j++)
		{
			printf("%d ",self_send_map[iter].target_procs[j]);
		}
		printf("\n");
	}*/


	for (iter = 0; iter < total_iteration; iter++) {
		k=0;
		current_node = &all_columns_orig[self_cols[iter]];
		for (i = 0; i < current_node->dependency_count; i++) {
			recv_col = current_node->dependency_col[i];
			for (j = 0; j < MATRIX_SIZE; j++) {
				if (all_columns[j].col_no == recv_col) {
					self_recv_map[iter][k] = j % num_proc;
					k++;
					break;
				}
			}
		}
	}

	for (iter = 0; iter < total_iteration; iter++) {
		current_node = &all_columns_orig[self_cols[iter]];
		//printf("rank %d compute %d col at total_iteration %d\n",rank_id,current_node->col_no,iter);
		if (current_node->tier_level == 0) {
			cdiv(mat_in, current_node->col_no);
			available[current_node->col_no] = 1;

		} else {
			//==========  receive and cmod  ==========//
			for (i = 0; i < current_node->dependency_count; i++) {
				recv_col = current_node->dependency_col[i];
				int recv_rank;
				recv_rank=self_recv_map[iter][i];
				/*for (j = 0; j < MATRIX_SIZE; j++) {
					if (all_columns[j].col_no == recv_col) {
						recv_rank = j % num_proc;
						break;
					}
				}*/

				if (rank_id == recv_rank && available[recv_col]==1) {
					//available[current_node->col_no] = 1;
					cmod(mat_in, current_node->col_no, current_node->dependency_col[i]);

				} else if (available[recv_col] == 0) {
					//printf("rank %d computing %d want to recv col %d from rank %d\n", rank_id, current_node->col_no, recv_col, recv_rank);
					MPI_Irecv(buffer_mat[recv_col],
					          MATRIX_SIZE,
					          MPI_DOUBLE,
					          recv_rank,
					          recv_col,
					          MPI_COMM_WORLD, &request);
					MPI_Wait(&request, MPI_STATUSES_IGNORE);
					available[recv_col] = 1;
					//printf("rank %d received %d\n",rank_id, recv_col);

					int ccc;
					for (ccc = 0; ccc < MATRIX_SIZE; ccc++) {
						mat_in[ccc][recv_col] = buffer_mat[recv_col][ccc];
					}

					cmod(mat_in, current_node->col_no, current_node->dependency_col[i]);

				} else {
					//printf("rank %d computing %d start without communication\n",rank_id,current_node->col_no);
					cmod(mat_in, current_node->col_no, current_node->dependency_col[i]);

				}
			}

			cdiv(mat_in, current_node->col_no);
			available[current_node->col_no] = 1;
		}

		//==========  send  ==========//
		//calculate which rank to send
		/*k = 0;
		//dependent col must appear after current col
		for (i = current_node->col_no; i < MATRIX_SIZE; i++) {
			for (j = 0; j < all_columns[i].dependency_count; j++) {
				if (all_columns[i].dependency_col[j] == current_node->col_no) {
					if (rank_id == i % num_proc) {
						continue;
					}
					send_rank[k] = i % num_proc;
					//printf("rank %d computing %d send to rank %d to compute %d col\n",rank_id,current_node->col_no,send_rank[k],all_columns[i].col_no);
					k++;
					break;
				}
			}
		}
		send_count = k;*/

		for (i = 0; i < MATRIX_SIZE; i++) {
			buffer_mat[current_node->col_no][i] = mat_in[i][current_node->col_no];
		}

		/*for (i = 0; i < send_count; i++) {
			MPI_Isend(buffer_mat[current_node->col_no],
			          MATRIX_SIZE,
			          MPI_DOUBLE,
			          send_rank[i],
			          current_node->col_no,
			          MPI_COMM_WORLD, &request);
		}*/
		//printf("self id:%d\n",rank_id);


		for (i = 0; i < self_send_map[iter].target_count; i++) {
			MPI_Isend(buffer_mat[current_node->col_no],
			          MATRIX_SIZE,
			          MPI_DOUBLE,
					//send_rank[i],
					  self_send_map[iter].target_procs[i],
					  current_node->col_no,
					  MPI_COMM_WORLD, &request);
		}


		MPI_Barrier(MPI_COMM_WORLD);
	}
	if (total_iteration * num_proc < MATRIX_SIZE) {
		MPI_Barrier(MPI_COMM_WORLD);
	}

	// send result to rank 0
	int ccc;
	if (rank_id == 0) {
		for (i = 1; i < num_proc; i++) {
			for (j = 0; j < iteration_per_rank[i]; j++) {
				recv_col = rank_col_map[i][j];
				MPI_Recv(buffer_mat[recv_col], MATRIX_SIZE, MPI_DOUBLE, i, i, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
				for (ccc = 0; ccc < MATRIX_SIZE; ccc++) {
					mat_in[ccc][recv_col] = buffer_mat[recv_col][ccc];
				}
			}
		}
	} else {
		for (i = 0; i < total_iteration; i++) {
			for (j = 0; j < MATRIX_SIZE; j++) {
				buffer_mat[self_cols[i]][j] = mat_in[j][self_cols[i]];
			}
			MPI_Send(buffer_mat[self_cols[i]], MATRIX_SIZE, MPI_DOUBLE, 0, rank_id, MPI_COMM_WORLD);
		}
	}
	//MPI_Wait(&request, MPI_STATUSES_IGNORE);

	// test final result
	/*if(rank_id==0)
	{
		for(i=0;i<MATRIX_SIZE;i++)
		{
			for(j=0;j<MATRIX_SIZE;j++)
			{
				if(i>=j)
					printf("%4.0lf",mat_in[i][j]);
				else
					printf("%4.0lf",0.00);
			}
			printf("\n");
		}
	}*/

	if(rank_id==0)
	{
		end_timer=clock();
		timer_period = (end_timer-start_timer)/CLOCKS_PER_SEC;
		printf("Computation spent: %lf seconds\n", timer_period);
	}

	if(rank_id==0){
		int error_flag=0;
		for(i=0;i<MATRIX_SIZE;i++)
		{
			for(j=0;j<MATRIX_SIZE;j++)
			{
				if(i>=j)
				{
					if(mat_in[i][j]!=mat_res_verifier[i][j])
					{
						printf("Well, fucked up.\n");
						error_flag=1;
						break;
					}
				}
			}
			if(error_flag==1)
				break;
		}
		if(error_flag==0)
		{
			printf("Test passed.\n");
		}
	}



	MPI_Finalize();
}

void cmod(double **matrix, int col_num_j, int col_num_k) {
	int i, j, k;
	j = col_num_j;
	k = col_num_k;

	for (i = j; i < MATRIX_SIZE; i++) {
		matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[j][k];
	}
}

void cdiv(double **matrix, int col_num_j) {
	int i, j;
	j = col_num_j;
	matrix[col_num_j][col_num_j] = sqrt(matrix[col_num_j][col_num_j]);

	for (i = j + 1; i < MATRIX_SIZE; i++) {
		matrix[i][j] = matrix[i][j] / matrix[j][j];
	}
}

int check_sat(struct node_info node, int tier) {
	int i, j, k;
	int dep_count;
	int found_flag = 0;

	for (dep_count = 0; dep_count < node.dependency_count; dep_count++) {
		found_flag = 0;
		for (i = tier-1; i >=0; i--) {
			for (j = 0; j < MATRIX_SIZE; j++) {
				if (tiers[i][j] == node.dependency_col[dep_count]) {
					found_flag = 1;
					break;
				}
			}
			if (found_flag == 1)
				break;
		}
		if (found_flag == 0)
			return 0;
	}

	return 1;
}

void dependency_checker(struct tier_map *all_nodes_sortmap, int num_proc) {
	int row_counter1;
	int col_counter1, col_counter2;

	int i, j, k;
	//int fill_in_temp[MATRIX_SIZE][MATRIX_SIZE];
	int cols_left=MATRIX_SIZE;

	int **fill_in_temp = malloc(sizeof(int*)*MATRIX_SIZE);
	for(i=0;i<MATRIX_SIZE;i++)
		fill_in_temp[i] = malloc(sizeof(int)*MATRIX_SIZE);

	/*mat_in=(double**) malloc(sizeof(double)*MATRIX_SIZE);
	for(i=0;i<MATRIX_SIZE;i++)
	{
		mat_in[i]=sparse_matrix[i];
	}*/

	for (i = 0; i < MATRIX_SIZE; i++) {
		for (j = 0; j < MATRIX_SIZE; j++) {
			if (mat_in[i][j] == 0)
				fill_in_temp[i][j] = 0;
			else
				fill_in_temp[i][j] = 1;
		}
	}


	for (col_counter1 = 1; col_counter1 < MATRIX_SIZE; col_counter1++) {
		for (col_counter2 = 0; col_counter2 < col_counter1; col_counter2++) {
			if (mat_in[col_counter1][col_counter2] != 0) {
				for (row_counter1 = col_counter1; row_counter1 < MATRIX_SIZE; row_counter1++) {
					if (mat_in[row_counter1][col_counter2] != 0 && fill_in_temp[row_counter1][col_counter1] == 0)
						fill_in_temp[row_counter1][col_counter1] = 2;
				}
			}
		}
	}

	/*printf("Fill-in Matrix:\n");
	for(i=0;i<MATRIX_SIZE;i++)
	{
		for(j=0;j<MATRIX_SIZE;j++)
		{
			printf("%d ",fill_in_temp[i][j]);
		}
		printf("\n");
	}
	printf("\n\n");*/

	//printf("col 0: \n");
	all_columns[0].col_no = 0;
	all_columns[0].dependency_count = 0;
	all_columns[0].tier_level = -1;

	for (col_counter1 = 1; col_counter1 < MATRIX_SIZE; col_counter1++) {
		//printf("col %d: ",col_counter1);
		all_columns[col_counter1].col_no = col_counter1;
		all_columns[col_counter1].dependency_count = 0;
		all_columns[col_counter1].tier_level = -1;
		for (col_counter2 = 0; col_counter2 < col_counter1; col_counter2++) {
			if (fill_in_temp[col_counter1][col_counter2] != 0) {
				//printf("%d ",col_counter2);
				all_columns[col_counter1].dependency_col[all_columns[col_counter1].dependency_count] = col_counter2;
				all_columns[col_counter1].dependency_count++;
			}
		}
		//printf("\n");
	}

/*
	printf("Dependency Info:\n");
	for(i=0;i<MATRIX_SIZE;i++)
	{
		printf("col %d: ",all_nodes[i].col_no);
		for(j=0;j<all_nodes[i].dependency_count;j++)
		{
			printf("%d ",all_nodes[i].dependency_col[j]);
		}
		printf("\n");
	}
	printf("\n\n");*/

	/*for (i = 0; i < MATRIX_SIZE; i++) {
		for (j = 0; j < MATRIX_SIZE; j++) {
			tiers[i][j] == -2;
		}
	}*/

	j = 0;
	for (i = 0; i < MATRIX_SIZE; i++) {
		if (all_columns[i].dependency_count == 0) {
			all_columns[i].tier_level = 0;
			cols_left--;
			tiers[0][j] = i;
			j++;
			zero_tier_size++;
		}
	}

	current_tier = 1;
	/*while(cols_left>0){
		current_tier_size = 0;
		for (i = 0; i < MATRIX_SIZE; i++) {
			if (all_columns[i].tier_level == -1 && check_sat(all_columns[i], current_tier)) {
				all_columns[i].tier_level = current_tier;
				cols_left--;
				tiers[current_tier][current_tier_size] = i;
				current_tier_size++;
			}
		}

		current_tier++;
	}*/
	int threshold = MATRIX_SIZE*0.002;
	while (current_tier<threshold){
		current_tier_size = 0;
		for (i = 0; i < MATRIX_SIZE; i++) {
			if (all_columns[i].tier_level == -1 && check_sat(all_columns[i], current_tier)) {
				all_columns[i].tier_level = current_tier;
				tiers[current_tier][current_tier_size] = i;
				current_tier_size++;
			}
		}
		current_tier++;
	}

	for (i = 0; i < MATRIX_SIZE; i++) {
		if (all_columns[i].tier_level == -1) {
			all_columns[i].tier_level = current_tier;
			tiers[current_tier][current_tier_size]=1;
			current_tier++;
		}
	}


/*	printf("Tier Info:\n");
	for(i=0;i<MATRIX_SIZE;i++)
	{
		printf("col %d: %d\n",all_nodes[i].col_no,all_nodes[i].tier_level);
	}
	printf("\n\n");*/

	memcpy(all_columns_orig, all_columns, sizeof(struct node_info) * MATRIX_SIZE);
	//sort by tier info
	int low = 0, high = MATRIX_SIZE - 1;
	for (i = 0; i < MATRIX_SIZE; i++) {
		all_nodes_sortmap[i].tier_level_origin = all_columns[i].tier_level;
		all_nodes_sortmap[i].col_no = all_columns[i].col_no;
	}
	quick_sort(all_nodes_sortmap, low, high);

/*    printf("sorted map:\n");
    for(i=0;i<MATRIX_SIZE;i++){
        printf("col %d tier: %d\n",all_nodes_sortmap[i].col_no,all_nodes_sortmap[i].tier_level_origin);
    }
    printf("\n\n");*/

	struct node_info *temp;
	temp = (struct node_info *) malloc(sizeof(struct node_info) * MATRIX_SIZE);
	int id;
	for (i = 0; i < MATRIX_SIZE; i++) {
		id = all_nodes_sortmap[i].col_no;
		memcpy(&temp[i], &all_columns[id], sizeof(struct node_info));
	}
	all_columns = temp;

/*    printf("Tier Info After Sort:\n");
    for(i=0;i<MATRIX_SIZE;i++)
    {
        printf("col %d: %d\n",all_nodes[i].col_no,all_nodes[i].tier_level);
    }
    printf("\n\n");*/



	/*int out_node_count=0;
	int current_tier_count=0;
	current_tier=0;
	while(out_node_count<MATRIX_SIZE)
	{
		printf("Tier %d: ",current_tier);
		current_tier_count=0;
		for(i=0;i<MATRIX_SIZE;i++)
		{
			if(all_nodes[i].tier_level==current_tier) {
				current_tier_count++;
			}
		}
		printf("%d: ",current_tier_count);
		for(i=0;i<MATRIX_SIZE;i++)
		{
			if(all_nodes[i].tier_level==current_tier)
			{
				printf("%d ",i);
				out_node_count++;
			}
		}
		printf("\n");
		current_tier++;
		current_tier_count=0;
	}*/

}

void quick_sort(struct tier_map *all_nodes_sortmap, int low, int high) {
	int pi;
	if (low < high) {
		pi = partition(all_nodes_sortmap, low, high);

		quick_sort(all_nodes_sortmap, low, pi - 1);
		quick_sort(all_nodes_sortmap, pi + 1, high);
	}
}

int partition(struct tier_map *all_nodes_sortmap, int low, int high) {
	int i, j;
	struct tier_map temp;
	int target;

	target = all_nodes_sortmap[high].tier_level_origin;
	i = low;

	for (j = low; j < high; j++) {
		if (all_nodes_sortmap[j].tier_level_origin <= target) {
			//swap arr[i] and arr[j]
			temp.tier_level_origin = all_nodes_sortmap[i].tier_level_origin;
			temp.col_no = all_nodes_sortmap[i].col_no;

			all_nodes_sortmap[i].tier_level_origin = all_nodes_sortmap[j].tier_level_origin;
			all_nodes_sortmap[i].col_no = all_nodes_sortmap[j].col_no;

			all_nodes_sortmap[j].tier_level_origin = temp.tier_level_origin;
			all_nodes_sortmap[j].col_no = temp.col_no;

			i++; //increase the smaller number
		}
	}

	//swap arr[i] and arr[high]
	temp.tier_level_origin = all_nodes_sortmap[i].tier_level_origin;
	temp.col_no = all_nodes_sortmap[i].col_no;

	all_nodes_sortmap[i].tier_level_origin = all_nodes_sortmap[j].tier_level_origin;
	all_nodes_sortmap[i].col_no = all_nodes_sortmap[j].col_no;

	all_nodes_sortmap[j].tier_level_origin = temp.tier_level_origin;
	all_nodes_sortmap[j].col_no = temp.col_no;

	return i;

}
