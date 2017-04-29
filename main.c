#include <stdio.h>
#include <stdlib.h>
#include <mpi/mpi.h>
#include <math.h>
#include <string.h>

#define MATRIX_SIZE 20

//TODO: Refactor code so matrix is read from a file.
double mat_in[MATRIX_SIZE][MATRIX_SIZE] = {{   9,   18,    0,   45,   39,   45,   18,   36,   27,    3,    6,   21,   30,   57,    9,   18,    0,   36,   48,   33 },
                                           {  18,  100,   56,  162,   94,  170,   52,   96,  110,  126,   12,   58,   76,  258,   90,   92,  104,  200,  184,   82 },
                                           {   0,   56,  130,  180,   14,   70,   50,  174,   49,  141,  135,  104,  131,  180,  162,  193,  208,  130,   77,  158 },
                                           {  45,  162,  180,  476,  218,  320,  164,  435,  214,  207,  231,  262,  350,  542,  273,  376,  288,  355,  353,  398 },
                                           {  39,   94,   14,  218,  394,  296,  102,  337,  309,  180,   56,  392,  311,  424,  119,  321,  232,  213,  468,  406 },
                                           {  45,  170,   70,  320,  296,  430,  282,  381,  361,  270,  172,  394,  387,  646,  279,  423,  252,  373,  604,  284 },
                                           {  18,   52,   50,  164,  102,  282,  722,  606,  246,  458,  379,  585,  470,  517,  515,  584,  542,  304,  684,  451 },
                                           {  36,   96,  174,  435,  337,  381,  606, 1168,  570,  521,  756,  956,  829,  842,  856, 1133,  947,  483,  864,  925 },
                                           {  27,  110,   49,  214,  309,  361,  246,  570,  601,  354,  375,  560,  568,  770,  462,  724,  406,  402,  710,  427 },
                                           {   3,  126,  141,  207,  180,  270,  458,  521,  354, 1210,  418,  725,  588, 1075,  716,  674,  893,  646, 1007,  983 },
                                           {   6,   12,  135,  231,   56,  172,  379,  756,  375,  418,  848,  801,  744,  698,  801, 1023,  730,  278,  639,  614 },
                                           {  21,   58,  104,  262,  392,  394,  585,  956,  560,  725,  801, 1357, 1011, 1050,  900, 1265, 1057,  473, 1278, 1136 },
                                           {  30,   76,  131,  350,  311,  387,  470,  829,  568,  588,  744, 1011, 1352, 1190, 1056, 1390,  850,  912, 1193,  937 },
                                           {  57,  258,  180,  542,  424,  646,  517,  842,  770, 1075,  698, 1050, 1190, 2192, 1174, 1642, 1229, 1342, 1753, 1245 },
                                           {   9,   90,  162,  273,  119,  279,  515,  856,  462,  716,  801,  900, 1056, 1174, 1250, 1480, 1252, 1013, 1237,  935 },
                                           {  18,   92,  193,  376,  321,  423,  584, 1133,  724,  674, 1023, 1265, 1390, 1642, 1480, 2223, 1646, 1237, 1779, 1207 },
                                           {   0,  104,  208,  288,  232,  252,  542,  947,  406,  893,  730, 1057,  850, 1229, 1252, 1646, 1822, 1138, 1524, 1344 },
                                           {  36,  200,  130,  355,  213,  373,  304,  483,  402,  646,  278,  473,  912, 1342, 1013, 1237, 1138, 2045, 1922, 1262 },
                                           {  48,  184,   77,  353,  468,  604,  684,  864,  710, 1007,  639, 1278, 1193, 1753, 1237, 1779, 1524, 1922, 3227, 1979 },
                                           {  33,   82,  158,  398,  406,  284,  451,  925,  427,  983,  614, 1136,  937, 1245,  935, 1207, 1344, 1262, 1979, 2261 }};


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

int tiers[MATRIX_SIZE][MATRIX_SIZE];
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

int has_node_left(struct node_info all_nodes[]);

int check_sat(struct node_info node, int tier);

void dependency_checker(struct tier_map *all_nodes_sorted, int num_proc);

void quick_sort(struct tier_map *all_nodes_sorted, int low, int high);

int partition(struct tier_map *all_nodes_sortmap, int low, int high);

void cdiv(double (*matrix)[MATRIX_SIZE], int col_num_i);

void cmod(double (*matrix)[MATRIX_SIZE], int col_num_j, int col_num_k);

int main(int argc, char *argv[]) {
	MPI_Init(&argc, &argv);

	int rank_id, num_proc;
	int i, j, k;

	//rank_id=3;
	//num_proc=4;
	MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
	MPI_Request request;

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
			MPI_Irecv(rank_col_map[temp], iteration_per_rank[temp], MPI_INT, temp, temp, MPI_COMM_WORLD, &request);
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
	MPI_Wait(&request, MPI_STATUSES_IGNORE);

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

	for(iter=0;iter<total_iteration; iter++)
	{
		current_node = &all_columns_orig[self_cols[iter]];
		printf("Col %d going to: ",current_node->col_no);
		for(j=0;j<self_send_map[iter].target_count;j++)
		{
			printf("%d ",self_send_map[iter].target_procs[j]);
		}
		printf("\n");
	}


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

	int sssss=0;

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
					if(current_node->col_no==5){
						printf("Cmoding %d with %d\n",current_node->col_no,current_node->dependency_col[i]);
					}
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
					if(current_node->col_no==5){
						printf("Cmoding %d with %d\n",current_node->col_no,current_node->dependency_col[i]);
					}
				} else {
					//printf("rank %d computing %d start without communication\n",rank_id,current_node->col_no);
					cmod(mat_in, current_node->col_no, current_node->dependency_col[i]);
					if(current_node->col_no==5){
						printf("Cmoding %d with %d\n",current_node->col_no,current_node->dependency_col[i]);
					}
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
	MPI_Wait(&request, MPI_STATUSES_IGNORE);

	// test final result
	if(rank_id==0)
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
	}

	MPI_Finalize();
}

void cmod(double (*matrix)[MATRIX_SIZE], int col_num_j, int col_num_k) {
	int i, j, k;
	j = col_num_j;
	k = col_num_k;

	for (i = j; i < MATRIX_SIZE; i++) {
		matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[j][k];
	}
}

void cdiv(double (*matrix)[MATRIX_SIZE], int col_num_j) {
	int i, j;
	j = col_num_j;
	matrix[col_num_j][col_num_j] = sqrt(matrix[col_num_j][col_num_j]);

	for (i = j + 1; i < MATRIX_SIZE; i++) {
		matrix[i][j] = matrix[i][j] / matrix[j][j];
	}
}

int has_node_left(struct node_info all_nodes[]) {
	int i;
	int res = 0;
	for (i = 0; i < MATRIX_SIZE; i++) {
		if (all_nodes[i].tier_level == -1)
			return 1;
	}
	return res;
}

int check_sat(struct node_info node, int tier) {
	int i, j, k;
	int dep_count;
	int found_flag = 0;

	for (dep_count = 0; dep_count < node.dependency_count; dep_count++) {
		found_flag = 0;
		for (i = 0; i < tier; i++) {
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
	int fill_in_temp[MATRIX_SIZE][MATRIX_SIZE];

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

	for (i = 0; i < MATRIX_SIZE; i++) {
		for (j = 0; j < MATRIX_SIZE; j++) {
			tiers[i][j] == -2;
		}
	}

	j = 0;
	for (i = 0; i < MATRIX_SIZE; i++) {
		if (all_columns[i].dependency_count == 0) {
			all_columns[i].tier_level = 0;
			tiers[0][j] = i;
			j++;
			zero_tier_size++;
		}
	}

	current_tier = 1;
	while (has_node_left(all_columns) == 1) {
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