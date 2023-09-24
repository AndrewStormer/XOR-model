#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


typedef float trainingmodel[3];
trainingmodel xor_model[] = {
   {0, 0, 0},
   {0, 1, 1},
   {1, 0, 1},
   {1, 1, 0}
};

trainingmodel * training_example = xor_model;
int TRAININGCOUNT = 4;

#define COLCOUNT 2
#define ROWCOUNT 3
typedef struct {
    float ** W;
    float * b;
} Xor;


Xor backprop(Xor m);
float forwardprop(Xor m, float * x);
float cost(Xor m);
Xor rand_model();


void free_model(Xor m) {

    for (int i = 0; i < ROWCOUNT; i++) {
        free(m.W[i]);
    }
    free(m.W);
    free(m.b);
}


float sigmoid_f(float x) {
    return (1.0f / (1.0f + expf(-x)));
}


float rand_f(void) {
    return (float)rand() / (float)RAND_MAX;
}


void print_cost(Xor m) {
    printf("%f\n", cost(m));
}


void print_model(Xor m) {
    for (int i = 0; i < ROWCOUNT; i++) {
        for (int j = 0; j < COLCOUNT; j++) {
            printf("W[%d][%d] = %f ", i, j, m.W[i][j]);
        }
        printf("b[%d] = %f\n", i, m.b[i]);
    }
}