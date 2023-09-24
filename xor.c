#include <xor.h>

//R = set of all real numbers, R^n = Ordered n-tuple of real numbers
//x = [x1, x2, ... , xn] (x in R^n)
//w = [w1, w2, ... , wn] (w in R^n)
//b = C (some constant C in R)
//y = x * w (y in R)


Xor rand_model() {
    Xor m;
    m.W = malloc(sizeof(float*)*ROWCOUNT);
    m.b = malloc(sizeof(float)*ROWCOUNT);
    for (int i = 0; i < ROWCOUNT; i++) {
        m.W[i] = malloc(sizeof(float)*COLCOUNT);
    }
    for (int i = 0; i < ROWCOUNT; i++) {
        for (int j = 0; j < COLCOUNT; j++) {
            m.W[i][j] = rand_f();
        }
        m.b[i] = rand_f();
    }
    return m;
}


float forwardprop(Xor m, float * x) {
    float a[ROWCOUNT-1];
    for (int i = 0; i < ROWCOUNT-1; i++) {
        a[i] = sigmoid_f(m.W[i][0]*x[0] + m.W[i][1]*x[1] + m.b[i]);
    }

    return sigmoid_f(m.W[ROWCOUNT-1][0]*a[0] + m.W[ROWCOUNT-1][1]*a[1] + m.b[ROWCOUNT-1]);
}


Xor compute_gradient(Xor m) {
    Xor g;

    g.W = malloc(sizeof(float*)*ROWCOUNT);
    g.b = malloc(sizeof(float)*ROWCOUNT);
    for (int i = 0; i < ROWCOUNT; i++) {
        g.W[i] = malloc(sizeof(float)*COLCOUNT);
    }
    
    float c = cost(m);
    float delta = 1.0e-3;
    float temp;


    for (int i = 0; i < ROWCOUNT; i++) {
        for (int j = 0; j < COLCOUNT; j++) {
            temp = m.W[i][j];
            m.W[i][j] += delta;
            g.W[i][j] = (cost(m) - c)/delta;
            m.W[i][j] = temp;
        }
        temp = m.b[i];
        m.b[i] += delta;
        g.b[i] = (cost(m) - c)/delta;
        m.b[i] = temp;
    }
    
    return g;
}


Xor backprop(Xor m) {
    float learningrate = 1.0e-1;

    Xor g = compute_gradient(m);

    for (int i = 0; i < ROWCOUNT; i++) {
        for (int j = 0; j < COLCOUNT; j++) {
            m.W[i][j] -= learningrate*g.W[i][j];
        }
        m.b[i] -= learningrate*g.b[i];
    }
    free_model(g);
    return m;
}



float cost(Xor model) {
    float y, diff, result = 0.0;
    float * x = malloc(sizeof(float)*COLCOUNT);

    for (int i = 0; i < TRAININGCOUNT; i++) {
        for (int j = 0; j < COLCOUNT; j++) {
            x[j] = training_example[i][j];
        }
        y = forwardprop(model, x);
        diff = y - training_example[i][2];
        result += diff*diff;
    }
    result /= TRAININGCOUNT;
    return result;
}


int main(void) {
    Xor m = rand_model();
    print_model(m);
    print_cost(m);


    for (int i = 0; i < 1000000; i++) {
        if (i % 100000 == 0)
            print_cost(m);
        m = backprop(m);
    }
    print_model(m);
    float xor_def[4][2] = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
    for (int i = 0; i < 4; i++) {
        printf("%f ^ %f = %f\n", xor_def[i][0], xor_def[i][1], forwardprop(m, xor_def[i]));

    }

}