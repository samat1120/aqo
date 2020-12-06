/*
 *******************************************************************************
 *
 *	MACHINE LEARNING TECHNIQUES
 *
 * This module does not know anything about DBMS, cardinalities and all other
 * stuff. It learns matrices, predicts values and is quite happy.
 * The proposed method is designed for working with limited number of objects.
 * It is guaranteed that number of rows in the matrix will not exceed aqo_K
 * setting after learning procedure. This property also allows to adapt to
 * workloads which properties are slowly changed.
 *
 *******************************************************************************
 *
 * Copyright (c) 2016-2020, Postgres Professional
 *
 * IDENTIFICATION
 *	  aqo/machine_learning.c
 *
 */

#include "aqo.h"

double
neural_predict (int nfeatures, double **W1, double *b1, double **W2, double *b2, double *W3, double b3, double *feature) //prediction
{
    double *out1;
    double *out2;
    double *out3;
    double *out4;
    double out5;
    out1 = palloc0(WIDTH_1 * sizeof(*out1));
    for (int i = 0; i < WIDTH_1; ++i){
        for (int j = 0; j < nfeatures; ++j)
            out1[i] = out1[i]+feature[j]*W1[i][j];
        out1[i]=out1[i]+b1[i];
    }
    out2 = palloc0(WIDTH_1 * sizeof(*out2)); // vector for the output of Leaky ReLU activation function in the first layer
    for (int i = 0; i < WIDTH_1; ++i){
        if (out1[i]<out1[i]*slope)
            out2[i]=out1[i]*slope;
        else
            out2[i]=out1[i];
    }
    out3 = palloc0(WIDTH_2 * sizeof(*out3));
    for (int i = 0; i < WIDTH_2; ++i){
        for (int j = 0; j < WIDTH_1; ++j)
            out3[i]=out3[i]+out2[j]*W2[i][j];
        out3[i]=out3[i]+b2[i];
    }
    out4 = palloc0(WIDTH_2 * sizeof(*out4)); // vector for the output of Leaky ReLU activation function in the second layer
    for (int i = 0; i < WIDTH_2; ++i){
        if (out3[i]<out3[i]*slope)
            out4[i]=out3[i]*slope;
        else
            out4[i]=out3[i];
    }
    out5=0; //final result (one number)
    for (int j = 0; j < WIDTH_2; ++j)
        out5=out5+out4[j]*W3[j];
    out5=out5+b3;
    pfree(out1);
    pfree(out2);
    pfree(out3);
    pfree(out4);
    return out5;
}

void
neural_learn (int nfeatures, double **W1, double *b1, double **W2, double *b2, double *W3, double b3, double *feature, double target)
{
    double *out1;
    double *out2;
    double *out3;
    double *out4;
    double out5, loss, dp1;
    double *gradW3;
    double *dp2;
    double	*gradW2[WIDTH_2];
    double *dp3;
    double	*gradW1[WIDTH_1];
    double lrr;
    lrr = lr;
    for (int k = 0; k < N_ITERS; ++k){
        if (k==25 || k==50 || k==75 || k==100)
            lrr = lrr/10;
        out1 = palloc0(WIDTH_1 * sizeof(*out1));
        for (int i = 0; i < WIDTH_1; ++i){
            for (int j = 0; j < nfeatures; ++j)
                out1[i] = out1[i]+feature[j]*W1[i][j];
            out1[i]=out1[i]+b1[i];
        }
        out2 = palloc0(WIDTH_1 * sizeof(*out2));
        for (int i = 0; i < WIDTH_1; ++i){
            if (out1[i]<out1[i]*slope)
                out2[i]=out1[i]*slope;
            else
                out2[i]=out1[i];
        }
        out3 = palloc0(WIDTH_2 * sizeof(*out3));
        for (int i = 0; i < WIDTH_2; ++i){
            for (int j = 0; j < WIDTH_1; ++j)
                out3[i]=out3[i]+out2[j]*W2[i][j];
            out3[i]=out3[i]+b2[i];
        }
        out4 = palloc0(WIDTH_2 * sizeof(*out4));
        for (int i = 0; i < WIDTH_2; ++i){
            if (out3[i]<out3[i]*slope)
                out4[i]=out3[i]*slope;
            else
                out4[i]=out3[i];
        }
        out5=0;
        for (int j = 0; j < WIDTH_2; ++j)
            out5=out5+out4[j]*W3[j];
        out5=out5+b3; // output of forward pass
        loss = pow(out5 - target,2); 
        dp1 = (out5 - target) * 2; // derivative of the loss w.r.t. the output of forward pass
        gradW3=palloc(WIDTH_2 * sizeof(*gradW3)); // vector for the gradient of the loss w.r.t. weight vector in the third layer
        for (int i = 0; i < WIDTH_2; ++i)
            gradW3[i] = dp1 * out4[i];
        dp2=palloc(WIDTH_2 * sizeof(*dp2));
        for (int i = 0; i < WIDTH_2; ++i)
            dp2[i] = dp1 * W3[i];
        for (int i = 0; i < WIDTH_2; ++i)
            if (out3[i]<slope*out3[i])
                dp2[i] = dp2[i]*slope;
        gradW2[WIDTH_2]; // matrix for the gradient of the loss w.r.t. weight matrix in the second layer
        for (int i = 0; i < WIDTH_2; ++i)
            gradW2[i] = palloc(sizeof(**gradW2) * WIDTH_1);
        for (int i = 0; i < WIDTH_2; ++i)
            for (int j = 0; j < WIDTH_1; ++j)
                gradW2[i][j]=dp2[i]*out2[j];
        dp3=palloc0(sizeof(*dp3) * WIDTH_1);
        for (int i = 0; i < WIDTH_1; ++i)
            for (int j = 0; j < WIDTH_2; ++j)
                dp3[j]+= dp2[j]*W2[j][i];
        for (int i = 0; i < WIDTH_1; ++i)
            if (out1[i]<slope*out1[i])
                dp3[i] = dp3[i]*slope;
        for (int i = 0; i < WIDTH_1; ++i)
            gradW1[i] = palloc(sizeof(**gradW1) * nfeatures);
        for (int i = 0; i < WIDTH_1; ++i)
            for (int j = 0; j < nfeatures; ++j)
                gradW1[i][j] = dp3[i] * feature[j];
        for (int i = 0; i < WIDTH_1; ++i){
            for (int j = 0; j < nfeatures; ++j)
                W1[i][j] = W1[i][j] - lrr*gradW1[i][j]; // updating the weights in the first layer
            b1[i] = b1[i] - lrr*dp3[i];
        }
        for (int i = 0; i < WIDTH_2; ++i){
            for (int j = 0; j < WIDTH_1; ++j)
                W2[i][j] = W2[i][j] - lrr*gradW2[i][j]; // updating the weights in the second layer
            b2[i] = b2[i] - lrr*dp2[i];
        }
        for (int i = 0; i < WIDTH_2; ++i)
            W3[i] = W3[i] - lrr*gradW3[i]; // updating the weights in the third layer
        b3 = b3 - lrr*dp1;
        if (nfeatures>0)
        	for (int i = 0; i < WIDTH_1; ++i)
        		pfree(gradW1[i]);
        for (int i = 0; i < WIDTH_2; ++i)
        	pfree(gradW2[i]);
        pfree(gradW3);
        pfree(out1);
        pfree(out2);
        pfree(out3);
        pfree(out4);
        pfree(dp2);
        pfree(dp3);
    }
}
