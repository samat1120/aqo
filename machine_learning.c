/*
 *******************************************************************************
 *
 *	MACHINE LEARNING TECHNIQUES
 *
 * This module does not know anything about DBMS, cardinalities and all other
 * stuff. It learns matrices, predicts values and is quite happy.
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

static void zero_grad(int n_batch, int n_cols,  double **gradInput2,  double **gradInput3,  double **gradInput4,  double **gradInput5,  double *gradInput,
                        double **gradW1,  double *gradb1,  double **gradW2,  double *gradb2,  double *gradW3,  double *gradb3);
static void optim_step(int n_batch, int n_cols,  double **W1,  double *b1,  double **W2,  double *b2,  double *W3,  double *b3,
                       double **gradW1,  double *gradb1,  double **gradW2,  double *gradb2,  double *gradW3,  double *gradb3,
                       double **W1_m,  double **W1_v,  double *b1_m,  double *b1_v,  double **W2_m,  double **W2_v,  double *b2_m,  double *b2_v,  double *W3_m,  double *W3_v,  double *b3_m, double *b3_v,
                      int *step_layer1, int *steps
                      );

static void zero_grad(int n_batch, int n_cols,  double **gradInput2,  double **gradInput3,  double **gradInput4,  double **gradInput5,  double *gradInput,
                        double **gradW1,  double *gradb1,  double **gradW2,  double *gradb2,  double *gradW3,  double *gradb3){
    int i,j;
    for (i=0;i<n_batch;i++){
         for (j=0;j<WIDTH_1;++j){
             gradInput2[i][j] = 0;
             gradInput3[i][j] = 0;}
             for (j=0;j<WIDTH_2;++j){
             gradInput4[i][j] = 0;
             gradInput5[i][j] = 0;}
         gradInput[i]=0;
    }

    for (i=0;i<WIDTH_1;i++){
        for (j=0;j<n_cols;++j)
            gradW1[i][j] = 0;
        gradb1[i] = 0;}
    for (i=0;i<WIDTH_2;i++){
        for (j=0;j<WIDTH_1;++j)
            gradW2[i][j] = 0;
        gradb2[i] = 0;
        gradW3[i] = 0;}
    (*gradb3) = 0;
}

static void optim_step(int n_batch, int n_cols,  double **W1,  double *b1,  double **W2,  double *b2,  double *W3,  double *b3,
                       double **gradW1,  double *gradb1,  double **gradW2,  double *gradb2,  double *gradW3,  double *gradb3,
                       double **W1_m,  double **W1_v,  double *b1_m,  double *b1_v,  double **W2_m,  double **W2_v,  double *b2_m,  double *b2_v,  double *W3_m,  double *W3_v,  double *b3_m, double *b3_v,
                      int *step_layer1, int *steps
                      )
{
    int i,j;
    double bias_correction1, bias_correction2;
    for (i=0;i<n_cols;i++)
        step_layer1[i] = step_layer1[i] + 1;
    (*steps) = (*steps) + 1;
    for (j=0;j<n_cols;j++){
        bias_correction1 = 1 / (1 - pow(beta_1, step_layer1[j]));
        bias_correction2 = 1 / (1 - pow(beta_2, step_layer1[j]));
        for (i=0;i<WIDTH_1;i++){
            W1_m[i][j] = beta_1*W1_m[i][j]+(1-beta_1)*gradW1[i][j];
            W1_v[i][j] = beta_2*W1_v[i][j]+(1-beta_2)*pow(gradW1[i][j],2);
            W1[i][j] = W1[i][j] - (lr * bias_correction1 / sqrt(bias_correction2)) * W1_m[i][j] / (sqrt(W1_v[i][j]) + eps);
        }
    }
    bias_correction1 = 1 / (1 - pow(beta_1, (*steps)));
    bias_correction2 = 1 / (1 - pow(beta_2, (*steps)));
    for (i=0;i<WIDTH_1;i++){
        b1_m[i] = beta_1*b1_m[i]+(1-beta_1)*gradb1[i];
        b1_v[i] = beta_2*b1_v[i]+(1-beta_2)*pow(gradb1[i],2);
        b1[i] = b1[i] - (lr * bias_correction1 / sqrt(bias_correction2)) * b1_m[i] / (sqrt(b1_v[i]) + eps);}
    for (i=0;i<WIDTH_2;i++){
        for (j=0;j<WIDTH_1;j++){
            W2_m[i][j] = beta_1*W2_m[i][j]+(1-beta_1)*gradW2[i][j];
            W2_v[i][j] = beta_2*W2_v[i][j]+(1-beta_2)*pow(gradW2[i][j],2);
            W2[i][j] = W2[i][j] - (lr * bias_correction1 / sqrt(bias_correction2)) * W2_m[i][j] / (sqrt(W2_v[i][j]) + eps);
        }
        b2_m[i] = beta_1*b2_m[i]+(1-beta_1)*gradb2[i];
        b2_v[i] = beta_2*b2_v[i]+(1-beta_2)*pow(gradb2[i],2);
        b2[i] = b2[i] - (lr * bias_correction1 / sqrt(bias_correction2)) * b2_m[i] / (sqrt(b2_v[i]) + eps);
    }
    for (i=0;i<WIDTH_2;i++){
        W3_m[i] = beta_1*W3_m[i]+(1-beta_1)*gradW3[i];
        W3_v[i] = beta_2*W3_v[i]+(1-beta_2)*pow(gradW3[i],2);
        W3[i] = W3[i] - (lr * bias_correction1 / sqrt(bias_correction2)) * W3_m[i] / (sqrt(W3_v[i]) + eps);
    }
    (*b3_m) = beta_1*(*b3_m)+(1-beta_1)*(*gradb3);
    (*b3_v) = beta_2*(*b3_v)+(1-beta_2)*pow((*gradb3),2);
    (*b3) = (*b3) - (lr * bias_correction1 / sqrt(bias_correction2)) * (*b3_m) / (sqrt((*b3_v)) + eps);
}

void
neural_learn (int n_batch, int n_cols,  double **W1,  double *b1,  double **W2,  double *b2,  double *W3,  double *b3,
                       double **W1_m,  double **W1_v,  double *b1_m,  double *b1_v,  double **W2_m,  double **W2_v,  double *b2_m,
                       double *b2_v,  double *W3_m,  double *W3_v,  double *b3_m,  double *b3_v,
                       int *step_layer1, int *steps, double **features, double *targets)
{
    int i,j,k, iter;
     double elem, output;
     double *output1[n_all_samples], *output2[n_all_samples], *output3[n_all_samples], *output4[n_all_samples], output5[n_all_samples],
           *gradInput2[n_all_samples], *gradInput3[n_all_samples], *gradInput4[n_all_samples], *gradInput5[n_all_samples], gradInput[n_all_samples],
           *gradW1[WIDTH_1], gradb1[WIDTH_1], *gradW2[WIDTH_2], gradb2[WIDTH_2], gradW3[WIDTH_2], gradb3;
    for (i=0;i<n_batch;i++){
        output1[i] = palloc0(sizeof(**output1) * WIDTH_1);
        output2[i] = palloc0(sizeof(**output2) * WIDTH_1);
        output3[i] = palloc0(sizeof(**output3) * WIDTH_2);
        output4[i] = palloc0(sizeof(**output4) * WIDTH_2);
        gradInput2[i] = palloc0(sizeof(**gradInput2) * WIDTH_1);
        gradInput3[i] = palloc0(sizeof(**gradInput3) * WIDTH_1);
        gradInput4[i] = palloc0(sizeof(**gradInput4) * WIDTH_2);
        gradInput5[i] = palloc0(sizeof(**gradInput5) * WIDTH_2);
    }
    if (n_cols>0)
       for (i=0;i<WIDTH_1;i++)
           gradW1[i] = palloc0(sizeof(**gradW1) * n_cols);
    for (i=0;i<WIDTH_2;i++)
        gradW2[i] = palloc0(sizeof(**gradW2) * WIDTH_1);


    for (iter=0;iter<N_ITERS;++iter){
        zero_grad(n_batch, n_cols, gradInput2, gradInput3, gradInput4, gradInput5, gradInput,
                       gradW1, gradb1, gradW2, gradb2, gradW3, &gradb3);

        for (i=0;i<n_batch;i++)
            for (j=0;j<WIDTH_1;j++){
                elem=0;
                for (k=0;k<n_cols;k++)
                    elem=elem+(features[i][k]/C_mul)*W1[j][k];
                elem=elem+b1[j];
                output1[i][j]=elem;
            }
        for (i=0;i<n_batch;i++)
            for (j=0;j<WIDTH_1;j++){
                output2[i][j]=output1[i][j];
                if (output1[i][j]<slope * output1[i][j])
                    output2[i][j]=slope * output1[i][j];
            }
        for (i=0;i<n_batch;i++)
            for (j=0;j<WIDTH_2;j++){
                elem=0;
                for (k=0;k<WIDTH_1;k++)
                    elem=elem+output2[i][k]*W2[j][k];
                elem=elem+b2[j];
                output3[i][j]=elem;
            }
        for (i=0;i<n_batch;i++)
            for (j=0;j<WIDTH_2;j++){
                output4[i][j]=output3[i][j];
                if (output3[i][j]<slope * output3[i][j])
                    output4[i][j]=slope * output3[i][j];
            }
        for (i=0;i<n_batch;i++){
            elem=0;
            for (k=0;k<WIDTH_2;k++)
                elem=elem+output4[i][k]*W3[k];
            elem=elem+(*b3);
            output5[i]=elem;
        }
        output=0;
        for (i=0;i<n_batch;i++)
            output=output+pow(output5[i]-targets[i]/C_mul_2,2);
        output=output/n_batch;
        for (i=0;i<n_batch;i++)
            gradInput[i]=2*(output5[i]-targets[i]/C_mul_2)/n_batch;
        for (i=0;i<n_batch;i++)
            for (j=0;j<WIDTH_2;j++)
                gradInput5[i][j] = gradInput[i]*W3[j];
        for (i=0;i<WIDTH_2;i++){
            elem=0;
            for (j=0;j<n_batch;j++)
                elem=elem+gradInput[j]*output4[j][i];
            gradW3[i]=elem;
        }
        for (i=0;i<n_batch;i++)
            gradb3=gradb3+gradInput[i];
        for (i=0;i<n_batch;i++)
            for (j=0;j<WIDTH_2;j++){
                gradInput4[i][j] = gradInput5[i][j];
                if (output3[i][j] < slope * output3[i][j])
                    gradInput4[i][j] = gradInput5[i][j]*slope;
            }
        for (i=0;i<n_batch;i++)
            for (j=0;j<WIDTH_1;j++){
                elem=0;
                for (k=0;k<WIDTH_2;k++)
                    elem=elem+gradInput4[i][k]*W2[k][j];
                gradInput3[i][j]=elem;
            }
        for (i=0;i<WIDTH_2;i++)
            for (j=0;j<WIDTH_1;j++){
                elem=0;
                for (k=0;k<n_batch;k++)
                    elem=elem+gradInput4[k][i]*output2[k][j];
                gradW2[i][j]=elem;
            }
        for (i=0;i<WIDTH_2;i++)
            for (j=0;j<n_batch;j++)
                gradb2[i]=gradb2[i]+gradInput4[j][i];
        for (i=0;i<n_batch;i++)
            for (j=0;j<WIDTH_1;j++){
                gradInput2[i][j] = gradInput3[i][j];
                if (output1[i][j] < slope * output1[i][j])
                    gradInput2[i][j] = gradInput3[i][j]*slope;
            }
        for (i=0;i<WIDTH_1;i++)
            for (j=0;j<n_cols;j++){
                elem=0;
                for (k=0;k<n_batch;k++)
                    elem=elem+gradInput2[k][i]*(features[k][j]/C_mul);
                gradW1[i][j]=elem;
            }
        for (i=0;i<WIDTH_1;i++)
            for (j=0;j<n_batch;j++)
                gradb1[i]=gradb1[i]+gradInput2[j][i];
        optim_step(n_batch, n_cols, W1, b1, W2, b2, W3, b3,
                   gradW1, gradb1, gradW2, gradb2, gradW3, &gradb3,
                   W1_m, W1_v, b1_m, b1_v, W2_m, W2_v, b2_m, b2_v, W3_m, W3_v, b3_m, b3_v,
                   step_layer1, steps
                   );
        }
        if (WIDTH_1>0)
            for (i=0;i<n_batch;i++){
                pfree(output1[i]);
                pfree(output2[i]);
                pfree(gradInput2[i]);
                pfree(gradInput3[i]);
            }
        if (WIDTH_2>0)
            for (i=0;i<n_batch;i++){
                pfree(output3[i]);
                pfree(output4[i]);
                pfree(gradInput4[i]);
                pfree(gradInput5[i]);
            }
        if (n_cols>0)
            for (i=0;i<WIDTH_1;i++)
                pfree(gradW1[i]);
        if (WIDTH_1>0)
            for (i=0;i<WIDTH_2;i++)
                pfree(gradW2[i]);
}

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
            out1[i] = out1[i]+(feature[j]/C_mul)*W1[i][j];
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
    return out5*C_mul_2;
}
