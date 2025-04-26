#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#define IMG_SIZE 3072
#define BATCH_SIZE 32
#define N_BATCHES 1000
#define N_CLASSES 10
#define HIDDEN_SIZE 32
#define ROWS_PER_FILE 10000
#define N_FILES 5

typedef struct Model
{
    size_t n_params;
    float* params;
    float* grads;
    float* l1_w;
    float* l1_b;
    float* l2_w;
    float* l2_b;
    float* l1_w_grad;
    float* l1_b_grad;
    float* l2_w_grad;
    float* l2_b_grad;
    float* m;
    float* v;
} Model;

typedef struct Acts
{
    float* l1_out;
    float* relu_out;
    float* l2_out;
    float* softmax_out;
    float* loss;
    float* dloss;
    float* dlogits;
    float* dinp_l2;
    float* drelu;
    float* dinp_l1;
} Acts;

Model allocate_model() {
    size_t n_params = IMG_SIZE*HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE*N_CLASSES + N_CLASSES;
    float* params = (float*)malloc(sizeof(float)*n_params);
    float* grads = (float*)malloc(sizeof(float)*n_params);
    Model model = {
        .n_params = n_params,
        .params = params,
        .grads = grads,
        .m = (float*)malloc(sizeof(float)*n_params),
        .v = (float*)malloc(sizeof(float)*n_params),
        .l1_w = params,
        .l1_w_grad = grads,
        .l1_b = params + IMG_SIZE*HIDDEN_SIZE,
        .l1_b_grad = grads + IMG_SIZE*HIDDEN_SIZE,
        .l2_w = params + IMG_SIZE*HIDDEN_SIZE + HIDDEN_SIZE,
        .l2_w_grad = grads + IMG_SIZE*HIDDEN_SIZE + HIDDEN_SIZE,
        .l2_b = params + IMG_SIZE*HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE*N_CLASSES,
        .l2_b_grad = grads + IMG_SIZE*HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE*N_CLASSES,
    };
    
    return model;
}

void free_model(Model model) {
    free(model.params);
    free(model.grads);
    free(model.m);
    free(model.v);
}

Acts allocate_acts() {
    Acts acts = {
        .l1_out = (float*)malloc(sizeof(float)*BATCH_SIZE*HIDDEN_SIZE),
        .relu_out = (float*)malloc(sizeof(float)*BATCH_SIZE*HIDDEN_SIZE),
        .l2_out = (float*)malloc(sizeof(float)*BATCH_SIZE*N_CLASSES),
        .softmax_out = (float*)malloc(sizeof(float)*BATCH_SIZE*N_CLASSES),
        .loss = (float*)malloc(sizeof(float)*BATCH_SIZE),
        .dloss = (float*)malloc(sizeof(float)*BATCH_SIZE),
        .dlogits = (float*)malloc(sizeof(float)*BATCH_SIZE*N_CLASSES),
        .dinp_l2 = (float*)malloc(sizeof(float)*BATCH_SIZE*HIDDEN_SIZE),
        .drelu = (float*)malloc(sizeof(float)*BATCH_SIZE*HIDDEN_SIZE),
        .dinp_l1 = (float*)malloc(sizeof(float)*BATCH_SIZE*IMG_SIZE),
    };
    return acts;
}

void free_acts(Acts acts) {
    free(acts.l1_out);
    free(acts.relu_out);
    free(acts.l2_out);
    free(acts.softmax_out);
    free(acts.loss);
    free(acts.dloss);
    free(acts.dlogits);
    free(acts.dinp_l2);
    free(acts.drelu);
    free(acts.dinp_l1);
}

void zero_acts(Acts* acts) {
    memset(acts->l1_out, 0, sizeof(float)*BATCH_SIZE*HIDDEN_SIZE);
    memset(acts->relu_out, 0, sizeof(float)*BATCH_SIZE*HIDDEN_SIZE);
    memset(acts->l2_out, 0, sizeof(float)*BATCH_SIZE*N_CLASSES);
    memset(acts->softmax_out, 0, sizeof(float)*BATCH_SIZE*N_CLASSES);
    memset(acts->loss, 0, sizeof(float)*BATCH_SIZE);
    memset(acts->dloss, 0, sizeof(float)*BATCH_SIZE);
    memset(acts->dlogits, 0, sizeof(float)*BATCH_SIZE*N_CLASSES);
    memset(acts->dinp_l2, 0, sizeof(float)*BATCH_SIZE*HIDDEN_SIZE);
    memset(acts->drelu, 0, sizeof(float)*BATCH_SIZE*HIDDEN_SIZE);
    memset(acts->dinp_l1, 0, sizeof(float)*BATCH_SIZE*IMG_SIZE);
}

void linear_forward(float* w, float* bias, float *inp, float *out, int B, int C, int OC) {
    // inp B, C
    // out B, OC
    for (int b = 0; b < B; b++) {
        for (int o = 0; o < OC; o++) {
            float val = bias[o];
            for (int i = 0; i < C; i++) {
                val += inp[b * C + i] * w[o*C + i];
            }
            out[b * OC + o] = val;
        }
    }
}

void matmul_backward(float* dinp, float* dweight, float* dbias,
                     const float* dout, const float* inp, const float* weight,
                     int B, int C, int OC) {
    // weight, dweight OC, C
    // dinp, inp B, C
    // dout B, OC
    // dbias OC

    for (int b = 0; b < B; b++) {
        const float* dout_bt = dout + b*OC;
        float* dinp_bt = dinp + b*C;
        for (int o = 0; o < OC; o++) {
            const float* wrow = weight + o*C;
            float d = dout_bt[o];
            for (int i = 0; i < C; i++) {
                dinp_bt[i] += wrow[i]*d;
            }
        }
    }

    for (int o = 0; o < OC; o++) {
        for (int b = 0; b < B; b++) {
            const float* dout_bt = dout + b*OC;
            const float* inp_bt = inp + b*C;
            float* dwrow = dweight + o*C;
            float d = dout_bt[o];
            dbias[o] += d;
            for (int i = 0; i < C; i++) {
                dwrow[i] += inp_bt[i]*d;
            }
        }
    }
}


void relu_forward(float *inp, float *out, int B, int C) {
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            out[b*C+c] = fmax(0, inp[b*C+c]);
        }
    }
}

void relu_backward(float* dinp, const float* inp, const float* dout, int B, int C) {
    // dinp, inp, dout are all B, C
    for (int b = 0; b < B; b++) {
        for (int i = 0; i < C; i++) {
            dinp[b*C+i] = inp[b*C+i] > 0.0f ? dout[b*C+i] : 0.0f;
        }
    }
}

void softmax_forward(float* probs, float* logits, int B, int C) {
    // probs, logits are (B,C)
    for (int b = 0; b < B; b++) {
        float* logits_bt = logits + b * C;
        float* probs_bt = probs + b * C;

        // maxval is only calculated and subtracted for numerical stability
        float maxval = -10000.0f; // TODO something better
        for (int i = 0; i < C; i++) {
            if (logits_bt[i] > maxval) {
                maxval = logits_bt[i];
            }
        }
        float sum = 0.0f;
        for (int i = 0; i < C; i++) {
            probs_bt[i] = expf(logits_bt[i] - maxval);
            sum += probs_bt[i];
        }
        for (int i = 0; i < C; i++) {
            probs_bt[i] /= sum;
        }
    }
}

void crossentropy_forward(float* losses, float* probs, int* targets, int B, int C) {
    // losses, targets B
    // probs B, C
    for (int b = 0; b < B; b++) {
        losses[b] = -logf(probs[b*C+targets[b]]);
    }
}

void crossentropy_softmax_backward(float* dlogits, float* dlosses, float* probs, int* targets, int B, int C) {
    // dlogits, probs (B, C)
    // targets, dloss B
    for (int b = 0; b < B; b++) {
        float dloss = dlosses[b];
        int ix = targets[b];
        for (int i = 0; i < C; i++) {
            float* dlogits_bt = dlogits + b * C;
            float* probs_bt = probs + b * C;
            float p = probs_bt[i];
            float indicator = i == ix ? 1.0f : 0.0f;
            dlogits_bt[i] += (p - indicator) * dloss;
        }
    }
}

void update_params(Model* model, float lr, float beta1, float beta2, float eps, float weight_decay, int t)
{
    for (size_t i = 0; i < model->n_params; i++) {
        float param = model->params[i];
        float grad = model->grads[i];

        // update the first moment (momentum)
        float m = beta1 * model->m[i] + (1.0f - beta1) * grad;
        // update the second moment (RMSprop)
        float v = beta2 * model->v[i] + (1.0f - beta2) * grad * grad;
        // bias-correct both moments
        float m_hat = m / (1.0f - powf(beta1, t));
        float v_hat = v / (1.0f - powf(beta2, t));

        model->m[i] = m;
        model->v[i] = v;
        model->params[i] -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
    }
}

void init_linear(float* w, float* b, int C, int OC) {
    // torch Linear init is uniform from -sqrt(k) to sqrt(k) where k is 1/in_features
    for (int i = 0; i < OC; i++) {
        b[i] = (rand()/(ceil(RAND_MAX/2)) - 1.0f)*sqrt(1.0f/C);
        for (int j = 0; j < C; j++) {
            w[i*C + j] = (rand()/(ceil(RAND_MAX)/2) - 1.0f)*sqrt(1.0f/C);
        }
    }
}

void get_batch(unsigned char* data, float* imgs, int* labels, size_t n_samples) {
    //TODO: dedup
    for (int b = 0; b < BATCH_SIZE; b++) {
        size_t i = rand() % n_samples;
        labels[b] = data[i*(IMG_SIZE+1)];
        for (int j = 0; j < IMG_SIZE; j++) {
            imgs[b*IMG_SIZE+j] = (float)(data[i*(IMG_SIZE+1)+j+1])/255;
        }
    }
}

void model_forward(Model* model, Acts* acts, float* imgs) {
    zero_acts(acts);

    linear_forward(model->l1_w, model->l1_b, imgs, acts->l1_out, BATCH_SIZE, IMG_SIZE, HIDDEN_SIZE);
    relu_forward(acts->l1_out, acts->relu_out, BATCH_SIZE, HIDDEN_SIZE);
    linear_forward(model->l2_w, model->l2_b, acts->relu_out, acts->l2_out, BATCH_SIZE, HIDDEN_SIZE, N_CLASSES);
}

void model_forward_backward(Model* model, Acts* acts, float* imgs, int* labels, int step) {
    memset(model->grads, 0, model->n_params*sizeof(float));
    zero_acts(acts);

    linear_forward(model->l1_w, model->l1_b, imgs, acts->l1_out, BATCH_SIZE, IMG_SIZE, HIDDEN_SIZE);
    relu_forward(acts->l1_out, acts->relu_out, BATCH_SIZE, HIDDEN_SIZE);
    linear_forward(model->l2_w, model->l2_b, acts->relu_out, acts->l2_out, BATCH_SIZE, HIDDEN_SIZE, N_CLASSES);
    softmax_forward(acts->softmax_out, acts->l2_out, BATCH_SIZE, N_CLASSES);
    crossentropy_forward(acts->loss, acts->softmax_out, labels, BATCH_SIZE, N_CLASSES);

    float dloss_mean = 1.0f / (BATCH_SIZE);
    for (int b = 0; b < BATCH_SIZE; b++) { 
        acts->dloss[b] = dloss_mean;
    }

    crossentropy_softmax_backward(acts->dlogits, acts->dloss, acts->softmax_out, labels, BATCH_SIZE, N_CLASSES);
    matmul_backward(acts->dinp_l2, model->l2_w_grad, model->l2_b_grad, acts->dlogits, acts->relu_out, model->l2_w, BATCH_SIZE, HIDDEN_SIZE, N_CLASSES);
    relu_backward(acts->drelu, acts->l1_out, acts->dinp_l2, BATCH_SIZE, HIDDEN_SIZE);
    matmul_backward(acts->dinp_l1, model->l1_w_grad, model->l1_b_grad, acts->drelu, imgs, model->l1_w, BATCH_SIZE, IMG_SIZE, HIDDEN_SIZE);

    update_params(model, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.0f, step+1);
}

int array_argmax(float* x, size_t n) {
    int ret = 0;
    for (size_t i = 0; i < n; i++) {
        if (x[i] > x[ret]) {
            ret = i;
        }
    }
    return ret;
}

void load_data(unsigned char* data) {
    for (int i = 0; i < N_FILES; i++) {
        char filename[64];
        sprintf(filename, "cifar-10-batches-bin/data_batch_%d.bin", i+1);
        FILE* f = fopen(filename, "rb");
        fread(data+i*ROWS_PER_FILE*(IMG_SIZE+1), sizeof(unsigned char), ROWS_PER_FILE*(IMG_SIZE+1), f);
        fclose(f);
    }
}

int main() {
    srand(time(NULL));

    unsigned char* data = (unsigned char*)malloc(N_FILES*ROWS_PER_FILE*(IMG_SIZE+1));
    load_data(data);

    size_t train_samples = 40000;
    size_t test_samples = 10000;
    unsigned char* train_data = data;
    unsigned char* test_data = data+train_samples*(IMG_SIZE+1);

    Model model = allocate_model();
    init_linear(model.l1_w, model.l1_b, IMG_SIZE, HIDDEN_SIZE);
    init_linear(model.l2_w, model.l2_b, HIDDEN_SIZE, N_CLASSES);
    Acts acts = allocate_acts();

    int* labels = (int*)malloc(sizeof(int)*BATCH_SIZE);
    float* imgs = (float*)malloc(sizeof(float)*BATCH_SIZE*IMG_SIZE);
    for (int batch_i = 0; batch_i < N_BATCHES; batch_i++) {
        printf("\b%d/%d steps\r", batch_i, N_BATCHES);
        fflush(stdout);
        get_batch(train_data, imgs, labels, train_samples);
        model_forward_backward(&model, &acts, imgs, labels, batch_i);
    }

    int n_test_samples = 0;
    int n_correct = 0;
    for (int i = 0; i < floor(test_samples/BATCH_SIZE); i++) {
        for (int b = 0; b < BATCH_SIZE; b++) {
            labels[b] = test_data[(i*BATCH_SIZE + b)*(IMG_SIZE+1)];
            for (int j = 0; j < IMG_SIZE; j++) {
                imgs[b*IMG_SIZE+j] = (float)(test_data[(i*BATCH_SIZE + b)*(IMG_SIZE+1)+j+1])/255;
            }
        }

        model_forward(&model, &acts, imgs);
        for (int b = 0; b < BATCH_SIZE; b++) {
            int pred = array_argmax(acts.l2_out+b*N_CLASSES, N_CLASSES);

            if (pred == labels[b]) {
                n_correct++;
            }
            n_test_samples++;
        }
    }
    printf("Accuracy: %f\n", ((float)n_correct)/n_test_samples);

    free_model(model);
    free_acts(acts);
    free(data);
    free(labels);
    free(imgs);
    return 0;
}