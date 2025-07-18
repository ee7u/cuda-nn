#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <string.h>

#define IMG_SIZE 3072
#define BATCH_SIZE 256
#define N_BATCHES 1000
#define N_CLASSES 10
#define HIDDEN_SIZE 32
#define ROWS_PER_FILE 10000
#define N_FILES 5
#define LR 1e-3f
#define BETA1 0.9f
#define BETA2 0.999f
#define WEIGHT_DECAY 0.0f
#define EPS 1e-8f


#define L1_OUT_SIZE (BATCH_SIZE*HIDDEN_SIZE)
#define RELU_OUT_SIZE (BATCH_SIZE*HIDDEN_SIZE)
#define L2_OUT_SIZE (BATCH_SIZE*N_CLASSES)
#define SOFTMAX_OUT_SIZE (BATCH_SIZE*N_CLASSES)
#define LOSS_SIZE (BATCH_SIZE)
#define DLOSS_SIZE (BATCH_SIZE)
#define DLOGITS_SIZE (BATCH_SIZE*N_CLASSES)
#define DINP_L2_SIZE (BATCH_SIZE*HIDDEN_SIZE)
#define DRELU_SIZE (BATCH_SIZE*HIDDEN_SIZE)
#define DINP_L1_SIZE (BATCH_SIZE*IMG_SIZE)

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
        .l1_w = params,
        .l1_b = params + IMG_SIZE*HIDDEN_SIZE,
        .l2_w = params + IMG_SIZE*HIDDEN_SIZE + HIDDEN_SIZE,
        .l2_b = params + IMG_SIZE*HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE*N_CLASSES,
        .l1_w_grad = grads,
        .l1_b_grad = grads + IMG_SIZE*HIDDEN_SIZE,
        .l2_w_grad = grads + IMG_SIZE*HIDDEN_SIZE + HIDDEN_SIZE,
        .l2_b_grad = grads + IMG_SIZE*HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE*N_CLASSES,
        .m = (float*)malloc(sizeof(float)*n_params),
        .v = (float*)malloc(sizeof(float)*n_params),
    };
    memset(model.m, 0, sizeof(float)*n_params);
    memset(model.v, 0, sizeof(float)*n_params);
    return model;
}

void model_to_device(Model* model) {
    size_t n_bytes = sizeof(float)*model->n_params;
    float* d_params;
    cudaMalloc(&d_params, n_bytes);
    cudaMemcpy(d_params, model->params, n_bytes, cudaMemcpyHostToDevice);

    float* d_grads;
    cudaMalloc(&d_grads, n_bytes);

    float* d_m;
    cudaMalloc(&d_m, n_bytes);
    cudaMemset(d_m, 0, n_bytes);

    float* d_v;
    cudaMalloc(&d_v, n_bytes);
    cudaMemset(d_v, 0, n_bytes);

    model->params = d_params;
    model->grads = d_grads;
    model->m = d_m;
    model->v = d_v;
    model->l1_w = model->params;
    model->l1_b = model->params + IMG_SIZE*HIDDEN_SIZE;
    model->l2_w = model->params + IMG_SIZE*HIDDEN_SIZE + HIDDEN_SIZE;
    model->l2_b = model->params + IMG_SIZE*HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE*N_CLASSES;
    model->l1_w_grad = model->grads;
    model->l1_b_grad = model->grads + IMG_SIZE*HIDDEN_SIZE;
    model->l2_w_grad = model->grads + IMG_SIZE*HIDDEN_SIZE + HIDDEN_SIZE;
    model->l2_b_grad = model->grads + IMG_SIZE*HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE*N_CLASSES;
}

void free_model(Model model) {
    free(model.params);
    free(model.grads);
    free(model.m);
    free(model.v);
}

void free_model_cuda(Model model) {
    cudaFree(model.params);
    cudaFree(model.grads);
    cudaFree(model.m);
    cudaFree(model.v);
}

Acts allocate_acts() {
    Acts acts = {
        .l1_out = (float*)malloc(sizeof(float)*L1_OUT_SIZE),
        .relu_out = (float*)malloc(sizeof(float)*RELU_OUT_SIZE),
        .l2_out = (float*)malloc(sizeof(float)*L2_OUT_SIZE),
        .softmax_out = (float*)malloc(sizeof(float)*SOFTMAX_OUT_SIZE),
        .loss = (float*)malloc(sizeof(float)*LOSS_SIZE),
        .dloss = (float*)malloc(sizeof(float)*DLOSS_SIZE),
        .dlogits = (float*)malloc(sizeof(float)*DLOGITS_SIZE),
        .dinp_l2 = (float*)malloc(sizeof(float)*DINP_L2_SIZE),
        .drelu = (float*)malloc(sizeof(float)*DRELU_SIZE),
        .dinp_l1 = (float*)malloc(sizeof(float)*DINP_L1_SIZE),
    };
    return acts;
}

Acts allocate_acts_device() {
    float* l1_out;
    cudaMalloc(&l1_out, sizeof(float)*L1_OUT_SIZE);

    float* relu_out;
    cudaMalloc(&relu_out, sizeof(float)*RELU_OUT_SIZE);

    float* l2_out;
    cudaMalloc(&l2_out, sizeof(float)*L2_OUT_SIZE);

    float* softmax_out;
    cudaMalloc(&softmax_out, sizeof(float)*SOFTMAX_OUT_SIZE);

    float* loss;
    cudaMalloc(&loss, sizeof(float)*LOSS_SIZE);

    float* dloss;
    cudaMalloc(&dloss, sizeof(float)*DLOSS_SIZE);

    float* dlogits;
    cudaMalloc(&dlogits, sizeof(float)*DLOGITS_SIZE);

    float* dinp_l2;
    cudaMalloc(&dinp_l2, sizeof(float)*DINP_L2_SIZE);

    float* drelu;
    cudaMalloc(&drelu, sizeof(float)*DRELU_SIZE);

    float* dinp_l1;
    cudaMalloc(&dinp_l1, sizeof(float)*DINP_L1_SIZE);

    Acts acts = {
        .l1_out = l1_out,
        .relu_out = relu_out,
        .l2_out = l2_out,
        .softmax_out = softmax_out,
        .loss = loss,
        .dloss = dloss,
        .dlogits = dlogits,
        .dinp_l2 = dinp_l2,
        .drelu = drelu,
        .dinp_l1 = dinp_l1,
    };
    return acts;
}

void copy_acts_from_device(Acts src, Acts dest) {
    cudaMemcpy(dest.l1_out, src.l1_out, sizeof(float)*L1_OUT_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(dest.relu_out, src.relu_out, sizeof(float)*RELU_OUT_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(dest.l2_out, src.l2_out, sizeof(float)*L2_OUT_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(dest.softmax_out, src.softmax_out, sizeof(float)*SOFTMAX_OUT_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(dest.loss, src.loss, sizeof(float)*LOSS_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(dest.dloss, src.dloss, sizeof(float)*DLOSS_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(dest.dlogits, src.dlogits, sizeof(float)*DLOGITS_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(dest.dinp_l2, src.dinp_l2, sizeof(float)*DINP_L2_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(dest.drelu, src.drelu, sizeof(float)*DRELU_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(dest.dinp_l1, src.dinp_l1, sizeof(float)*DINP_L1_SIZE, cudaMemcpyDeviceToHost);
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

void free_acts_cuda(Acts acts) {
    cudaFree(acts.l1_out);
    cudaFree(acts.relu_out);
    cudaFree(acts.l2_out);
    cudaFree(acts.softmax_out);
    cudaFree(acts.loss);
    cudaFree(acts.dloss);
    cudaFree(acts.dlogits);
    cudaFree(acts.dinp_l2);
    cudaFree(acts.drelu);
    cudaFree(acts.dinp_l1);
}

void zero_acts(Acts* acts) {
    memset(acts->l1_out, 0, sizeof(float)*L1_OUT_SIZE);
    memset(acts->relu_out, 0, sizeof(float)*RELU_OUT_SIZE);
    memset(acts->l2_out, 0, sizeof(float)*L2_OUT_SIZE);
    memset(acts->softmax_out, 0, sizeof(float)*SOFTMAX_OUT_SIZE);
    memset(acts->loss, 0, sizeof(float)*LOSS_SIZE);
    memset(acts->dloss, 0, sizeof(float)*DLOSS_SIZE);
    memset(acts->dlogits, 0, sizeof(float)*DLOGITS_SIZE);
    memset(acts->dinp_l2, 0, sizeof(float)*DINP_L2_SIZE);
    memset(acts->drelu, 0, sizeof(float)*DRELU_SIZE);
    memset(acts->dinp_l1, 0, sizeof(float)*DINP_L1_SIZE);
}

void zero_acts_cuda(Acts* acts) {
    cudaMemset(acts->l1_out, 0, sizeof(float)*L1_OUT_SIZE);
    cudaMemset(acts->relu_out, 0, sizeof(float)*RELU_OUT_SIZE);
    cudaMemset(acts->l2_out, 0, sizeof(float)*L2_OUT_SIZE);
    cudaMemset(acts->softmax_out, 0, sizeof(float)*SOFTMAX_OUT_SIZE);
    cudaMemset(acts->loss, 0, sizeof(float)*LOSS_SIZE);
    cudaMemset(acts->dloss, 0, sizeof(float)*DLOSS_SIZE);
    cudaMemset(acts->dlogits, 0, sizeof(float)*DLOGITS_SIZE);
    cudaMemset(acts->dinp_l2, 0, sizeof(float)*DINP_L2_SIZE);
    cudaMemset(acts->drelu, 0, sizeof(float)*DRELU_SIZE);
    cudaMemset(acts->dinp_l1, 0, sizeof(float)*DINP_L1_SIZE);
}

void linear_forward(float* w, float* bias, float* inp, float* out, int B, int C, int OC) {
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

__global__ void linear_forward_cuda(int B, int OC, int C, const float* w, const float* bias, const float* inp, float* out) {
  const uint o = blockIdx.x * blockDim.x + threadIdx.x;
  const uint b = blockIdx.y * blockDim.y + threadIdx.y;

  if (o < OC && b < B) {
    float val = bias[o];
    for (int i = 0; i < C; ++i) {
      val += w[o * C + i] * inp[b * C + i];
    }
    out[b*OC + o] = val;
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

__global__ void dinp_backward_cuda(float* dinp, const float* dout, const float* weight, const int B, const int C, const int OC) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= C || b >= B) return;

    for (int o = 0; o < OC; o++) {
        dinp[b*C + i] += weight[o*C + i]*dout[b*OC + o];
    }
}

__global__ void dweight_backward_cuda(float* dweight, float* dbias, const float* dout, const float* inp, const int B, const int C, const int OC) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int o = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= C || o >= OC) return;

    float dweight_acc = 0.0f;
    for (int b = 0; b < B; b++) {
        float d = dout[b*OC + o];
        dweight_acc += inp[b*C + i]*d;
        if (i == 0) {
            dbias[o] += d;
        }
    }

    dweight[o*C + i] = dweight_acc;
}

void matmul_backward_cuda(float* dinp, float* dweight, float* dbias, const float* dout, const float* inp, const float* weight, const int B, const int C, const int OC) {
    dim3 gridDim1(C/32, B/32, 1);
    dim3 blockDim1(32, 32, 1);
    dinp_backward_cuda<<<gridDim1, blockDim1>>>(dinp, dout, weight, B, C, OC);
    dim3 gridDim2(C/32, OC/32 >= 32 ? OC : 1, 1);
    dim3 blockDim2(32, 32, 1);
    dweight_backward_cuda<<<gridDim2, blockDim2>>>(dweight, dbias, dout, inp, B, C, OC);
}

void relu_forward(float *inp, float *out, int B, int C) {
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            out[b*C+c] = fmax(0, inp[b*C+c]);
        }
    }
}

__global__ void relu_forward_cuda(const int B, const int C, const float* acts, float* out) {
    const uint i = blockIdx.x*blockDim.x + threadIdx.x;
    const uint b = blockIdx.y*blockDim.y + threadIdx.y;
    if (i < C && b < B)
        out[i*B + b] = acts[i*B + b] >= 0.0f ? acts[i*B + b] : 0.0f;
}

void relu_backward(float* dinp, const float* inp, const float* dout, int B, int C) {
    // dinp, inp, dout are all B, C
    for (int b = 0; b < B; b++) {
        for (int i = 0; i < C; i++) {
            dinp[b*C+i] = inp[b*C+i] > 0.0f ? dout[b*C+i] : 0.0f;
        }
    }
}

__global__ void relu_backward_cuda(float* dinp, const float* inp, const float* dout, int B, int C) {
    const uint i = blockIdx.x*blockDim.x + threadIdx.x;
    const uint b = blockIdx.y*blockDim.y + threadIdx.y;
    if (i < C && b < B)
        dinp[b*C+i] = inp[b*C+i] > 0.0f ? dout[b*C+i] : 0.0f;
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

__global__ void softmax_forward_cuda(float* probs, float* logits, const int B, const int C) {
    const uint b = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= B) return;

    float max = -10000.0f;
    for (int i = 0; i < C; i++) {
        if (logits[b*C + i] > max) {
            max = logits[b*C + i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < C; i++) {
        probs[b*C + i] = expf(logits[b*C + i] - max);
        sum += probs[b*C + i];
    }

    for (int i = 0; i < C; i++) {
        probs[b*C + i] /= sum;
    }
}

void crossentropy_forward(float* losses, float* probs, int* targets, int B, int C) {
    // losses, targets B
    // probs B, C
    for (int b = 0; b < B; b++) {
        losses[b] = -logf(probs[b*C+targets[b]]);
    }
}

__global__ void crossentropy_forward_cuda(float* losses, float* probs, const int* targets, const int B, const int C) {
    const uint b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < B)
        losses[b] = -logf(probs[b*C + targets[b]]);
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

__global__ void crossentropy_softmax_backward_cuda(float* dlogits, float*dlosses, float* probs, const int* targets, const int B, const int C) {
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint b = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= C || b >= B) return;

    float dloss_mean = 1.0f / (BATCH_SIZE);
    dlosses[b] = dloss_mean;

    float dloss = dlosses[b];
    int ix = targets[b];
    float p = probs[b*C + i];
    float indicator = i == ix ? 1.0f : 0.0f;
    dlogits[b*C + i] += (p - indicator) * dloss;
}

void update_params(Model* model, float lr, float beta1, float beta2, float eps, float weight_decay, int t)
{
    for (size_t i = 0; i < model->n_params; i++) {
        float param = model->params[i];
        float grad = model->grads[i];

        float m = beta1 * model->m[i] + (1.0f - beta1) * grad;
        float v = beta2 * model->v[i] + (1.0f - beta2) * grad * grad;
        // bias-correct both moments
        float m_hat = m / (1.0f - powf(beta1, t));
        float v_hat = v / (1.0f - powf(beta2, t));

        model->m[i] = m;
        model->v[i] = v;
        model->params[i] -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
    }
}

__global__ void update_params_cuda(const int n_params, float* params, float* grads, float* ms, float* vs, const float lr, const float beta1, const float beta2, const float eps, const float weight_decay, const int t) {
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_params) return;

    float param = params[i];
    float grad = grads[i];

    float m = beta1 * ms[i] + (1.0f - beta1) * grad;
    float v = beta2 * vs[i] + (1.0f - beta2) * grad * grad;
    // bias-correct both moments
    float m_hat = m / (1.0f - powf(beta1, t));
    float v_hat = v / (1.0f - powf(beta2, t));

    ms[i] = m;
    vs[i] = v;
    params[i] -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
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

void model_forward_backward(Model* model, Acts* acts, float* imgs, int* labels) {
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
}

void model_forward_cuda(Model* model, Acts* acts, float* imgs) {
    cudaMemset(model->grads, 0, model->n_params*sizeof(float));
    zero_acts_cuda(acts);

    dim3 gridDim2(HIDDEN_SIZE/32, BATCH_SIZE/32, 1);
    dim3 blockDim2(32, 32, 1);
    linear_forward_cuda<<<gridDim2, blockDim2>>>(BATCH_SIZE, HIDDEN_SIZE, IMG_SIZE, model->l1_w, model->l1_b, imgs, acts->l1_out);

    relu_forward_cuda<<<gridDim2, blockDim2>>>(BATCH_SIZE, HIDDEN_SIZE, acts->l1_out, acts->relu_out);

    dim3 gridDim3(N_CLASSES >= 32 ? N_CLASSES/32 : 1, BATCH_SIZE/32, 1);
    dim3 blockDim3(32, 32, 1);
    linear_forward_cuda<<<gridDim3, blockDim3>>>(BATCH_SIZE, N_CLASSES, HIDDEN_SIZE, model->l2_w, model->l2_b, acts->relu_out, acts->l2_out);
}

void model_forward_backward_cuda(Model* model, Acts* acts, float* imgs, int* labels) {
    cudaMemset(model->grads, 0, model->n_params*sizeof(float));
    zero_acts_cuda(acts);

    dim3 gridDim1(BATCH_SIZE/32, 1, 1);
    dim3 blockDim1(32, 1, 1);

    dim3 gridDim2(HIDDEN_SIZE/32, BATCH_SIZE/32, 1);
    dim3 blockDim2(32, 32, 1);
    linear_forward_cuda<<<gridDim2, blockDim2>>>(BATCH_SIZE, HIDDEN_SIZE, IMG_SIZE, model->l1_w, model->l1_b, imgs, acts->l1_out);

    relu_forward_cuda<<<gridDim2, blockDim2>>>(BATCH_SIZE, HIDDEN_SIZE, acts->l1_out, acts->relu_out);

    dim3 gridDim3(N_CLASSES >= 32 ? N_CLASSES/32 : 1, BATCH_SIZE/32, 1);
    dim3 blockDim3(32, 32, 1);
    linear_forward_cuda<<<gridDim3, blockDim3>>>(BATCH_SIZE, N_CLASSES, HIDDEN_SIZE, model->l2_w, model->l2_b, acts->relu_out, acts->l2_out);

    softmax_forward_cuda<<<gridDim1, blockDim1>>>(acts->softmax_out, acts->l2_out, BATCH_SIZE, N_CLASSES);
    crossentropy_forward_cuda<<<gridDim1, blockDim1>>>(acts->loss, acts->softmax_out, labels, BATCH_SIZE, N_CLASSES);

    crossentropy_softmax_backward_cuda<<<gridDim3, blockDim3>>>(acts->dlogits, acts->dloss, acts->softmax_out, labels, BATCH_SIZE, N_CLASSES);
    matmul_backward_cuda(acts->dinp_l2, model->l2_w_grad, model->l2_b_grad, acts->dlogits, acts->relu_out, model->l2_w, BATCH_SIZE, HIDDEN_SIZE, N_CLASSES);
    relu_backward_cuda<<<gridDim2, blockDim2>>>(acts->drelu, acts->l1_out, acts->dinp_l2, BATCH_SIZE, HIDDEN_SIZE);
    matmul_backward_cuda(acts->dinp_l1, model->l1_w_grad, model->l1_b_grad, acts->drelu, imgs, model->l1_w, BATCH_SIZE, IMG_SIZE, HIDDEN_SIZE);


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
        size_t bytes_read = fread(data+i*ROWS_PER_FILE*(IMG_SIZE+1), sizeof(unsigned char), ROWS_PER_FILE*(IMG_SIZE+1), f);
        assert(bytes_read == sizeof(unsigned char)*ROWS_PER_FILE*(IMG_SIZE+1) && "Error reading data");
        fclose(f);
    }
}

bool fequal(const float a, const float b) {
    return abs(abs(a) - abs(b)) < 0.000001f;
}

void compare_cpu_gpu() {
    srand(time(NULL));

    unsigned char* data = (unsigned char*)malloc(N_FILES*ROWS_PER_FILE*(IMG_SIZE+1));
    load_data(data);

    size_t train_samples = 40000;
    unsigned char* train_data = data;

    Model model = allocate_model();
    init_linear(model.l1_w, model.l1_b, IMG_SIZE, HIDDEN_SIZE);
    init_linear(model.l2_w, model.l2_b, HIDDEN_SIZE, N_CLASSES);
    Acts acts = allocate_acts();

    int* labels = (int*)malloc(sizeof(int)*BATCH_SIZE);
    float* imgs = (float*)malloc(sizeof(float)*BATCH_SIZE*IMG_SIZE);
    get_batch(train_data, imgs, labels, train_samples);
    model_forward_backward(&model, &acts, imgs, labels);

    float* cpu_l2_w_grad = (float*)malloc(sizeof(float)*HIDDEN_SIZE*N_CLASSES);
    memcpy(cpu_l2_w_grad, model.l2_w_grad, sizeof(float)*HIDDEN_SIZE*N_CLASSES);
    float* cpu_l2_b_grad = (float*)malloc(sizeof(float)*N_CLASSES);
    memcpy(cpu_l2_b_grad, model.l2_b_grad, sizeof(float)*N_CLASSES);

    float* cpu_l1_w_grad = (float*)malloc(sizeof(float)*HIDDEN_SIZE*IMG_SIZE);
    memcpy(cpu_l1_w_grad, model.l1_w_grad, sizeof(float)*HIDDEN_SIZE*IMG_SIZE);
    float* cpu_l1_b_grad = (float*)malloc(sizeof(float)*HIDDEN_SIZE);
    memcpy(cpu_l1_b_grad, model.l1_b_grad, sizeof(float)*HIDDEN_SIZE);

    model_to_device(&model);
    Acts acts_d = allocate_acts_device();


    int* labels_d;
    cudaMalloc(&labels_d, sizeof(int)*BATCH_SIZE);
    cudaMemcpy(labels_d, labels, sizeof(int)*BATCH_SIZE, cudaMemcpyHostToDevice);

    float* imgs_d;
    cudaMalloc(&imgs_d, sizeof(float)*BATCH_SIZE*IMG_SIZE);
    cudaMemcpy(imgs_d, imgs, sizeof(float)*BATCH_SIZE*IMG_SIZE, cudaMemcpyHostToDevice);

    model_forward_backward_cuda(&model, &acts_d, imgs_d, labels_d);

    Acts gpu_acts = allocate_acts();
    copy_acts_from_device(acts_d, gpu_acts);

    float* gpu_l2_w_grad = (float*)malloc(sizeof(float)*HIDDEN_SIZE*N_CLASSES);
    cudaMemcpy(gpu_l2_w_grad, model.l2_w_grad, sizeof(float)*HIDDEN_SIZE*N_CLASSES, cudaMemcpyDeviceToHost);
    float* gpu_l2_b_grad = (float*)malloc(sizeof(float)*N_CLASSES);
    cudaMemcpy(gpu_l2_b_grad, model.l2_b_grad, sizeof(float)*N_CLASSES, cudaMemcpyDeviceToHost);

    float* gpu_l1_w_grad = (float*)malloc(sizeof(float)*HIDDEN_SIZE*IMG_SIZE);
    cudaMemcpy(gpu_l1_w_grad, model.l1_w_grad, sizeof(float)*HIDDEN_SIZE*IMG_SIZE, cudaMemcpyDeviceToHost);
    float* gpu_l1_b_grad = (float*)malloc(sizeof(float)*HIDDEN_SIZE);
    cudaMemcpy(gpu_l1_b_grad, model.l1_b_grad, sizeof(float)*HIDDEN_SIZE, cudaMemcpyDeviceToHost);

    for (int i = 0; i < L1_OUT_SIZE; i++) {
        assert(fequal(acts.l1_out[i], gpu_acts.l1_out[i]));
    }

    for (int i = 0; i < RELU_OUT_SIZE; i++) {
        assert(fequal(acts.relu_out[i], gpu_acts.relu_out[i]));
    }

    for (int i = 0; i < L2_OUT_SIZE; i++) {
        assert(fequal(acts.l2_out[i], gpu_acts.l2_out[i]));
    }

    for (int i = 0; i < SOFTMAX_OUT_SIZE; i++) {
        assert(fequal(acts.softmax_out[i], gpu_acts.softmax_out[i]));
    }

    for (int i = 0; i < LOSS_SIZE; i++) {
        assert(fequal(acts.loss[i], gpu_acts.loss[i]));
    }

    for (int i = 0; i < DLOSS_SIZE; i++) {
        assert(fequal(acts.dloss[i], gpu_acts.dloss[i]));
    }

    for (int i = 0; i < DLOGITS_SIZE; i++) {
        assert(fequal(acts.dlogits[i], gpu_acts.dlogits[i]));
    }

    for (int i = 0; i < N_CLASSES; i++) {
        assert(fequal(cpu_l2_b_grad[i], gpu_l2_b_grad[i]));
    }

    for (int i = 0; i < HIDDEN_SIZE*N_CLASSES; i++) {
        assert(fequal(cpu_l2_w_grad[i], gpu_l2_w_grad[i]));
    }

    for (int i = 0; i < DINP_L2_SIZE; i++) {
        assert(fequal(acts.dinp_l2[i], gpu_acts.dinp_l2[i]));
    }

    for (int i = 0; i < DRELU_SIZE; i++) {
        assert(fequal(acts.drelu[i], gpu_acts.drelu[i]));
    }

    for (int i = 0; i < DINP_L1_SIZE; i++) {
        assert(fequal(acts.dinp_l1[i], gpu_acts.dinp_l1[i]));
    }

    for (int i = 0; i < HIDDEN_SIZE*IMG_SIZE; i++) {
        assert(fequal(cpu_l1_w_grad[i], gpu_l1_w_grad[i]));
    }

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        assert(fequal(cpu_l1_b_grad[i], gpu_l1_b_grad[i]));
    }

    printf("Tests passed!\n");
}

void train_cpu() {
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

    clock_t total_time = 0;
    for (int batch_i = 0; batch_i < N_BATCHES; batch_i++) {
        printf("\b%d/%d steps\r", batch_i+1, N_BATCHES);
        fflush(stdout);

        clock_t batch_start = clock();
        get_batch(train_data, imgs, labels, train_samples);
        model_forward_backward(&model, &acts, imgs, labels);
        update_params(&model, LR, BETA1, BETA2, EPS, WEIGHT_DECAY, batch_i+1);
        total_time += (clock() - batch_start);
    }

    double avg_batch_time = ((double)total_time) / CLOCKS_PER_SEC / N_BATCHES;
    printf("\nAverage batch time (CPU): %.6f seconds\n", avg_batch_time);

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
}

void train_gpu() {
    unsigned char* data = (unsigned char*)malloc(N_FILES*ROWS_PER_FILE*(IMG_SIZE+1));
    load_data(data);

    size_t train_samples = 40000;
    size_t test_samples = 10000;
    unsigned char* train_data = data;
    unsigned char* test_data = data+train_samples*(IMG_SIZE+1);

    Model model = allocate_model();
    init_linear(model.l1_w, model.l1_b, IMG_SIZE, HIDDEN_SIZE);
    init_linear(model.l2_w, model.l2_b, HIDDEN_SIZE, N_CLASSES);
    model_to_device(&model);
    Acts acts = allocate_acts_device();

    int* labels_d;
    cudaMalloc(&labels_d, sizeof(int)*BATCH_SIZE);
    float* imgs_d;
    cudaMalloc(&imgs_d, sizeof(float)*BATCH_SIZE*IMG_SIZE);

    int* labels = (int*)malloc(sizeof(int)*BATCH_SIZE);
    float* imgs = (float*)malloc(sizeof(float)*BATCH_SIZE*IMG_SIZE);

    clock_t total_time = 0;
    for (int batch_i = 0; batch_i < N_BATCHES; batch_i++) {
        printf("\b%d/%d steps\r", batch_i+1, N_BATCHES);
        fflush(stdout);

        clock_t batch_start = clock();
        get_batch(train_data, imgs, labels, train_samples);
        cudaMemcpy(labels_d, labels, sizeof(int)*BATCH_SIZE, cudaMemcpyHostToDevice);
        cudaMemcpy(imgs_d, imgs, sizeof(float)*BATCH_SIZE*IMG_SIZE, cudaMemcpyHostToDevice);

        dim3 gridDimu(model.n_params/32, 1, 1);
        dim3 blockDimu(32, 1, 1);
        update_params_cuda<<<gridDimu,  blockDimu>>>(model.n_params, model.params, model.grads, model.m, model.v, LR, BETA1, BETA2, EPS, WEIGHT_DECAY, batch_i+1);
        model_forward_backward_cuda(&model, &acts, imgs_d, labels_d);
        total_time += (clock() - batch_start);
    }

    double avg_batch_time = ((double)total_time) / CLOCKS_PER_SEC / N_BATCHES;
    printf("\nAverage batch time (GPU): %.6f seconds\n", avg_batch_time);

    int n_test_samples = 0;
    int n_correct = 0;
    for (int i = 0; i < floor(test_samples/BATCH_SIZE); i++) {
        for (int b = 0; b < BATCH_SIZE; b++) {
            labels[b] = test_data[(i*BATCH_SIZE + b)*(IMG_SIZE+1)];
            for (int j = 0; j < IMG_SIZE; j++) {
                imgs[b*IMG_SIZE+j] = (float)(test_data[(i*BATCH_SIZE + b)*(IMG_SIZE+1)+j+1])/255;
            }
        }

        cudaMemcpy(imgs_d, imgs, sizeof(float)*BATCH_SIZE*IMG_SIZE, cudaMemcpyHostToDevice);
        model_forward_cuda(&model, &acts, imgs_d);

        Acts gpu_acts = allocate_acts();
        copy_acts_from_device(acts, gpu_acts);
        for (int b = 0; b < BATCH_SIZE; b++) {
            int pred = array_argmax(gpu_acts.l2_out+b*N_CLASSES, N_CLASSES);

            if (pred == labels[b]) {
                n_correct++;
            }
            n_test_samples++;
        }
    }
    printf("Accuracy: %f\n", ((float)n_correct)/n_test_samples);
}

int main(int argc, char* argv[]) {
    srand(time(NULL));

    if (argc == 1) {
        train_gpu();
    }
    else {
        if (strcasecmp(argv[1], "CPU") == 0) {
            train_cpu();
        } else if (strcasecmp(argv[1], "TEST") == 0) {
            compare_cpu_gpu();
        } else if (strcasecmp(argv[1], "GPU") == 0) {
            train_gpu();
        } else {
            assert(false && "Invalid argument");
        }
    }

    return 0;
}
