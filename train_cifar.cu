#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#define IMG_SIZE 3072
#define BATCH_SIZE 1024
#define N_BATCHES 1000
#define N_CLASSES 10
#define HIDDEN_SIZE 256
#define ROWS_PER_FILE 10000
#define N_FILES 5
#define LR 1e-3f
#define BETA1 0.9f
#define BETA2 0.999f
#define WEIGHT_DECAY 0.0f
#define EPS 1e-8f
#define TILE_SIZE 32

#define BLOCK_ROWS 64
#define BLOCK_COLS 64
#define BLOCKTILE_ROWS 8
#define BLOCKTILE_COLS 8

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
                val += inp[b * C + i] * w[i*OC + o];
            }
            out[b * OC + o] = val;
        }
    }
}

__global__ void linear_forward_cuda(const float* a, const float* b, const float* bias, float* c, const int AX, const int AY, const int BX) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= BX || y >= AY) return;

    float acc = bias[x];
    for (int k = 0; k < AX; k++) {
        acc += a[y * AX + k] * b[k * BX + x];
    }
    c[y * BX + x] = acc;
}

__global__ void linear_forward_cuda_tiled(const float* a, const float* b, const float* bias, float* c, const int ACOLS, const int AROWS, const int BCOLS) {
    __shared__ float as[TILE_SIZE*TILE_SIZE];
    __shared__ float bs[TILE_SIZE*TILE_SIZE];

    const uint tCol = threadIdx.x % TILE_SIZE;
    const uint tRow = threadIdx.x / TILE_SIZE;
    const uint cCol = blockIdx.y*TILE_SIZE + tCol;
    const uint cRow = blockIdx.x*TILE_SIZE + tRow;

    float acc = bias[cCol];
    for (int k = 0; k < max(ACOLS, AROWS)/TILE_SIZE; k++) {
        if (tCol + k*TILE_SIZE < ACOLS && cRow < AROWS){
            as[tRow*TILE_SIZE + tCol] = a[cRow*ACOLS + tCol + k*TILE_SIZE];
        } else {
            as[tRow*TILE_SIZE + tCol] = 0.0f;
        }

        if (TILE_SIZE*k + tRow < ACOLS && cCol < BCOLS) {
            bs[tRow*TILE_SIZE + tCol] = b[(k*TILE_SIZE + tRow)*BCOLS + cCol];
        } else {
            bs[tRow*TILE_SIZE + tCol] = 0.0f;
        }
        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            acc += as[tRow*TILE_SIZE + i] * bs[i*TILE_SIZE + tCol];
        }
        __syncthreads();
    }
    if (cCol < BCOLS && cRow < AROWS) {
        c[cRow*BCOLS + cCol] = acc;
    }
}

__global__ void linear_forward_cuda_blocktiled(const float* a, const float* b, const float* bias, float* c, const int ACOLS, const int AROWS, const int BCOLS) {
    __shared__ float as[BLOCK_ROWS*BLOCKTILE_COLS];
    __shared__ float bs[BLOCKTILE_COLS*BLOCK_COLS];

    const uint cRow = blockIdx.y*BLOCK_ROWS;
    const uint cCol = blockIdx.x*BLOCK_COLS;
    const uint tCol = threadIdx.x % BLOCK_COLS;
    const uint tRow = threadIdx.x / BLOCK_COLS;
    const uint aCol = threadIdx.x % BLOCKTILE_COLS;
    const uint aRow = threadIdx.x / BLOCKTILE_COLS;

    float threadAccs[BLOCKTILE_ROWS] = {0.0};
    for (uint i = 0; i < max(ACOLS, AROWS); i += BLOCKTILE_COLS) {
        if (cRow + aRow < AROWS && aCol + i < ACOLS) {
            as[aRow * BLOCKTILE_COLS + aCol] = a[(cRow + aRow) * ACOLS + aCol + i];
        } else {
            as[aRow * BLOCKTILE_COLS + aCol] = 0.0f;
        }
        if (tRow + i < ACOLS && tCol + cCol < BCOLS) {
            bs[tRow * BLOCK_COLS + tCol] = b[(i + tRow) * BCOLS + tCol + cCol];
        } else {
            bs[tRow * BLOCK_COLS + tCol] = 0.0f;
        }
        __syncthreads();

        for (uint tbCol = 0; tbCol < BLOCKTILE_COLS; ++tbCol) {
            float tmpB = bs[tbCol * BLOCK_COLS + tCol];
            for (uint tbRow = 0; tbRow < BLOCKTILE_ROWS; ++tbRow) {
                threadAccs[tbRow] += as[(tRow * BLOCKTILE_ROWS + tbRow)*BLOCKTILE_COLS + tbCol]*tmpB;
            }
        }
        __syncthreads();
    }

    if (cCol + tCol >= BCOLS) return;
    const float colBias = bias[cCol + tCol];
    for (uint tbRow = 0; tbRow < BLOCKTILE_ROWS; ++tbRow) {
        if ((tRow*BLOCKTILE_ROWS + cRow + tbRow) < AROWS) {
            c[(tRow*BLOCKTILE_ROWS + cRow + tbRow)*BCOLS + cCol + tCol] = threadAccs[tbRow] + colBias;
        }
    }
}

__global__ void linear_forward_relu_cuda_tiled(const float* a, const float* b, const float* bias, float* c, const int ACOLS, const int AROWS, const int BCOLS) {
    __shared__ float as[TILE_SIZE*TILE_SIZE];
    __shared__ float bs[TILE_SIZE*TILE_SIZE];

    const uint tCol = threadIdx.x % TILE_SIZE;
    const uint tRow = threadIdx.x / TILE_SIZE;
    const uint cCol = blockIdx.y*TILE_SIZE + tCol;
    const uint cRow = blockIdx.x*TILE_SIZE + tRow;

    float acc = bias[cCol];
    for (int k = 0; k < max(ACOLS, AROWS)/TILE_SIZE; k++) {
        if (tCol + k*TILE_SIZE < ACOLS && cRow < AROWS){
            as[tRow*TILE_SIZE + tCol] = a[cRow*ACOLS + tCol + k*TILE_SIZE];
        } else {
            as[tRow*TILE_SIZE + tCol] = 0.0f;
        }

        if (TILE_SIZE*k + tRow < ACOLS && cCol < BCOLS) {
            bs[tRow*TILE_SIZE + tCol] = b[(k*TILE_SIZE + tRow)*BCOLS + cCol];
        } else {
            bs[tRow*TILE_SIZE + tCol] = 0.0f;
        }
        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            acc += as[tRow*TILE_SIZE + i] * bs[i*TILE_SIZE + tCol];
        }
        __syncthreads();
    }
    if (cCol < BCOLS && cRow < AROWS) {
        c[cRow*BCOLS + cCol] = max(acc, 0.0f);
    }
}

__global__ void linear_forward_relu_cuda_blocktiled(const float* a, const float* b, const float* bias, float* c, const int ACOLS, const int AROWS, const int BCOLS) {
    __shared__ float as[BLOCK_ROWS*BLOCKTILE_COLS];
    __shared__ float bs[BLOCKTILE_COLS*BLOCK_COLS];

    const uint cRow = blockIdx.y*BLOCK_ROWS;
    const uint cCol = blockIdx.x*BLOCK_COLS;
    const uint tCol = threadIdx.x % BLOCK_COLS;
    const uint tRow = threadIdx.x / BLOCK_COLS;
    const uint aCol = threadIdx.x % BLOCKTILE_COLS;
    const uint aRow = threadIdx.x / BLOCKTILE_COLS;

    float threadAccs[BLOCKTILE_ROWS] = {0.0};
    for (uint i = 0; i < max(ACOLS, AROWS); i += BLOCKTILE_COLS) {
        if (cRow + aRow < AROWS && aCol + i < ACOLS) {
            as[aRow * BLOCKTILE_COLS + aCol] = a[(cRow + aRow) * ACOLS + aCol + i];
        } else {
            as[aRow * BLOCKTILE_COLS + aCol] = 0.0f;
        }
        if (tRow + i < ACOLS && tCol + cCol < BCOLS) {
            bs[tRow * BLOCK_COLS + tCol] = b[(i + tRow) * BCOLS + tCol + cCol];
        } else {
            bs[tRow * BLOCK_COLS + tCol] = 0.0f;
        }
        __syncthreads();

        for (uint tbCol = 0; tbCol < BLOCKTILE_COLS; ++tbCol) {
            float tmpB = bs[tbCol * BLOCK_COLS + tCol];
            for (uint tbRow = 0; tbRow < BLOCKTILE_ROWS; ++tbRow) {
                threadAccs[tbRow] += as[(tRow * BLOCKTILE_ROWS + tbRow)*BLOCKTILE_COLS + tbCol]*tmpB;
            }
        }
        __syncthreads();
    }

    if (cCol + tCol >= BCOLS) return;
    const float colBias = bias[cCol + tCol];
    for (uint tbRow = 0; tbRow < BLOCKTILE_ROWS; ++tbRow) {
        if ((tRow*BLOCKTILE_ROWS + cRow + tbRow) < AROWS) {
            c[(tRow*BLOCKTILE_ROWS + cRow + tbRow)*BCOLS + cCol + tCol] = max(threadAccs[tbRow] + colBias, 0.0f);
        }
    }
}

void matmul_backward(float* dinp, float* dweight, float* dbias,
                     const float* dout, const float* inp, const float* weight,
                     int B, int C, int OC) {
    // weight, dweight C, OC
    // dinp, inp B, C
    // dout B, OC
    // dbias OC

    for (int b = 0; b < B; b++) {
        for (int i = 0; i < C; i++) {
            for (int o = 0; o < OC; o++) {
                dinp[b*C + i] += weight[i*OC+o]*dout[b*OC + o];
            }
        }
    }

    for (int i = 0; i < C; i++) {
        for (int b = 0; b < B; b++) {
            for (int o = 0; o < OC; o++) {
                float d = dout[b*OC + o];
                if (i == 0) {
                    dbias[o] += d;
                }
                dweight[i*OC + o] += inp[b*C + i]*d;
            }
        }
    }
}

__global__ void dinp_backward_cuda(float* dinp, const float* dout, const float* weight, const int B, const int C, const int OC) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= C || b >= B) return;

    for (int o = 0; o < OC; o++) {
        dinp[b*C + i] += weight[i*OC + o]*dout[b*OC + o];
    }
}

__global__ void dinp_backward_cuda_tiled(const float* a, const float* b, float* c, const int ACOLS, const int AROWS, const int CCOLS) {
    __shared__ float as[TILE_SIZE*TILE_SIZE];
    __shared__ float bs[TILE_SIZE*TILE_SIZE];

    const uint tCol = threadIdx.x % TILE_SIZE;
    const uint tRow = threadIdx.x / TILE_SIZE;
    const uint cCol = blockIdx.y*TILE_SIZE + tCol;
    const uint cRow = blockIdx.x*TILE_SIZE + tRow;

    float acc = 0.0f;
    for (int k = 0; k < max(ACOLS, AROWS)/TILE_SIZE; k++) {
        if (tCol + k*TILE_SIZE < ACOLS && cRow < AROWS){
            as[tRow*TILE_SIZE + tCol] = a[cRow*ACOLS + tCol + k*TILE_SIZE];
        } else {
            as[tRow*TILE_SIZE + tCol] = 0.0f;
        }

        if (TILE_SIZE*k + tRow < ACOLS && cCol < CCOLS) {
            bs[tRow*TILE_SIZE + tCol] = b[cCol*ACOLS + k*TILE_SIZE + tRow];
        } else {
            bs[tRow*TILE_SIZE + tCol] = 0.0f;
        }
        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            acc += as[tRow*TILE_SIZE + i] * bs[i*TILE_SIZE + tCol];
        }
        __syncthreads();
    }
    if (cCol < CCOLS && cRow < AROWS) {
        c[cRow*CCOLS + cCol] = acc;
    }
}

__global__ void dinp_backward_cuda_blocktiled(const float* a, const float* b, float* c, const int ACOLS, const int AROWS, const int CCOLS) {
    __shared__ float as[BLOCK_ROWS*BLOCKTILE_COLS];
    __shared__ float bs[BLOCKTILE_COLS*BLOCK_COLS];

    const uint cRow = blockIdx.y*BLOCK_ROWS;
    const uint cCol = blockIdx.x*BLOCK_COLS;
    const uint tCol = threadIdx.x % BLOCK_COLS;
    const uint tRow = threadIdx.x / BLOCK_COLS;
    const uint aCol = threadIdx.x % BLOCKTILE_COLS;
    const uint aRow = threadIdx.x / BLOCKTILE_COLS;

    float threadAccs[BLOCKTILE_ROWS] = {0.0};
    for (uint i = 0; i < max(ACOLS, AROWS); i += BLOCKTILE_COLS) {
        if (cRow + aRow < AROWS && aCol + i < ACOLS) {
            as[aRow * BLOCKTILE_COLS + aCol] = a[(cRow + aRow) * ACOLS + aCol + i];
        } else {
            as[aRow * BLOCKTILE_COLS + aCol] = 0.0f;
        }
        if (tRow + i < ACOLS && tCol + cCol < CCOLS) {
            bs[tRow * BLOCK_COLS + tCol] = b[(tCol + cCol) * ACOLS + i + tRow];
        } else {
            bs[tRow * BLOCK_COLS + tCol] = 0.0f;
        }
        __syncthreads();

        for (uint tbCol = 0; tbCol < BLOCKTILE_COLS; ++tbCol) {
            float tmpB = bs[tbCol * BLOCK_COLS + tCol];
            for (uint tbRow = 0; tbRow < BLOCKTILE_ROWS; ++tbRow) {
                threadAccs[tbRow] += as[(tRow * BLOCKTILE_ROWS + tbRow)*BLOCKTILE_COLS + tbCol]*tmpB;
            }
        }
        __syncthreads();
    }

    if (cCol + tCol >= CCOLS) return;
    for (uint tbRow = 0; tbRow < BLOCKTILE_ROWS; ++tbRow) {
        if ((tRow*BLOCKTILE_ROWS + cRow + tbRow) < AROWS) {
            c[(tRow*BLOCKTILE_ROWS + cRow + tbRow)*CCOLS + cCol + tCol] = threadAccs[tbRow];
        }
    }
}

__global__ void dweight_backward_cuda(float* dweight, float* dbias, const float* dout, const float* inp, const int B, const int C, const int OC) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= C || o >= OC) return;

    float dweight_acc = 0.0f;
    float dbias_acc = 0.0f;
    for (int b = 0; b < B; b++) {
        float d = dout[b*OC + o];
        dweight_acc += inp[b*C + i]*d;
        dbias_acc += d;
    }

    if (i == 0) {
        dbias[o] = dbias_acc;
    }

    dweight[i*OC + o] = dweight_acc;
}

__global__ void dweight_backward_cuda_tiled(const float* inp, const float* dout, float* dw, float* db, const int ACOLS, const int AROWS, const int CCOLS) {
    __shared__ float as[TILE_SIZE*TILE_SIZE];
    __shared__ float bs[TILE_SIZE*TILE_SIZE];

    const uint tCol = threadIdx.x % TILE_SIZE;
    const uint tRow = threadIdx.x / TILE_SIZE;
    const uint cCol = blockIdx.y*TILE_SIZE + tCol;
    const uint cRow = blockIdx.x*TILE_SIZE + tRow;

    float dw_acc = 0.0f;
    float db_acc = 0.0f;
    for (int k = 0; k < AROWS/TILE_SIZE; k++) {
        if (tCol + k*TILE_SIZE < AROWS && cRow < ACOLS){
            as[tRow*TILE_SIZE + tCol] = inp[(tCol + k*TILE_SIZE)*ACOLS + cRow];
        } else {
            as[tRow*TILE_SIZE + tCol] = 0.0f;
        }

        if (TILE_SIZE*k + tRow < AROWS && cCol < CCOLS) {
            bs[tRow*TILE_SIZE + tCol] = dout[(k*TILE_SIZE + tRow)*CCOLS + cCol];
        } else {
            bs[tRow*TILE_SIZE + tCol] = 0.0f;
        }
        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            const float d = bs[i*TILE_SIZE + tCol];
            dw_acc += as[tRow*TILE_SIZE + i] * d;
            db_acc += d;
        }
        __syncthreads();
    }
    if (cCol < CCOLS && cRow < ACOLS) {
        dw[cRow*CCOLS + cCol] = dw_acc;
        if (cRow == 0) {
            db[cCol] = db_acc;
        }
    }
}

__global__ void dweight_backward_cuda_blocktiled(const float* inp, const float* dout, float* dw, float* db, const int ACOLS, const int AROWS, const int CCOLS) {
    __shared__ float as[BLOCK_ROWS*BLOCKTILE_COLS];
    __shared__ float bs[BLOCKTILE_COLS*BLOCK_COLS];

    const uint cRow = blockIdx.y*BLOCK_ROWS;
    const uint cCol = blockIdx.x*BLOCK_COLS;
    const uint tCol = threadIdx.x % BLOCK_COLS;
    const uint tRow = threadIdx.x / BLOCK_COLS;
    const uint aCol = threadIdx.x % BLOCKTILE_COLS;
    const uint aRow = threadIdx.x / BLOCKTILE_COLS;

    float threadDwAccs[BLOCKTILE_ROWS] = {0.0};
    float threadDbAccs[BLOCKTILE_ROWS] = {0.0};
    for (uint i = 0; i < max(ACOLS, AROWS); i += BLOCKTILE_COLS) {
        if (cRow + aRow < ACOLS && aCol + i < AROWS) {
            as[aRow * BLOCKTILE_COLS + aCol] = inp[(aCol + i) * ACOLS + cRow + aRow];
        } else {
            as[aRow * BLOCKTILE_COLS + aCol] = 0.0f;
        }
        if (tRow + i < AROWS && tCol + cCol < CCOLS) {
            bs[tRow * BLOCK_COLS + tCol] = dout[(i + tRow) * CCOLS + tCol + cCol];
        } else {
            bs[tRow * BLOCK_COLS + tCol] = 0.0f;
        }
        __syncthreads();

        for (uint tbCol = 0; tbCol < BLOCKTILE_COLS; ++tbCol) {
            float tmpB = bs[tbCol * BLOCK_COLS + tCol];
            for (uint tbRow = 0; tbRow < BLOCKTILE_ROWS; ++tbRow) {
                threadDwAccs[tbRow] += as[(tRow * BLOCKTILE_ROWS + tbRow)*BLOCKTILE_COLS + tbCol]*tmpB;
                threadDbAccs[tbRow] += tmpB;
            }
        }
        __syncthreads();
    }

    if (cCol + tCol >= CCOLS) return;
    for (uint tbRow = 0; tbRow < BLOCKTILE_ROWS; ++tbRow) {
        if ((tRow*BLOCKTILE_ROWS + cRow + tbRow) < ACOLS) {
            dw[(tRow*BLOCKTILE_ROWS + cRow + tbRow)*CCOLS + cCol + tCol] = threadDwAccs[tbRow];
            if (cRow == 0) {
                db[cCol + tCol] = threadDbAccs[tbRow];
            }
        }
    }
}

void matmul_backward_cuda(float* dinp, float* dweight, float* dbias, const float* dout, const float* inp, const float* weight, const int B, const int C, const int OC) {
    dim3 gridDim1(C/32, B/32, 1);
    dim3 blockDim1(32, 32, 1);
    dinp_backward_cuda<<<gridDim1, blockDim1>>>(dinp, dout, weight, B, C, OC);

    dim3 gridDim2(OC >= 32 ? OC/32 : 1, C >= 32 ? C/32 : 1, 1);
    dim3 blockDim2(32, 32, 1);
    dweight_backward_cuda<<<gridDim2, blockDim2>>>(dweight, dbias, dout, inp, B, C, OC);
}

void matmul_backward_cuda_tiled(float* dinp, float* dweight, float* dbias, const float* dout, const float* inp, const float* weight, const int B, const int C, const int OC) {
    // dout @ weight, dout = (B, OC), weight = (C, OC), dinp = (B, C)
    dim3 gridDim1(C>= BLOCK_COLS ? C/BLOCK_COLS : 1, B/BLOCK_ROWS);
    dim3 blockDim1((BLOCK_COLS*BLOCK_ROWS)/BLOCKTILE_ROWS);
    dinp_backward_cuda_blocktiled<<<gridDim1, blockDim1>>>(dout, weight, dinp, OC, B, C);

    // inp @ dout, transpose inp
    // dweight = (C, OC), dout = (B, OC), inp = (B, C)
    dim3 gridDim2(OC >= BLOCK_COLS ? OC/BLOCK_COLS : 1, C >= BLOCK_ROWS ? C/BLOCK_ROWS : 1);
    dim3 blockDim2((BLOCK_COLS*BLOCK_ROWS)/BLOCKTILE_ROWS);
    dweight_backward_cuda_blocktiled<<<gridDim2, blockDim2>>>(inp, dout, dweight, dbias, C, B, OC);
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
        float maxval = -10000.0f; // TODO: something better
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

void model_forward_cuda_naive(Model* model, Acts* acts, float* imgs) {
    cudaMemset(model->grads, 0, model->n_params*sizeof(float));
    zero_acts_cuda(acts);

    dim3 gridDim2(HIDDEN_SIZE/32, BATCH_SIZE/32, 1);
    dim3 blockDim2(32, 32, 1);
    linear_forward_cuda<<<gridDim2, blockDim2>>>(imgs, model->l1_w, model->l1_b, acts->l1_out, IMG_SIZE, BATCH_SIZE, HIDDEN_SIZE);

    relu_forward_cuda<<<gridDim2, blockDim2>>>(BATCH_SIZE, HIDDEN_SIZE, acts->l1_out, acts->relu_out);

    dim3 gridDim3(N_CLASSES >= 32 ? N_CLASSES/32 : 1, BATCH_SIZE/32, 1);
    dim3 blockDim3(32, 32, 1);
    linear_forward_cuda<<<gridDim3, blockDim3>>>(acts->relu_out, model->l2_w, model->l2_b, acts->l2_out, HIDDEN_SIZE, BATCH_SIZE, N_CLASSES);
}

void model_forward_backward_cuda_naive(Model* model, Acts* acts, float* imgs, int* labels) {
    cudaMemset(model->grads, 0, model->n_params*sizeof(float));
    zero_acts_cuda(acts);

    dim3 gridDim1(BATCH_SIZE/32, 1, 1);
    dim3 blockDim1(32, 1, 1);

    dim3 gridDim2(HIDDEN_SIZE/32, BATCH_SIZE/32, 1);
    dim3 blockDim2(32, 32, 1);
    linear_forward_cuda<<<gridDim2, blockDim2>>>(imgs, model->l1_w, model->l1_b, acts->l1_out, IMG_SIZE, BATCH_SIZE, HIDDEN_SIZE);

    relu_forward_cuda<<<gridDim2, blockDim2>>>(BATCH_SIZE, HIDDEN_SIZE, acts->l1_out, acts->relu_out);

    dim3 gridDim3(N_CLASSES >= 32 ? N_CLASSES/32 : 1, BATCH_SIZE/32, 1);
    dim3 blockDim3(32, 32, 1);
    linear_forward_cuda<<<gridDim3, blockDim3>>>(acts->relu_out, model->l2_w, model->l2_b, acts->l2_out, HIDDEN_SIZE, BATCH_SIZE, N_CLASSES);

    softmax_forward_cuda<<<gridDim1, blockDim1>>>(acts->softmax_out, acts->l2_out, BATCH_SIZE, N_CLASSES);
    crossentropy_forward_cuda<<<gridDim1, blockDim1>>>(acts->loss, acts->softmax_out, labels, BATCH_SIZE, N_CLASSES);

    crossentropy_softmax_backward_cuda<<<gridDim3, blockDim3>>>(acts->dlogits, acts->dloss, acts->softmax_out, labels, BATCH_SIZE, N_CLASSES);
    matmul_backward_cuda(acts->dinp_l2, model->l2_w_grad, model->l2_b_grad, acts->dlogits, acts->relu_out, model->l2_w, BATCH_SIZE, HIDDEN_SIZE, N_CLASSES);
    relu_backward_cuda<<<gridDim2, blockDim2>>>(acts->drelu, acts->l1_out, acts->dinp_l2, BATCH_SIZE, HIDDEN_SIZE);
    matmul_backward_cuda(acts->dinp_l1, model->l1_w_grad, model->l1_b_grad, acts->drelu, imgs, model->l1_w, BATCH_SIZE, IMG_SIZE, HIDDEN_SIZE);
}

void model_forward_cuda_optim(Model* model, Acts* acts, float* imgs) {
    cudaMemset(model->grads, 0, model->n_params*sizeof(float));
    zero_acts_cuda(acts);

    dim3 gridDim1(HIDDEN_SIZE/BLOCK_COLS, BATCH_SIZE/BLOCK_ROWS);
    dim3 blockDim1((BLOCK_COLS*BLOCK_ROWS)/BLOCKTILE_ROWS);
    linear_forward_relu_cuda_blocktiled<<<gridDim1, blockDim1>>>(imgs, model->l1_w, model->l1_b, acts->relu_out, IMG_SIZE, BATCH_SIZE, HIDDEN_SIZE);

    dim3 gridDim2(N_CLASSES >= BLOCK_COLS ? N_CLASSES/BLOCK_COLS : 1, BATCH_SIZE/BLOCK_ROWS);
    dim3 blockDim2((BLOCK_COLS*BLOCK_ROWS)/BLOCKTILE_ROWS);
    linear_forward_cuda_blocktiled<<<gridDim2, blockDim2>>>(acts->relu_out, model->l2_w, model->l2_b, acts->l2_out, HIDDEN_SIZE, BATCH_SIZE, N_CLASSES);
}

void model_forward_backward_cuda_optim(Model* model, Acts* acts, float* imgs, int* labels) {
    cudaMemset(model->grads, 0, model->n_params*sizeof(float));
    zero_acts_cuda(acts);

    dim3 gridDim1(HIDDEN_SIZE/BLOCK_COLS, BATCH_SIZE/BLOCK_ROWS);
    dim3 blockDim1((BLOCK_COLS*BLOCK_ROWS)/BLOCKTILE_ROWS);
    linear_forward_relu_cuda_blocktiled<<<gridDim1, blockDim1>>>(imgs, model->l1_w, model->l1_b, acts->relu_out, IMG_SIZE, BATCH_SIZE, HIDDEN_SIZE);

    dim3 gridDim2(N_CLASSES >= BLOCK_COLS ? N_CLASSES/BLOCK_COLS : 1, BATCH_SIZE/BLOCK_ROWS);
    dim3 blockDim2((BLOCK_COLS*BLOCK_ROWS)/BLOCKTILE_ROWS);
    linear_forward_cuda_blocktiled<<<gridDim2, blockDim2>>>(acts->relu_out, model->l2_w, model->l2_b, acts->l2_out, HIDDEN_SIZE, BATCH_SIZE, N_CLASSES);

    dim3 gridDim3(BATCH_SIZE/32, 1, 1);
    dim3 blockDim3(32, 1, 1);
    softmax_forward_cuda<<<gridDim3, blockDim3>>>(acts->softmax_out, acts->l2_out, BATCH_SIZE, N_CLASSES);
    crossentropy_forward_cuda<<<gridDim3, blockDim3>>>(acts->loss, acts->softmax_out, labels, BATCH_SIZE, N_CLASSES);

    dim3 gridDim4(N_CLASSES >= 32 ? N_CLASSES/32 : 1, BATCH_SIZE/32, 1);
    dim3 blockDim4(32, 32, 1);
    crossentropy_softmax_backward_cuda<<<gridDim4, blockDim4>>>(acts->dlogits, acts->dloss, acts->softmax_out, labels, BATCH_SIZE, N_CLASSES);
    matmul_backward_cuda_tiled(acts->dinp_l2, model->l2_w_grad, model->l2_b_grad, acts->dlogits, acts->relu_out, model->l2_w, BATCH_SIZE, HIDDEN_SIZE, N_CLASSES);

    dim3 gridDim5(HIDDEN_SIZE/32, BATCH_SIZE/32);
    dim3 blockDim5(32, 32, 1);
    relu_backward_cuda<<<gridDim5, blockDim5>>>(acts->drelu, acts->relu_out, acts->dinp_l2, BATCH_SIZE, HIDDEN_SIZE);
    matmul_backward_cuda_tiled(acts->dinp_l1, model->l1_w_grad, model->l1_b_grad, acts->drelu, imgs, model->l1_w, BATCH_SIZE, IMG_SIZE, HIDDEN_SIZE);
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
    return abs(abs(a) - abs(b)) < 0.00001f;
}

void compare_cpu_gpu(const char* optim) {
    printf("Testing %s\n", optim);
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

    if (strcasecmp(optim, "NAIVE") == 0) {
        model_forward_backward_cuda_naive(&model, &acts_d, imgs_d, labels_d);
    }
    else {
        model_forward_backward_cuda_optim(&model, &acts_d, imgs_d, labels_d);
    }

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

    if (strcasecmp(optim, "NAIVE") == 0) { // optim version doesn't have this because of matmul-relu kernel
        for (int i = 0; i < L1_OUT_SIZE; i++) {
            assert(fequal(acts.l1_out[i], gpu_acts.l1_out[i]));
        }
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
    printf("Training cpu\n");

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

        get_batch(train_data, imgs, labels, train_samples);
        clock_t batch_start = clock();
        model_forward_backward(&model, &acts, imgs, labels);
        update_params(&model, LR, BETA1, BETA2, EPS, WEIGHT_DECAY, batch_i+1);
        total_time += (clock() - batch_start);
    }

    double avg_batch_time = ((double)total_time) / CLOCKS_PER_SEC / N_BATCHES;
    printf("\nAverage train batch time (CPU): %.10f seconds\n", avg_batch_time);

    int n_test_samples = 0;
    int n_correct = 0;
    total_time = 0;
    for (int i = 0; i < floor(test_samples/BATCH_SIZE); i++) {
        for (int b = 0; b < BATCH_SIZE; b++) {
            labels[b] = test_data[(i*BATCH_SIZE + b)*(IMG_SIZE+1)];
            for (int j = 0; j < IMG_SIZE; j++) {
                imgs[b*IMG_SIZE+j] = (float)(test_data[(i*BATCH_SIZE + b)*(IMG_SIZE+1)+j+1])/255;
            }
        }

        clock_t batch_start = clock();
        model_forward(&model, &acts, imgs);
        total_time += (clock() - batch_start);
        for (int b = 0; b < BATCH_SIZE; b++) {
            int pred = array_argmax(acts.l2_out+b*N_CLASSES, N_CLASSES);
            if (pred == labels[b]) {
                n_correct++;
            }
            n_test_samples++;
        }
    }
    avg_batch_time = ((double)total_time) / CLOCKS_PER_SEC / N_BATCHES;
    printf("Average inference batch time (CPU): %.10f seconds\n", avg_batch_time);
    printf("Accuracy: %f\n", ((float)n_correct)/n_test_samples);
}

void train_gpu(const char* optim_str) {
    printf("Training gpu: %s\n", optim_str);
    const bool optim = strcasecmp(optim_str, "OPTIM") == 0;

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

        get_batch(train_data, imgs, labels, train_samples);
        cudaMemcpy(labels_d, labels, sizeof(int)*BATCH_SIZE, cudaMemcpyHostToDevice);
        cudaMemcpy(imgs_d, imgs, sizeof(float)*BATCH_SIZE*IMG_SIZE, cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();
        clock_t batch_start = clock();
        if (optim) {
            model_forward_backward_cuda_optim(&model, &acts, imgs_d, labels_d);
        } else {
            model_forward_backward_cuda_naive(&model, &acts, imgs_d, labels_d);
        }
        dim3 gridDimu(model.n_params/32, 1, 1);
        dim3 blockDimu(32, 1, 1);
        update_params_cuda<<<gridDimu,  blockDimu>>>(model.n_params, model.params, model.grads, model.m, model.v, LR, BETA1, BETA2, EPS, WEIGHT_DECAY, batch_i+1);
        cudaDeviceSynchronize();
        total_time += (clock() - batch_start);
    }

    double avg_batch_time = (((double)total_time) / CLOCKS_PER_SEC) / N_BATCHES;
    printf("\nAverage train batch time (GPU): %.10f seconds\n", avg_batch_time);

    int n_test_samples = 0;
    int n_correct = 0;
    total_time = 0;
    for (int i = 0; i < floor(test_samples/BATCH_SIZE); i++) {
        for (int b = 0; b < BATCH_SIZE; b++) {
            labels[b] = test_data[(i*BATCH_SIZE + b)*(IMG_SIZE+1)];
            for (int j = 0; j < IMG_SIZE; j++) {
                imgs[b*IMG_SIZE+j] = (float)(test_data[(i*BATCH_SIZE + b)*(IMG_SIZE+1)+j+1])/255;
            }
        }

        cudaMemcpy(imgs_d, imgs, sizeof(float)*BATCH_SIZE*IMG_SIZE, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        clock_t batch_start = clock();
        if (optim) {
            model_forward_cuda_optim(&model, &acts, imgs_d);
        } else {
            model_forward_cuda_naive(&model, &acts, imgs_d);
        }
        cudaDeviceSynchronize();
        total_time += (clock() - batch_start);

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
    avg_batch_time = ((double)total_time) / CLOCKS_PER_SEC / N_BATCHES;
    printf("Average inference batch time (GPU): %.10f seconds\n", avg_batch_time);
    printf("Accuracy: %f\n", ((float)n_correct)/n_test_samples);
}

int main(int argc, char* argv[]) {
    srand(time(NULL));

    if (argc == 1) {
        train_gpu("OPTIM");
    }
    else {
        if (strcasecmp(argv[1], "CPU") == 0) {
            train_cpu();
        } else if (strcasecmp(argv[1], "TEST") == 0) {
            assert(argc == 3 && "Error: no optim level provided");
            compare_cpu_gpu(argv[2]);
        } else if (strcasecmp(argv[1], "GPU") == 0) {
            assert(argc == 3 && "Error: no optim level provided");
            train_gpu(argv[2]);
        } else {
            assert(false && "Invalid argument");
        }
    }

    return 0;
}
