#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <immintrin.h>
#include <stdint.h>
#include <string>
#include <vector>

#include "../custard-flow/include/CustardFlow.h"


#define SIZE_CLASSES 10
#define SIZE_MINI_BATCH 16
#define SIZE_OUTPUT 10
#define SIZE_HIDDEN 8
#define NUMBER_EPOCHS 100
#define PRINT_EVERY 1
#define LEARNING_RATE 0.1f
#define SIZE_TILE 256






typedef struct {
    unsigned char *images, *labels;
    int nImages, rows, cols;
} InputData;


typedef struct Layer{
    float *weights, *biases, *activations_input, *activations_output,
    *gradients_input, *gradients_output, *gradients_weights, *gradients_biases;
    size_t size_inputs, size_neurons;
    void (*activation_forward)(struct Layer* layer,  size_t size_batch);
    void (*activation_backward)(struct Layer* layer, unsigned char * labels, size_t size_batch);
    float (*generate_number)(size_t, size_t);
} Layer;

struct Model {
    std::vector<Layer> layers = {};
    std::vector<float> parameters = {};
    size_t size_layers = 0;
    size_t size_parameters = 0;
};

typedef struct {
    float *activations;
    size_t size_activations;
} Activations;

typedef struct {
    float *grads;
    size_t size_grads;
} Gradients;





void col_to_row_major(const float *src_matrix, float *dest_matrix, size_t num_rows, size_t num_cols){
    for (size_t r = 0; r < num_rows; r++){
        for (size_t c = 0; c < num_cols; c++){            
            int offset_dest = r * num_cols + c; //[i][j] 
            int offset_source = c * num_rows + r; //[j][i] 
            dest_matrix[offset_dest] = src_matrix[offset_source];
            int db_copied_value = dest_matrix[offset_dest];
            int db = 0;
        }
    }
}


void row_to_col_major(const float *src_matrix, float *dest_matrix, size_t num_rows, size_t num_cols){
    for (size_t r = 0; r < num_rows; r++){
        for (size_t c = 0; c < num_cols; c++){            
            int offset_dest = c * num_rows + r; //[j][i] 
            int offset_source = r * num_cols + c; //[i][j] 
            dest_matrix[offset_dest] = src_matrix[offset_source];
            int db_copied_value = dest_matrix[offset_dest];
            int db = 0;
        }
    }
}




void matmul_forward(Layer *layer, float *input, float *output, size_t size_batch)
{
    
    printf("mat mul naive ....\n");
    clock_t begin, end;
    double time_spent;
    begin = clock();

    for (size_t idx_sample = 0; idx_sample < size_batch; idx_sample++){
        for (size_t idx_neuron = 0; idx_neuron < layer->size_neurons; idx_neuron++){
            size_t offset_activation = idx_sample * layer->size_neurons + idx_neuron;
            layer->activations_output[offset_activation] = layer->biases[idx_neuron];
        }
    }

    for (size_t idx_image = 0; idx_image < size_batch; idx_image++) {
        for (size_t idx_neuron = 0; idx_neuron < layer->size_neurons; idx_neuron++) {
            // output[idx_image * layer->size_neurons + idx_neuron] = layer->biases[idx_neuron];
            for (size_t idx_input = 0; idx_input < layer->size_inputs; idx_input++) {
                output[idx_image * layer->size_neurons + idx_neuron] +=  // [idx_image][idx_neuron] row major
                input[idx_image * layer->size_inputs + idx_input] // [idx_image][idx_input] row major
                 * layer->weights[idx_neuron * layer->size_inputs + idx_input]; // [idx_input][idx_neuron] col major
            }
        }
    }
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time spent in matmul_forward: %f seconds\n", time_spent);
} 


     /*
     
     |a a| |a a|   |b b| b b
     |a a| |a a|   |b b| b b
     --    --      --      
     a a   a a    |b b| b b
     a a   a a    |b b| b b
     
     
     |c c| c c
     |c c| c c
     --
     c c  c c
     c c  c c


     
      a a   a a     b b |b b|
      a a   a a     b b |b b|
                         - -
     |a a| |a a|    b b |b b|
     |a a| |a a|    b b |b b|


       c c  c c
       c c  c c
       c c  |c c|
       c c  |c c|

       


     */


void matmul_forward_tiling(Layer *layer, float *input, float *output, size_t size_batch)
{
    clock_t begin, end;
    double time_spent;
    begin = clock();

    for (size_t idx_sample = 0; idx_sample < size_batch; idx_sample++) {
        for (size_t idx_neuron = 0; idx_neuron < layer->size_neurons; idx_neuron++) {
            layer->activations_output[idx_sample * layer->size_neurons + idx_neuron] = layer->biases[idx_neuron];
        }
     }

    for (size_t idx_tile_row_start = 0; idx_tile_row_start < size_batch; idx_tile_row_start += SIZE_TILE){
        for (size_t idx_tile_col_start = 0; idx_tile_col_start < layer->size_neurons; idx_tile_col_start += SIZE_TILE){
            for (size_t idx_tile_inner_start = 0; idx_tile_inner_start < layer->size_inputs; idx_tile_inner_start += SIZE_TILE){

                for (size_t idx_row_output = idx_tile_row_start; idx_row_output < idx_tile_row_start + SIZE_TILE && idx_row_output < size_batch; idx_row_output++){
                    for (size_t idx_col_output = idx_tile_col_start; idx_col_output <idx_tile_col_start + SIZE_TILE && idx_col_output < layer->size_neurons; idx_col_output++){
                        float dot_product = 0;
                        for (size_t idx_inner = idx_tile_inner_start; idx_inner < idx_tile_inner_start + SIZE_TILE && idx_inner < layer->size_inputs; idx_inner++){
                            dot_product += layer->activations_input[idx_row_output * layer->size_neurons + idx_inner] * 
                            layer->weights[idx_col_output * layer->size_neurons + idx_inner];
                        }
                        layer->activations_output[idx_row_output * layer->size_neurons + idx_col_output] += dot_product;
                    }
                }

            }
        }
    }
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time spent in matmul_tiling_forward: %f seconds\n", time_spent);

}


void matmul_forward_outer_product(Layer * layer, size_t size_batch){
    printf("matmul outer product\n");
    clock_t begin, end;
    double time_spent;
    begin = clock();

    for (size_t idx_sample = 0; idx_sample < size_batch; idx_sample++){
        for (size_t idx_neuron = 0; idx_neuron < layer->size_neurons; idx_neuron++){
            size_t offset_activation = idx_sample * layer->size_neurons + idx_neuron;
            layer->activations_output[offset_activation] = layer->biases[idx_neuron];
        }
    }

    for (size_t start_m_tile = 0; start_m_tile < size_batch; start_m_tile += SIZE_TILE){
        for (size_t start_n_tile = 0; start_n_tile < layer->size_neurons; start_n_tile += SIZE_TILE){
            for (size_t start_k_tile = 0; start_k_tile < layer->size_inputs; start_k_tile += SIZE_TILE){

                for (size_t idx_m = start_m_tile; idx_m < start_m_tile + SIZE_TILE && idx_m < size_batch; idx_m++){
                    for (size_t idx_n = start_n_tile; idx_n < start_n_tile + SIZE_TILE && idx_n < layer->size_neurons; idx_n ++){
                        float sum = 0.0f;
                        size_t base_a = idx_m * layer->size_inputs;
                        size_t base_b = idx_n * layer->size_inputs;
                        for (size_t idx_k = 0; idx_k < start_k_tile + SIZE_TILE && idx_k < layer->size_inputs; idx_k++){
                            sum += layer->activations_input[base_a + idx_k] * layer->weights[base_b + idx_k];
                        }
                        layer->activations_output[idx_m * layer->size_neurons + idx_n] += sum;
                    }
                }

            }
        }
    }

    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time spent in matmul_outer_product: %f seconds\n", time_spent);

}

/*

M = 4 N = 4 K = 4

tile_M = 2 tile_N = 2
c c
c c

a a a a   b b
a a a a   b b
          b b
          b b

a        b b
a

ab ab  + ab ab + ab ab  + ab ab
ab ab    ab ab   ab ab    ab ab

= 4ab 4ab
  4ab 4ab

*/

void old_simd_kernel(const float * tile_A, const float * tile_B, float * tile_C, size_t M, size_t N, size_t K, size_t tile_m, size_t tile_n){
    __m256 reg_array_C[8][1] = {};
    __m256 reg_col_tile_A_1;
    __m256 reg_tile_B_element;

    for (size_t idx_k = 0; idx_k < K; idx_k++){
        reg_col_tile_A_1 = _mm256_loadu_ps(&tile_A[idx_k * M]);

        
        reg_tile_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N]);

        reg_array_C[0][0] = _mm256_fmadd_ps(reg_col_tile_A_1, reg_tile_B_element, reg_array_C[0][0]);


        reg_tile_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + 1]);

        reg_array_C[1][0] = _mm256_fmadd_ps(reg_col_tile_A_1, reg_tile_B_element, reg_array_C[1][0]);


        reg_tile_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + 2]);

        reg_array_C[2][0] = _mm256_fmadd_ps(reg_col_tile_A_1, reg_tile_B_element, reg_array_C[2][0]);


        reg_tile_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + 3]);

        reg_array_C[3][0] = _mm256_fmadd_ps(reg_col_tile_A_1, reg_tile_B_element, reg_array_C[3][0]);


        reg_tile_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + 4]);

        reg_array_C[4][0] = _mm256_fmadd_ps(reg_col_tile_A_1, reg_tile_B_element, reg_array_C[4][0]);


        reg_tile_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + 5]);
        reg_array_C[5][0] = _mm256_fmadd_ps(reg_col_tile_A_1, reg_tile_B_element, reg_array_C[5][0]);

        reg_tile_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + 6]);
        reg_array_C[6][0] = _mm256_fmadd_ps(reg_col_tile_A_1, reg_tile_B_element, reg_array_C[6][0]);
        reg_tile_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + 7]);
        reg_array_C[7][0] = _mm256_fmadd_ps(reg_col_tile_A_1, reg_tile_B_element, reg_array_C[7][0]);
    }   

    for (size_t idx = 0;  idx < 8; idx++){
        _mm256_storeu_ps(&tile_C[idx * M], reg_array_C[idx][0]);
    }
}


void simd_kernel(const float * tile_A, const float * tile_B, float * C, size_t M, size_t N, size_t K, size_t tile_m, size_t tile_n, size_t offset_tile_C){
    __m256 reg_array_C[8][1] = {};
    __m256 reg_col_tile_A_1;
    __m256 reg_tile_B_element;

    for (size_t idx_k = 0; idx_k < K; idx_k++){
        reg_col_tile_A_1 = _mm256_loadu_ps(&tile_A[idx_k * M]);

        
        reg_tile_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N]);

        reg_array_C[0][0] = _mm256_fmadd_ps(reg_col_tile_A_1, reg_tile_B_element, reg_array_C[0][0]);


        reg_tile_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + 1]);

        reg_array_C[1][0] = _mm256_fmadd_ps(reg_col_tile_A_1, reg_tile_B_element, reg_array_C[1][0]);


        reg_tile_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + 2]);

        reg_array_C[2][0] = _mm256_fmadd_ps(reg_col_tile_A_1, reg_tile_B_element, reg_array_C[2][0]);


        reg_tile_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + 3]);

        reg_array_C[3][0] = _mm256_fmadd_ps(reg_col_tile_A_1, reg_tile_B_element, reg_array_C[3][0]);


        reg_tile_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + 4]);

        reg_array_C[4][0] = _mm256_fmadd_ps(reg_col_tile_A_1, reg_tile_B_element, reg_array_C[4][0]);


        reg_tile_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + 5]);
        reg_array_C[5][0] = _mm256_fmadd_ps(reg_col_tile_A_1, reg_tile_B_element, reg_array_C[5][0]);

        reg_tile_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + 6]);
        reg_array_C[6][0] = _mm256_fmadd_ps(reg_col_tile_A_1, reg_tile_B_element, reg_array_C[6][0]);
        reg_tile_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + 7]);
        reg_array_C[7][0] = _mm256_fmadd_ps(reg_col_tile_A_1, reg_tile_B_element, reg_array_C[7][0]);
    }   

    for (size_t idx = 0; idx * M + offset_tile_C < M * N && idx < 8; idx++){
        _mm256_storeu_ps(&C[idx * M + offset_tile_C], reg_array_C[idx][0]);
    }
}

void simd_kernel_div(const float * tile_A, const float * tile_B, float * C, size_t M, size_t N, size_t K, size_t tile_m, size_t tile_n, size_t offset_tile_C){
    __m256 reg_array_C[8][1] = {};
    __m256 reg_col_tile_A_1;
    __m256 reg_tile_B_element;

    for (size_t idx_k = 0; idx_k < K; idx_k++){
        reg_col_tile_A_1 = _mm256_loadu_ps(&tile_A[idx_k * M]);

        
        reg_tile_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N]);

        reg_array_C[0][0] = _mm256_fmadd_ps(reg_col_tile_A_1, reg_tile_B_element, reg_array_C[0][0]);
        if (N == 10 && offset_tile_C == 0)
        {
            float db_sum = _mm256_cvtss_f32(reg_array_C[0][0]);
            float db_input = _mm256_cvtss_f32(reg_col_tile_A_1);
            float db_z_grad = tile_B[idx_k * N];
            printf("input = %f z grad = %f  wg sum[0][0]: %f  avg = %f\n",db_input, db_z_grad, db_sum, db_sum/K);
        }
        // _mm256_cvtss_f32 extracts the lowest (first) float from a __m256 AVX register.

        reg_tile_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + 1]);

        reg_array_C[1][0] = _mm256_fmadd_ps(reg_col_tile_A_1, reg_tile_B_element, reg_array_C[1][0]);


        reg_tile_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + 2]);

        reg_array_C[2][0] = _mm256_fmadd_ps(reg_col_tile_A_1, reg_tile_B_element, reg_array_C[2][0]);


        reg_tile_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + 3]);

        reg_array_C[3][0] = _mm256_fmadd_ps(reg_col_tile_A_1, reg_tile_B_element, reg_array_C[3][0]);


        reg_tile_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + 4]);

        reg_array_C[4][0] = _mm256_fmadd_ps(reg_col_tile_A_1, reg_tile_B_element, reg_array_C[4][0]);


        reg_tile_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + 5]);
        reg_array_C[5][0] = _mm256_fmadd_ps(reg_col_tile_A_1, reg_tile_B_element, reg_array_C[5][0]);

        reg_tile_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + 6]);
        reg_array_C[6][0] = _mm256_fmadd_ps(reg_col_tile_A_1, reg_tile_B_element, reg_array_C[6][0]);
        reg_tile_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + 7]);
        reg_array_C[7][0] = _mm256_fmadd_ps(reg_col_tile_A_1, reg_tile_B_element, reg_array_C[7][0]);
    }   

    // divide by K
    // No, K is a size_t (typically an integer type), but _mm256_broadcast_ss expects a pointer to a float.
    // You need to create a float variable from K, then broadcast that float.

    float Kf = (float)K;
    __m256 reg_K = _mm256_broadcast_ss(&Kf);
    for (size_t idx = 0; idx * M + offset_tile_C < M * N && idx < 8; idx++){
        reg_array_C[idx][0] = _mm256_div_ps(reg_array_C[idx][0], reg_K);
    }

    for (size_t idx = 0; idx * M + offset_tile_C < M * N && idx < 8; idx++){
        _mm256_storeu_ps(&C[idx * M + offset_tile_C], reg_array_C[idx][0]);
    }
}


void old_simd_matmul(const float *A, const float *B, float *C, size_t M, size_t N, size_t K)
{
    const size_t tile_m = 8;
    const size_t tile_n = 8;
    size_t offset_C = 0;
    for (size_t idx_m = 0; idx_m < M; idx_m += tile_m)
    {
        for (size_t idx_n = 0; idx_n < N; idx_n += tile_n)
        {
            offset_C= idx_m + idx_n * M;
            old_simd_kernel(&A[idx_m], &B[idx_n], &C[offset_C], M, N, K, tile_m, tile_n);
        }
    }
}




void simd_matmul_b(const float *A, const float *B, float *C, size_t M, size_t N, size_t K)
{
    const size_t tile_m = 8;
    const size_t tile_n = 8;
    size_t offset_C = 0;
    for (size_t idx_m = 0; idx_m < M; idx_m += tile_m)
    {
        for (size_t idx_n = 0; idx_n < N; idx_n += tile_n)
        {
            offset_C= idx_m + idx_n * M;
            simd_kernel_div(&A[idx_m], &B[idx_n], C, M, N, K, tile_m, tile_n, offset_C);
        }
    }
}

void simd_matmul_forward(Layer * layer, size_t size_batch){
    printf("matmul simd ....\n");
    clock_t begin, end;
    double time_spent;
    begin = clock();

    float *A = (float *)calloc(size_batch * layer->size_inputs, sizeof(float));
    float *B = (float *)calloc(layer->size_inputs * layer->size_neurons, sizeof(float));
    float *C = (float *)calloc(size_batch * layer->size_neurons, sizeof(float));

    
    memcpy(A, layer->activations_input, size_batch * layer->size_inputs * sizeof(float));
    memcpy(B, layer->weights, layer->size_inputs * layer->size_neurons * sizeof(float));
    
    row_to_col_major(layer->activations_input, A, size_batch, layer->size_inputs);
    col_to_row_major(layer->weights, B, layer->size_inputs, layer->size_neurons);
    row_to_col_major(layer->activations_output, C, size_batch, layer->size_neurons);

    simd_matmul(A, B, C, size_batch, layer->size_neurons, layer->size_inputs);
    col_to_row_major(C, layer->activations_output, size_batch, layer->size_neurons);
    free(A);
    free(B);
    free(C);
    for (size_t idx_sample = 0; idx_sample < size_batch; idx_sample++){
        for (size_t idx_neuron = 0; idx_neuron < layer->size_neurons; idx_neuron++){
            size_t offset_activation = idx_sample * layer->size_neurons + idx_neuron;
            layer->activations_output[offset_activation] += layer->biases[idx_neuron];
        }
     }

    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time spent in matmul_simd_forward: %f seconds\n", time_spent);
}


void relu_forward(Layer *layer, size_t size_batch)
{
    printf("relu forward ...\n");
    clock_t begin, end;
    double time_spent;
    begin = clock();

    for (size_t idx_image = 0; idx_image < size_batch; idx_image++) {
        for (size_t idx_neuron = 0; idx_neuron < layer->size_neurons; idx_neuron++) {
            layer->activations_output[idx_image * layer->size_neurons + idx_neuron] = fmaxf(0.0f, layer->activations_input[idx_image * layer->size_neurons + idx_neuron]);
        }
    }
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time spent in relu_forward: %f seconds\n", time_spent);
}

void softmax_forward(Layer *layer, size_t size_batch)
{
    printf("softmax forward ...\n");
    clock_t begin, end;
    double time_spent;
    begin = clock();
    for (size_t idx_image = 0; idx_image < size_batch; idx_image++) {
        float max_logit = layer->activations_output[idx_image * layer->size_neurons];
        for (size_t idx_neuron = 1; idx_neuron < layer->size_neurons; idx_neuron++) {
            if (layer->activations_output[idx_image * layer->size_neurons + idx_neuron] > max_logit) {
                max_logit = layer->activations_output[idx_image * layer->size_neurons + idx_neuron];
            }
        }

        float sum_exp = 0.0f;
        for (size_t idx_neuron = 0; idx_neuron < layer->size_neurons; idx_neuron++) {
            layer->activations_output[idx_image * layer->size_neurons + idx_neuron] = expf(layer->activations_output[idx_image * layer->size_neurons + idx_neuron] - max_logit);
            sum_exp += layer->activations_output[idx_image * layer->size_neurons + idx_neuron];
        }

        for (size_t idx_neuron = 0; idx_neuron < layer->size_neurons; idx_neuron++) {
            layer->activations_output[idx_image * layer->size_neurons + idx_neuron] /= sum_exp;
        }
    }
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time spent in softmax_forward: %f seconds\n", time_spent);
}

void model_forward(Model *model, Activations *activations, InputData *data)
{
    for (size_t idx_layer = 0; idx_layer < model->size_layers; idx_layer++) {
        Layer *layer = &model->layers[idx_layer];
        // matmul_forward(layer, layer->activations_input, layer->activations_output, data->nImages);
        // matmul_forward_tiling(layer, layer->activations_input, layer->activations_output, data->nImages);
        // matmul_forward_outer_product(layer, data->nImages);
        simd_matmul_forward(layer, data->nImages);
        layer->activation_forward(layer, data->nImages);
    }
}


void loss_softmax_backward(Layer *layer, unsigned char *labels, size_t size_batch)
{
    printf("loss softmax backward ...\n");
    clock_t begin, end;
    begin = clock();
    double time_spent;
    for (size_t idx_image = 0; idx_image < size_batch; idx_image++){
        for (size_t idx_logit = 0; idx_logit < layer->size_neurons; idx_logit++){
            float label = idx_logit == labels[idx_image] ? 1.0 : 0.0;
            size_t offset_logit = idx_image * layer->size_neurons + idx_logit;
            layer->gradients_output[offset_logit] = layer->activations_output[offset_logit] - label;
        }
    }    
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time spent in loss_softmax_backward: %f seconds\n", time_spent);
}

void relu_backward(Layer *layer, unsigned char *labels, size_t size_batch)
{
    printf("relu backward ...\n");
    clock_t begin, end;
    begin = clock();
    double time_spent;
    for (size_t idx_sample = 0; idx_sample < size_batch; idx_sample++){
        for (size_t idx_neuron = 0; idx_neuron < layer->size_neurons; idx_neuron++){
            size_t offset_grad = idx_sample * layer->size_neurons + idx_neuron;
            if (layer->activations_output[offset_grad] <= 0){
                layer->gradients_output[offset_grad] = 0;
            }
            // else dAct/dPreAct  = 1 so dLoss/dPreAct = dLoss/dAct
        }
    }
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time spent in relu_backward: %f seconds\n", time_spent);
}



void matmul_backward_separate(Layer *layer, size_t size_batch)
{
    printf("matmul backward (separate loops)...\n");
    clock_t begin, end;
    double time_spent;

    // Compute gradients for weights and biases
    begin = clock();
    for (size_t idx_sample = 0; idx_sample < size_batch; idx_sample++) {
        for (size_t idx_neuron = 0; idx_neuron < layer->size_neurons; idx_neuron++) {
            size_t offset_grad_pre_act = idx_sample * layer->size_neurons + idx_neuron;
            if (layer->gradients_biases) {
                layer->gradients_biases[idx_neuron] += layer->gradients_output[offset_grad_pre_act];
            }
            for (size_t idx_weight = 0; idx_weight < layer->size_inputs; idx_weight++) {
                size_t offset_weight = idx_neuron * layer->size_inputs + idx_weight;
                size_t offset_input = idx_sample * layer->size_inputs + idx_weight;
                layer->gradients_weights[offset_weight] += layer->activations_input[offset_input] * layer->gradients_output[offset_grad_pre_act];
                if (layer->size_inputs == 8 && idx_weight == 0 && idx_neuron == 0) {
                    printf("input %f, z grad %f, gradient output sum %f, gradient output average: %f\n", 
                        idx_weight, layer->activations_input[offset_input], layer->gradients_output[offset_grad_pre_act], 
                        layer->gradients_weights[offset_weight], layer->gradients_weights[offset_weight] / size_batch);    
                }
            }
        }
    }
    for (size_t idx_neuron = 0; idx_neuron < layer->size_neurons; idx_neuron++) {
        if (layer->gradients_biases) {
            layer->gradients_biases[idx_neuron] /= size_batch;
        }
        for (size_t idx_weight = 0; idx_weight < layer->size_inputs; idx_weight++) {
            size_t offset_weight = idx_neuron * layer->size_inputs + idx_weight;
            layer->gradients_weights[offset_weight] /= size_batch;
        }
    }
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time spent in matmul_backward (weights/biases): %f seconds\n", time_spent);

    // Compute gradients for input activations
    begin = clock();
    if (layer->gradients_input) {
        for (size_t idx_sample = 0; idx_sample < size_batch; idx_sample++) {
            for (size_t idx_input = 0; idx_input < layer->size_inputs; idx_input++) {
                size_t offset_input = idx_sample * layer->size_inputs + idx_input;
                float grad_input = 0.0f;
                for (size_t idx_neuron = 0; idx_neuron < layer->size_neurons; idx_neuron++) {
                    size_t offset_weight = idx_neuron * layer->size_inputs + idx_input;
                    size_t offset_grad_pre_act = idx_sample * layer->size_neurons + idx_neuron;
                    grad_input += layer->weights[offset_weight] * layer->gradients_output[offset_grad_pre_act];
                }
            }
        }
    }
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time spent in matmul_backward (input gradients): %f seconds\n", time_spent);
}




void matmul_backward(Layer * layer, size_t size_batch)
{
    printf("matmul backward ...\n");
    clock_t begin, end;
    begin = clock();
    double time_spent;
    for (size_t idx_sample = 0; idx_sample < size_batch; idx_sample++){
        for (size_t idx_neuron = 0; idx_neuron < layer->size_neurons; idx_neuron++){
            size_t offset_grad_pre_act = idx_sample * layer->size_neurons + idx_neuron;
            if (layer->gradients_biases){
                layer->gradients_biases[idx_neuron] += layer->gradients_output[offset_grad_pre_act];
            }
            for (size_t idx_weight = 0; idx_weight < layer->size_inputs; idx_weight++){
                size_t offset_weight = idx_neuron * layer->size_inputs + idx_weight;
                size_t offset_input = idx_sample * layer->size_inputs + idx_weight;
                layer->gradients_weights[offset_weight] += layer->activations_input[offset_input] * layer->gradients_output[offset_grad_pre_act];
                if (layer->gradients_input){
                    layer->gradients_input[offset_input] += layer->weights[offset_weight] * layer->gradients_output[offset_grad_pre_act];
                }
            }
        }
    }
    for (size_t idx_neuron = 0; idx_neuron < layer->size_neurons; idx_neuron++){
        if (layer->gradients_biases){
            layer->gradients_biases[idx_neuron] /= size_batch;
        }
        for (size_t idx_weight = 0; idx_weight < layer->size_inputs; idx_weight++){
            size_t offset_weight = idx_neuron * layer->size_inputs + idx_weight;
            layer->gradients_weights[offset_weight] /= size_batch;
        }
    }
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time spent in matmul_backward: %f seconds\n", time_spent);
}

void simd_matmul_backward(Layer * layer, size_t size_batch)
{
    printf("matmul simd backward weight grads ...\n");
    clock_t begin, end;
    double time_spent;
    begin = clock();
    float *A = (float *)calloc(layer->size_inputs * size_batch, sizeof(float));
    float *B = (float *)calloc(size_batch * layer->size_neurons, sizeof(float));
    float *C = (float *)calloc(layer->size_inputs * layer->size_neurons, sizeof(float));

    // row_to_col_major(layer->activations_input, A, size_batch, layer->size_inputs);
    // col_to_row_major(layer->gradients_output, B, size_batch, layer->size_neurons);
    memcpy(A, layer->activations_input, size_batch * layer->size_inputs * sizeof(float));
    memcpy(B, layer->gradients_output, size_batch * layer->size_neurons * sizeof(float));
    simd_matmul_b(A, B, C, layer->size_inputs, layer->size_neurons, size_batch);
    col_to_row_major(C, layer->gradients_weights, layer->size_inputs, layer->size_neurons);
    free(A);
    free(B);
    free(C);
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time spent in simd_matmul_backward weight grads: %f seconds\n", time_spent);
    printf("matmul simd backward input grads ...\n");
    begin = clock();
    for (size_t idx_sample = 0; idx_sample < size_batch; idx_sample++){
        for (size_t idx_neuron = 0; idx_neuron < layer->size_neurons; idx_neuron++){
            size_t offset_grad_pre_act = idx_sample * layer->size_neurons + idx_neuron;
            if (layer->gradients_biases){
                layer->gradients_biases[idx_neuron] += layer->gradients_output[offset_grad_pre_act];
            }
            for (size_t idx_weight = 0; idx_weight < layer->size_inputs; idx_weight++){
                size_t offset_weight = idx_neuron * layer->size_inputs + idx_weight;
                size_t offset_input = idx_sample * layer->size_inputs + idx_weight;
                if (layer->gradients_input){
                    layer->gradients_input[offset_input] += layer->weights[offset_weight] * layer->gradients_output[offset_grad_pre_act];
                }
            }
        }
    }
    for (size_t idx_neuron = 0; idx_neuron < layer->size_neurons; idx_neuron++){
        if (layer->gradients_biases){
            layer->gradients_biases[idx_neuron] /= size_batch;
        }
    }
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time spent for input activation grads: %f seconds\n", time_spent);
}



void update_layer(Layer *layer, float learning_rate)
{
    for (size_t idx_weight = 0; idx_weight < layer->size_inputs * layer->size_neurons; idx_weight++) {
        layer->weights[idx_weight] -= learning_rate * layer->gradients_weights[idx_weight];
    }
    for (size_t idx_bias = 0; idx_bias < layer->size_neurons; idx_bias++) {
        layer->biases[idx_bias] -= learning_rate * layer->gradients_biases[idx_bias];
    }
}


void model_backward(Model *model, Activations *activations, InputData *data)
{
    for (int idx_layer = model->size_layers - 1; idx_layer >= 0; idx_layer--) {
        Layer *layer = &model->layers[idx_layer];
        layer->activation_backward(layer, data->labels, data->nImages);
        matmul_backward(layer, data->nImages);
        // matmul_backward_separate(layer, data->nImages);
        // simd_matmul_backward(layer, data->nImages);
        printf("Layer %d weight grad [0][0] = %f\n", idx_layer, layer->gradients_weights[0]);
        update_layer(layer, LEARNING_RATE);
    }
}


void file_read(void *ptr, size_t size, size_t count, FILE *file) {
    if (fread(ptr, size, count, file) != count) {
        perror("Error reading file");
        exit(EXIT_FAILURE);
    }
}

void read_mnist_images(const char *filename, InputData *data) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Could not open file %s\n", filename);
        exit(1);
    }

    int temp;
    file_read(&temp, sizeof(int), 1, file);
    file_read(&data->nImages, sizeof(int), 1, file);
    data->nImages = __builtin_bswap32(data->nImages);

    file_read(&data->rows, sizeof(int), 1, file);
    file_read(&data->cols, sizeof(int), 1, file);
    data->rows = __builtin_bswap32(data->rows);
    data->cols = __builtin_bswap32(data->cols);
    printf("rows: %d, cols: %d\n", data->rows, data->cols);

    data->images = (unsigned char *)malloc(data->nImages * data->rows * data->cols);
    
    file_read(data->images, sizeof(unsigned char), data->nImages * data->rows * data->cols, file);
    fclose(file);

}


void read_mnist_labels(const char *filename, unsigned char **labels, int *nLabels) {
    FILE *file = fopen(filename, "rb");
    if (!file) exit(1);

    int temp;

    file_read(&temp, sizeof(int), 1, file);
    file_read(nLabels, sizeof(int), 1, file);
    *nLabels = __builtin_bswap32(*nLabels);

    *labels = (unsigned char *)malloc((*nLabels) * sizeof(unsigned char));
    file_read(*labels, sizeof(unsigned char), *nLabels, file);
    fclose(file);
}



void save_model(Model * model, const char *dir_path){
    char timestamp[32];
    time_t now;
    struct tm *ts;

    // Get current time
    time(&now);

    // Convert it to a human-readable format
    ts = localtime(&now);
    strftime(timestamp, sizeof timestamp, "%Y%m%d_%H%M%S", ts);

    // Create file name with the timestamp
    char filename_time[128];
    sprintf(filename_time, "%smodel_%s_h%d.mdl", dir_path, timestamp, SIZE_HIDDEN);

    printf("\nsaving in file: %s\n", filename_time);
    FILE *file = fopen(filename_time, "wb");
    if (file == NULL) {
        printf("Error opening file %s", dir_path);
        return;
    }



    // Save all parameters of the model to the file
    fwrite(model->layers[0].weights, sizeof(float), model->size_parameters, file);

    fclose(file);
}

void load_model_from_file(Model * model, const char *filename){
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Error opening file %s", filename);
        return;
    }
    // Load all parameters of the model from the file
    file_read(model->layers[0].weights, sizeof(float), model->size_parameters, file);
    fclose(file);
}

int load_model(Model * model, const char *dirname){
    DIR *dir;
    struct dirent *entry;
    struct stat file_info;
    time_t now;
    time(&now);
    time_t latest_timestamp = 0;
    char full_filename[260];
    char latest_fullname[260];

    // Open directory "models"
    dir = opendir(dirname);
    if (dir == NULL) {
        printf("Error opening directory %s", dirname);
        return 0;
    }

    // Iterate through files in the directory and find the one with the latest timestamp
    while ((entry = readdir(dir)) != NULL) {
        snprintf(full_filename, sizeof(full_filename), "%s/%s", dirname, entry->d_name);
        if (stat(full_filename, &file_info) == 0 && S_ISREG(file_info.st_mode)) {
            time_t file_timestamp = file_info.st_mtime; // Use the st_mtime field from stat() to get the last modification time

            if (file_timestamp > latest_timestamp) {
                latest_timestamp = file_timestamp;
                strcpy(latest_fullname, full_filename);
            } 
        }
    }

    closedir(dir);
    if (latest_timestamp == 0) {
        printf("\nNo model files found in directory %s\n\n", dirname);
        return 0;
    } 
    else {
        printf("\nLoading model from file %s\n\n", latest_fullname);
        load_model_from_file(model, latest_fullname); // Load model parameters from the file
        return 1;
    }

}



float generate_normal_random_number()
{
    // 2 uniformly distributed random numbers between 0 and 1
    float u1 = ((float)rand() / RAND_MAX);
    float u2 = ((float)rand() / RAND_MAX);

    // use Box-Muller to return a normally distributed number
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

// Xavier (Glorot) Normal Initialization
float generate_xavier_number(size_t inputs, size_t outputs)
{
    float stddev = sqrt(2.0f / (inputs + outputs));
    return generate_normal_random_number() * stddev;
}

// Kaiming Normal Initialization
float generate_kaiming_number(size_t inputs, size_t outputs)
{
    float stddev = sqrt(2.0f / inputs);
    return generate_normal_random_number() * stddev;
}


void initialize_layer(Layer *layer, float (*generate_number)(size_t, size_t)){
 
    for (size_t i = 0; i < layer->size_inputs * layer->size_neurons; i++)
        layer->weights[i] = generate_number(layer->size_inputs, layer->size_neurons);

    for (size_t i = 0; i < layer->size_neurons; i++)
        layer->biases[i] = 0.0f;
}

void kaiming_initialize_layer(Layer *layer, size_t inputs, size_t outputs)
{
    layer->size_inputs = inputs;
    layer->size_neurons = outputs;

    layer->activation_forward = relu_forward;
    layer->activation_backward = relu_backward;

    layer->weights = (float *)malloc(inputs * outputs * sizeof(float));
    layer->biases = (float *)malloc(outputs * sizeof(float));

    for (size_t i = 0; i < inputs * outputs; i++)
        layer->weights[i] = generate_xavier_number(inputs, outputs);

    for (size_t i = 0; i < outputs; i++)
        layer->biases[i] = 0.0f;
}

void xavier_initialize_layer(Layer *layer, size_t inputs, size_t outputs)
{
    layer->size_inputs = inputs;
    layer->size_neurons = outputs;

    layer->activation_forward = softmax_forward;
    layer->activation_backward = loss_softmax_backward;

    layer->weights = (float *)malloc(inputs * outputs * sizeof(float));
    layer->biases = (float *)malloc(outputs * sizeof(float));

    for (size_t i = 0; i < inputs * outputs; i++)
        layer->weights[i] = generate_xavier_number(inputs, outputs);

    for (size_t i = 0; i < outputs; i++)
        layer->biases[i] = 0.0f;
}



void add_layer(Model *model, size_t size_inputs, size_t size_neurons, void(*activation_forward)(Layer *layer, size_t size_batch),
               void(*activation_backward)(Layer *layer, unsigned char *labels, size_t size_batch),
            float (*generate_number)(size_t, size_t))
{
    Layer *layer = (Layer *)calloc(1, sizeof(Layer));
    layer->size_inputs = size_inputs;
    layer->size_neurons = size_neurons;
    layer->activation_forward = activation_forward;
    layer->activation_backward = activation_backward;
    layer->generate_number = generate_number;
    
    model->layers.push_back(*layer);
    model->size_layers++;
    model->size_parameters += size_inputs * size_neurons + size_neurons;
}


void allocate_parameters_memory(Model *model)
{
    model->parameters.reserve(model->size_parameters);
    float *parameters = model->parameters.data();
    size_t offset = 0;
    for (size_t i = 0; i < model->size_layers; i++) {
        Layer *layer = &model->layers[i];
        layer->weights = parameters + offset;
        offset += layer->size_inputs * layer->size_neurons;
        layer->biases = parameters + offset;
        offset += layer->size_neurons;
    }
}

void initialize_model(Model *model)
{
    for (size_t i = 0; i < model->size_layers; i++) {
        Layer *layer = &model->layers[i];
        initialize_layer(layer, layer->generate_number);
    }
}


void print_layer_weights(Layer *layer)
{
    printf("Weights:\n");
    for (size_t i = 0; i < layer->size_inputs * layer->size_neurons; i++)
        printf("%f ", layer->weights[i]);
    printf("\n");
}


void print_layer_biases(Layer *layer)
{
    printf("Biases:\n");
    for (size_t i = 0; i < layer->size_neurons; i++)
        printf("%f ", layer->biases[i]);
    printf("\n");
}



void print_layer(Layer *layer)
{
    printf("Layer: %zu inputs, %zu outputs\n", layer->size_inputs, layer->size_neurons);
    printf("Weights:\n");
    for (size_t i = 0; i < layer->size_inputs * layer->size_neurons; i++)
        printf("%f ", layer->weights[i]);
    printf("\nBiases:\n");
    for (size_t i = 0; i < layer->size_neurons; i++)
        printf("%f ", layer->biases[i]);
    printf("\n");
}

void free_layer(Layer *layer)
{
    free(layer->weights);
    free(layer->biases);
}


void print_probs(Model *model, Activations *activations, InputData *data)
{
    printf("Probabilities:\n");
    float *probs = model->layers[model->size_layers - 1].activations_output;
    float prob_sum = 0.0f;
    for (size_t idx_image = 0; idx_image < 2; idx_image++) {
        size_t start_sample = idx_image * SIZE_CLASSES;
        prob_sum = 0.0f;
        for (size_t idx_prob = 0; idx_prob < SIZE_CLASSES; idx_prob++) {
            prob_sum += probs[start_sample + idx_prob];
            printf("%f ", probs[start_sample + idx_prob]);
        }
        printf("\n");
        printf("\nprob sum = %f\n", prob_sum);
    }
}


float get_loss(Model *model, Activations *activations, InputData *data)
{
    float loss = 0.0f;
    float *probs = model->layers[model->size_layers - 1].activations_output;
    for (size_t idx_image = 0; idx_image < data->nImages; idx_image++) {
        unsigned char label = data->labels[idx_image];
        size_t start_sample = idx_image * SIZE_CLASSES;
        loss -= logf(probs[start_sample + label]);
    }
    return loss / data->nImages;
}

int arg_max(float *probs, size_t size)
{
    int max_idx = 0;
    float max_val = probs[0];
    for (size_t i = 1; i < size; i++) {
        if (probs[i] > max_val) {
            max_val = probs[i];
            max_idx = i;
        }
    }
    return max_idx;
}

float get_accuracy(Model *model, Activations *activations, InputData *data)
{
    int correct = 0;
    float *probs = model->layers[model->size_layers - 1].activations_output;
    for (size_t idx_image = 0; idx_image < data->nImages; idx_image++) {
        unsigned char label = data->labels[idx_image];
        size_t offset_probs_dist = idx_image * SIZE_CLASSES;
        int predicted_label = arg_max(probs + offset_probs_dist, SIZE_CLASSES);
        if (predicted_label == label) {
            correct++;
        }
    }
    return (float)correct / data->nImages;
}


void initialise_activations(Activations * activations, Model *model, InputData *data)
{
    activations->size_activations = 0;
    for (size_t i = 0; i < model->size_layers; i++) {
        activations->size_activations += model->layers[i].size_neurons;
    }
    activations->size_activations += data->rows * data->cols;
    activations->size_activations *= data->nImages;
    activations->activations = (float *)calloc(activations->size_activations, sizeof(float));

    for (size_t idx_pixel = 0; idx_pixel < data->nImages * data->rows * data->cols; idx_pixel++) {
        activations->activations[idx_pixel] = (float)data->images[idx_pixel] / 255.0f;
    }

    float *inputs = activations->activations;
    float *outputs = activations->activations + data->rows * data->cols;

    for (size_t idx_layer = 0; idx_layer < model->size_layers; idx_layer++) {
        Layer *layer = &model->layers[idx_layer];
        layer->activations_input = idx_layer == 0 ? inputs : model->layers[idx_layer - 1].activations_output;
        layer->activations_output = idx_layer == 0? outputs : outputs + model->layers[idx_layer -1].size_neurons * data->nImages;
    }
}

void free_activations(Activations * activations)
{
    free(activations->activations);
}

void initialise_gradients(Gradients * gradients, Model *model, InputData *data)
{
    for (size_t i = 0; i < model->size_layers; i++) {
        // size of gradients for parameters
        gradients->size_grads += model->layers[i].size_inputs * model->layers[i].size_neurons;
        gradients->size_grads += model->layers[i].size_neurons;
        // size of gradients for activations
        gradients->size_grads += model->layers[i].size_neurons * data->nImages;
    }
    gradients->grads = (float *)calloc(gradients->size_grads, sizeof(float));
    
    // connect gradients to layers

    for (size_t idx_layer = 0; idx_layer < model->size_layers; idx_layer++) {
        Layer *layer = &model->layers[idx_layer];
        layer->gradients_input = idx_layer == 0 ? NULL : model->layers[idx_layer - 1].gradients_output;

        layer->gradients_biases = idx_layer == 0 ? gradients->grads 
        : model->layers[idx_layer -1].gradients_output + model->layers[idx_layer -1].size_neurons * data->nImages;

        layer->gradients_weights =  layer->gradients_biases + layer->size_neurons;

        layer->gradients_output = layer->gradients_weights + layer->size_inputs * layer->size_neurons;
    }
}

void free_gradients(Gradients * gradients)
{
    free(gradients->grads);
}

void allocate_mini_batch_memory(InputData * mini_batch_data)
{
    mini_batch_data->images = (unsigned char *)calloc(mini_batch_data->nImages * mini_batch_data->rows * mini_batch_data->cols, sizeof(unsigned char));
    mini_batch_data->labels = (unsigned char *)calloc(mini_batch_data->nImages, sizeof(unsigned char));
}

void free_mini_batch_memory(InputData * mini_batch_data)
{
    free(mini_batch_data->images);
    free(mini_batch_data->labels);
}

void initialise_mini_batch(InputData * training_data, InputData * mini_batch_data){
    for (size_t idx_mini_batch_image = 0; idx_mini_batch_image < mini_batch_data->nImages; idx_mini_batch_image++) {
        size_t idx_training_image = rand() % training_data->nImages;
        mini_batch_data->images[idx_mini_batch_image] = training_data->images[idx_training_image];
        mini_batch_data->labels[idx_mini_batch_image] = training_data->labels[idx_training_image];
    }
}



int main() {
    // read input data
    InputData data_training, data_test, data_mini_batch = {0};

    std::string data_path_str = std::getenv("DATA_PATH");
    std::string models_path = std::getenv("MODELS_PATH");

    std::string training_images_path = data_path_str + "train-images.idx3-ubyte";
    std::string training_labels_path = data_path_str + "train-labels.idx1-ubyte";
    read_mnist_images(training_images_path.c_str(), &data_training);
    read_mnist_labels(training_labels_path.c_str(), &data_training.labels, &data_training.nImages);
    printf("Number of training images: %d\n", data_training.nImages);

    std::string test_images_path = data_path_str + "t10k-images.idx3-ubyte";
    std::string test_labels_path = data_path_str + "t10k-labels.idx1-ubyte";
    read_mnist_images(test_images_path.c_str(), &data_test);
    read_mnist_labels(test_labels_path.c_str(), &data_test.labels, &data_test.nImages);
    printf("Number of test images: %d\n", data_test.nImages);



    // create model
    Model model;

    printf("model layers = %zu\n", model.layers.size());
    
    // add layers to model
    add_layer(&model, data_test.cols * data_test.rows, SIZE_HIDDEN, relu_forward, relu_backward, generate_kaiming_number);
    add_layer(&model, SIZE_HIDDEN, SIZE_OUTPUT, softmax_forward, loss_softmax_backward, generate_xavier_number);


    printf("Model created with %zu layers\n", model.size_layers);
    for (size_t i = 0; i < model.size_layers; i++) {
        Layer *layer = &model.layers[i];
        printf("Layer %zu: %zu inputs, %zu neurons\n", i, layer->size_inputs, layer->size_neurons);
    }
    printf("Number of parameters: %zu\n", model.size_parameters);
    printf("Batch size: %zu\n", SIZE_MINI_BATCH);

    allocate_parameters_memory(&model);
    // load any persisted model parameters from models directory
    // otherwise initialize model parameters
    if (load_model(&model, models_path.c_str())) {
        printf("Model loaded successfully\n");
    } else {
        printf("No model found, training from scratch\n");
        initialize_model(&model);
    }

    // test loss before training
    Activations activations = {0};
    initialise_activations(&activations, &model, &data_test);
    model_forward(&model, &activations, &data_test);
    printf("Test loss before training: %f\n", get_loss(&model, &activations, &data_test));
    printf("Test accuracy before training: %f\n", get_accuracy(&model, &activations, &data_test));
    free_activations(&activations);

    // exit(0);

    // initialise activations and gradients for training
    // with mini batches

    Gradients gradients = {0};
    data_mini_batch.nImages = SIZE_MINI_BATCH;
    data_mini_batch.rows = data_training.rows;
    data_mini_batch.cols = data_training.cols;
    allocate_mini_batch_memory(&data_mini_batch);
    initialise_gradients(&gradients, &model, &data_mini_batch);
    // srand(time(NULL)); db
    srand(42);
    // training loop
    printf("\n");
    printf("\n");
    printf("training loop:\n");
    for (size_t epoch = 0; epoch < NUMBER_EPOCHS; epoch++) {
        printf("\nepoch: %zu\n", epoch);
        initialise_mini_batch(&data_training, &data_mini_batch);
        initialise_activations(&activations, &model, &data_mini_batch);
        model_forward(&model, &activations, &data_mini_batch);
        // print_probs(&model, &activations, &data_mini_batch);
        model_backward(&model, &activations, &data_mini_batch);
        printf("\n\n");
        if (epoch % PRINT_EVERY == 0) {
            printf("Training loss: %f\n", get_loss(&model, &activations, &data_mini_batch));
            printf("Training accuracy: %f\n", get_accuracy(&model, &activations, &data_mini_batch));
        }
        memset(gradients.grads, 0, gradients.size_grads * sizeof(float));
    }

    free_activations(&activations);

    // save model
    save_model(&model, models_path.c_str());

    // test loss after training
    initialise_activations(&activations, &model, &data_test);
    model_forward(&model, &activations, &data_test);
    printf("Test loss after training: %f\n", get_loss(&model, &activations, &data_test));
    printf("Test accuracy after training: %f\n", get_accuracy(&model, &activations, &data_test));



    free_mini_batch_memory(&data_mini_batch);
    free_gradients(&gradients);
    // free activations
    free_activations(&activations);
    // free input data
    free(data_training.images);
    free(data_training.labels);
    free(data_test.images);
    free(data_test.labels);

    return 0;
}