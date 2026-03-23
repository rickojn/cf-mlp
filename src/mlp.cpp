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
#define SIZE_MINI_BATCH 64
#define SIZE_OUTPUT 10
#define SIZE_HIDDEN 4
#define NUMBER_STEPS 1000
#define PRINT_EVERY 100
#define LEARNING_RATE 0.01f
#define SIZE_TILE 256






struct InputData {
    std::vector<unsigned char> images;
    std::vector<long> labels;
    size_t nImages, rows, cols;
};


typedef struct Layer{
    float *weights, *biases, *activations_input, *activations_output,
    *gradients_input, *gradients_output, *gradients_weights, *gradients_biases;
    size_t size_inputs, size_neurons;
    void (*activation_forward)(float * activations, size_t num_features, size_t size_batch);
    void (*activation_backward)(const float * inputs, float * gradients, const long * labels, size_t num_features, size_t size_batch);
    float (*generate_number)(size_t, size_t);
} Layer;

struct Model {
    std::vector<Layer> layers = {};
    std::vector<float> parameters = {};
    size_t size_layers = 0;
    size_t size_parameters = 0;
};

struct Activations {
    std::vector<float> activations;
    size_t size_activations;
};

struct Gradients {
    std::vector<float> grads;
    size_t size_grads;
};













void add_bias(float *output, const float *bias, size_t batch, size_t neurons) {
    for (size_t i = 0; i < batch; i++) {
        for (size_t j = 0; j < neurons; j++) {
            output[i * neurons + j] += bias[j];
        }
    }
}



void model_forward(Model *model, Activations *activations, InputData *data)
{
    for (size_t idx_layer = 0; idx_layer < model->size_layers; idx_layer++) {
        Layer *layer = &model->layers[idx_layer];
        simd_matmul(layer->activations_input, layer->weights, layer->activations_output, data->nImages, layer->size_neurons, layer->size_inputs);
        add_bias(layer->activations_output, layer->biases, data->nImages, layer->size_neurons);
        layer->activation_forward( layer->activations_output, layer->size_neurons, data->nImages);
    }
}


void loss_softmax_backward(const float *activations_output, float *gradients_output, const unsigned char *labels, size_t num_neurons, size_t size_batch)
{
    printf("loss softmax backward ...\n");
    clock_t begin, end;
    begin = clock();
    double time_spent;
    for (size_t idx_image = 0; idx_image < size_batch; idx_image++){
        for (size_t idx_logit = 0; idx_logit < num_neurons; idx_logit++){
            float label = idx_logit == labels[idx_image] ? 1.0 : 0.0;
            size_t offset_logit = idx_image * num_neurons + idx_logit;
            gradients_output[offset_logit] = activations_output[offset_logit] - label;
        }
    }    
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time spent in loss_softmax_backward: %f seconds\n", time_spent);
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


void bias_backward(const float *gradients_output, float *gradients_biases, size_t num_neurons, size_t size_batch)
{
    for (size_t idx_neuron = 0; idx_neuron < num_neurons; idx_neuron++) {
        float grad_bias = 0.0f;
        for (size_t idx_image = 0; idx_image < size_batch; idx_image++) {
            grad_bias += gradients_output[idx_image * num_neurons + idx_neuron];
        }
        gradients_biases[idx_neuron] = grad_bias;
    }
}


void model_backward(Model *model, Activations *activations, InputData *input_data)
{
    for (int idx_layer = model->size_layers - 1; idx_layer >= 0; idx_layer--) {
        Layer *layer = &model->layers[idx_layer];
        layer->activation_backward(layer->activations_output, layer->gradients_output, input_data->labels.data(), layer->size_neurons, input_data->nImages);
        simd_matmul_backwards(layer->gradients_output, layer->weights, layer->activations_input, layer->gradients_weights, layer->gradients_input, 
            input_data->nImages, layer->size_neurons, layer->size_inputs);
        bias_backward(layer->gradients_output, layer->gradients_biases, layer->size_neurons, input_data->nImages);
        update_layer(layer, LEARNING_RATE);
    }
}


void file_read(void *ptr, size_t size, size_t count, FILE *file) {
    if (fread(ptr, size, count, file) != count) {
        perror("Error reading file");
        exit(EXIT_FAILURE);
    }
}

void read_mnist_images(const char *filename, InputData *input_data) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Could not open file %s\n", filename);
        exit(1);
    }

    int mnist_magic_number;
    file_read(&mnist_magic_number, sizeof(int), 1, file);
    if (__builtin_bswap32(mnist_magic_number) != 2051) {
        printf("Invalid MNIST image file magic number: %d\n", __builtin_bswap32(mnist_magic_number));
        fclose(file);
        exit(1);
    }
    file_read(&input_data->nImages, sizeof(int), 1, file);
    input_data->nImages = __builtin_bswap32(input_data->nImages);

    file_read(&input_data->rows, sizeof(int), 1, file);
    file_read(&input_data->cols, sizeof(int), 1, file);
    input_data->rows = __builtin_bswap32(input_data->rows);
    input_data->cols = __builtin_bswap32(input_data->cols);
    printf("rows: %ld, cols: %ld\n", input_data->rows, input_data->cols);

    input_data->images.resize(input_data->nImages * input_data->rows * input_data->cols);

    file_read(input_data->images.data(), sizeof(unsigned char), input_data->nImages * input_data->rows * input_data->cols, file);
    fclose(file);

}


void read_mnist_labels(const char *filename, std::vector<long> *labels, size_t *nLabels) {
    FILE *file = fopen(filename, "rb");
    if (!file) exit(1);

    int magic;
    file_read(&magic, sizeof(int), 1, file);
    if (__builtin_bswap32(magic) != 2049) exit(1);

    file_read(nLabels, sizeof(int), 1, file);
    *nLabels = __builtin_bswap32(*nLabels);

    labels->resize(*nLabels);

    for (size_t idx_label = 0; idx_label < *nLabels; idx_label++) {
        unsigned char c;
        file_read(&c, 1, 1, file);
        (*labels)[idx_label] = c;
    }

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
    // save hidden layer weights as mnist format for visualization
    char filename_hidden[128];
    sprintf(filename_hidden, "%smodel_%s_h%d_hidden_weights.idx3-ubyte", dir_path, timestamp, SIZE_HIDDEN);
    printf("\nsaving hidden layer weights in file: %s\n", filename_hidden);
    file = fopen(filename_hidden, "wb");
    if (file == NULL) {
        printf("Error opening file %s", dir_path);
        return;
    }
    int magic_number = __builtin_bswap32(2051);
    fwrite(&magic_number, sizeof(int), 1, file);
    int n_images = __builtin_bswap32(SIZE_HIDDEN);
    fwrite(&n_images, sizeof(int), 1, file);  
    int rows = __builtin_bswap32(28);
    int cols = __builtin_bswap32(28);
    fwrite(&rows, sizeof(int), 1, file);
    fwrite(&cols, sizeof(int), 1, file);
    for (size_t idx_neuron = 0; idx_neuron < SIZE_HIDDEN; idx_neuron++) {
        for (size_t idx_input = 0; idx_input < 784; idx_input++) {
            float weight = model->layers[0].weights[idx_neuron * 784 + idx_input];
            unsigned char pixel_value = (unsigned char)(fminf(fmaxf((weight + 1.0f) / 2.0f * 255.0f, 0.0f), 255.0f));
            fwrite(&pixel_value, sizeof(unsigned char), 1, file);
        }
    }
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



void add_layer(Model *model, size_t size_inputs, size_t size_neurons, 
    void(*activation_forward)(float *activations, size_t num_classes, size_t size_batch),
               void(*activation_backward)(const float *inputs, float *gradients,  const long *labels, size_t num_features, size_t size_batch),
            float (*generate_number)(size_t, size_t))
{
    Layer layer = {};
    layer.size_inputs = size_inputs;
    layer.size_neurons = size_neurons;
    layer.activation_forward = activation_forward;
    layer.activation_backward = activation_backward;
    layer.generate_number = generate_number;
    
    model->layers.push_back(layer);
    model->size_layers++;
    model->size_parameters += size_inputs * size_neurons + size_neurons;
}


void allocate_parameters_memory(Model *model)
{
    model->parameters.resize(model->size_parameters);
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
void calculate_size(Activations *activations, Model *model, InputData *data)
{
    activations->size_activations = 0;
    for (size_t i = 0; i < model->size_layers; i++)
    {
        activations->size_activations += model->layers[i].size_neurons;
    }
    activations->size_activations += data->rows * data->cols;
    activations->size_activations *= data->nImages;
}


void initialise_activations(Activations *activations, Model *model, InputData *input_data)
{
    activations->activations.resize(activations->size_activations);

    for (size_t idx_pixel = 0; idx_pixel < input_data->nImages * input_data->rows * input_data->cols; idx_pixel++) {
        activations->activations[idx_pixel] = (float)input_data->images[idx_pixel] / 255.0f;
    }

    float *inputs = activations->activations.data();
    float *outputs = activations->activations.data() + input_data->rows * input_data->cols * input_data->nImages;

    for (size_t idx_layer = 0; idx_layer < model->size_layers; idx_layer++) {
        Layer *layer = &model->layers[idx_layer];
        layer->activations_input = idx_layer == 0 ? inputs : model->layers[idx_layer - 1].activations_output;
        layer->activations_output = idx_layer == 0? outputs : outputs + model->layers[idx_layer -1].size_neurons * input_data->nImages;
    }
}



void initialise_gradients(Gradients * gradients, Model *model, InputData *data)
{
    gradients->size_grads = 0;
    for (size_t i = 0; i < model->size_layers; i++) {
        // size of gradients for parameters
        gradients->size_grads += model->layers[i].size_inputs * model->layers[i].size_neurons;
        gradients->size_grads += model->layers[i].size_neurons;
        // size of gradients for activations
        gradients->size_grads += model->layers[i].size_neurons * data->nImages;
    }
    gradients->grads.resize(gradients->size_grads);
    
    // connect gradients to layers

    for (size_t idx_layer = 0; idx_layer < model->size_layers; idx_layer++) {
        Layer *layer = &model->layers[idx_layer];
        layer->gradients_input = idx_layer == 0 ? NULL : model->layers[idx_layer - 1].gradients_output;

        layer->gradients_biases = idx_layer == 0 ? gradients->grads.data()
                                                  : model->layers[idx_layer - 1].gradients_output + model->layers[idx_layer - 1].size_neurons * data->nImages;

        layer->gradients_weights = layer->gradients_biases + layer->size_neurons;

        layer->gradients_output = layer->gradients_weights + layer->size_inputs * layer->size_neurons;
    }
}


void allocate_mini_batch_memory(InputData * mini_batch_data)
{
    mini_batch_data->images.resize(mini_batch_data->nImages * mini_batch_data->rows * mini_batch_data->cols);
    mini_batch_data->labels.resize(mini_batch_data->nImages);
}


void initialise_mini_batch(InputData *training_data, InputData *mini_batch_data) {
    size_t image_size = training_data->rows * training_data->cols;
    for (size_t idx_batch_sample = 0; idx_batch_sample < mini_batch_data->nImages; idx_batch_sample++) {
        size_t idx_training_sample = rand() % training_data->nImages;

        memcpy(&mini_batch_data->images[idx_batch_sample * image_size],
               &training_data->images[idx_training_sample * image_size],
               image_size * sizeof(unsigned char));

        mini_batch_data->labels[idx_batch_sample] = training_data->labels[idx_training_sample];
    }
}


int main() {
    // read input data
    InputData data_training, data_test, data_mini_batch;

    std::string data_path_str = std::getenv("DATA_PATH");
    std::string models_path = std::getenv("MODELS_PATH");

    std::string training_images_path = data_path_str + "train-images.idx3-ubyte";
    std::string training_labels_path = data_path_str + "train-labels.idx1-ubyte";
    read_mnist_images(training_images_path.c_str(), &data_training);
    read_mnist_labels(training_labels_path.c_str(), &data_training.labels, &data_training.nImages);
    printf("Number of training images: %ld\n", data_training.nImages);

    std::string test_images_path = data_path_str + "t10k-images.idx3-ubyte";
    std::string test_labels_path = data_path_str + "t10k-labels.idx1-ubyte";
    read_mnist_images(test_images_path.c_str(), &data_test);
    read_mnist_labels(test_labels_path.c_str(), &data_test.labels, &data_test.nImages);
    printf("Number of test images: %ld\n", data_test.nImages);

    srand(42); // Set a fixed seed for reproducibility

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
    printf("Batch size: %d\n", SIZE_MINI_BATCH);

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
    Activations activations;
    calculate_size(&activations, &model, &data_test);
    initialise_activations(&activations, &model, &data_test);
    model_forward(&model, &activations, &data_test);
    float initial_loss = get_loss(&model, &activations, &data_test);
    printf("Test loss before training: %f\n", initial_loss);
    printf("Test accuracy before training: %f\n", get_accuracy(&model, &activations, &data_test));


    // exit(0);

    // initialise activations and gradients for training
    // with mini batches

    Gradients gradients;
    data_mini_batch.nImages = SIZE_MINI_BATCH;
    data_mini_batch.rows = data_training.rows;
    data_mini_batch.cols = data_training.cols;
    allocate_mini_batch_memory(&data_mini_batch);
    initialise_gradients(&gradients, &model, &data_mini_batch);
    printf("\n");
    printf("\n");
    printf("training loop:\n");
    for (size_t step = 0; step < NUMBER_STEPS; step++) {
        initialise_mini_batch(&data_training, &data_mini_batch);
        calculate_size(&activations, &model, &data_mini_batch);
        initialise_activations(&activations, &model, &data_mini_batch);
        model_forward(&model, &activations, &data_mini_batch);
        model_backward(&model, &activations, &data_mini_batch);
        if (step % PRINT_EVERY == 0) {
            printf("\nstep: %zu\n", step);
            printf("Training loss: %f\n", get_loss(&model, &activations, &data_mini_batch));
            printf("Training accuracy: %f\n", get_accuracy(&model, &activations, &data_mini_batch));
            printf("\n\n");
        }
        memset(gradients.grads.data(), 0, gradients.size_grads * sizeof(float));
    }

 

    // save model
    save_model(&model, models_path.c_str());

    // test loss after training
    initialise_activations(&activations, &model, &data_test);
    model_forward(&model, &activations, &data_test);
    float final_loss = get_loss(&model, &activations, &data_test);
    printf("\nTest loss after training: %f\n", final_loss);
    printf("Test accuracy after training: %f%%\n", get_accuracy(&model, &activations, &data_test) * 100);
    printf("Difference in loss: %f\n", initial_loss - final_loss);
 
    
    return 0;
}