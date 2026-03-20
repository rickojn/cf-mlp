#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <immintrin.h>
#include <stdint.h>
#include <string>
#include <vector>

#include "../custard-flow/include/CustardFlow.h"

#define SIZE_CLASSES 10
#define SIZE_OUTPUT 10
#define SIZE_HIDDEN 8

struct InputData {
    std::vector<unsigned char> images;
    std::vector<long> labels;
    int nImages, rows, cols;
};

typedef struct Layer {
    float *weights, *biases, *activations_input, *activations_output,
           *gradients_input, *gradients_output, *gradients_weights, *gradients_biases;
    size_t size_inputs, size_neurons;
    void (*activation_forward)(float *activations, size_t num_features, size_t size_batch);
    void (*activation_backward)(const float *inputs, float *gradients, const long *labels, size_t num_features, size_t size_batch);
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

// Utility functions
void file_read(void *ptr, size_t size, size_t count, FILE *file) {
    size_t result = fread(ptr, size, count, file);
    if (result != count) {
        fprintf(stderr, "Error reading file\n");
        exit(1);
    }
}

void read_mnist_images(const char *filename, InputData *input_data) {
    printf("Reading MNIST images from %s...\n", filename);
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
    printf("Image dimensions: %d x %d\n", input_data->rows, input_data->cols);

    input_data->images.resize(input_data->nImages * input_data->rows * input_data->cols);
    file_read(input_data->images.data(), sizeof(unsigned char), input_data->nImages * input_data->rows * input_data->cols, file);
    fclose(file);
}

void read_mnist_labels(const char *filename, std::vector<long> *labels, int *nLabels) {
    printf("Reading MNIST labels from %s...\n", filename);
    FILE *file = fopen(filename, "rb");
    if (!file) exit(1);

    int mnist_magic_number;
    file_read(&mnist_magic_number, sizeof(int), 1, file);
    if (__builtin_bswap32(mnist_magic_number) != 2049) {
        printf("Invalid MNIST label file magic number: %d\n", __builtin_bswap32(mnist_magic_number));
        fclose(file);
        exit(1);
    }
    file_read(nLabels, sizeof(int), 1, file);
    *nLabels = __builtin_bswap32(*nLabels);

    std::vector<unsigned char> temp_labels(*nLabels);
    labels->resize(*nLabels);
    file_read(temp_labels.data(), sizeof(unsigned char), *nLabels, file);
    for (unsigned char label : temp_labels) {
        labels->push_back((long)label);
    }
    fclose(file);
}

void load_model_from_file(Model *model, const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Error opening file %s\n", filename);
        return;
    }
    file_read(model->layers[0].weights, sizeof(float), model->size_parameters, file);
    fclose(file);
}

int load_model(Model *model, const char *dirname) {
    DIR *dir;
    struct dirent *entry;
    struct stat file_info;
    time_t latest_timestamp = 0;
    char full_filename[260];
    char latest_fullname[260];

    dir = opendir(dirname);
    if (dir == NULL) {
        printf("Error opening directory %s\n", dirname);
        return 0;
    }

    while ((entry = readdir(dir)) != NULL) {
        snprintf(full_filename, sizeof(full_filename), "%s/%s", dirname, entry->d_name);
        if (stat(full_filename, &file_info) == 0 && S_ISREG(file_info.st_mode)) {
            time_t file_timestamp = file_info.st_mtime;
            if (file_timestamp > latest_timestamp) {
                latest_timestamp = file_timestamp;
                strcpy(latest_fullname, full_filename);
            }
        }
    }

    closedir(dir);
    if (latest_timestamp == 0) {
        printf("No model files found in directory %s\n", dirname);
        return 0;
    } else {
        printf("Loading model from file %s\n", latest_fullname);
        load_model_from_file(model, latest_fullname);
        return 1;
    }
}

void model_forward(Model *model, Activations *activations, InputData *data) {
    for (size_t idx_layer = 0; idx_layer < model->size_layers; idx_layer++) {
        Layer *layer = &model->layers[idx_layer];
        simd_matmul(layer->activations_input, layer->weights, layer->activations_output,
                    data->nImages, layer->size_neurons, layer->size_inputs);
        layer->activation_forward(layer->activations_output, layer->size_neurons, data->nImages);
    }
}

void calculate_size(Activations *activations, Model *model, InputData *data) {
    activations->size_activations = 0;
    for (size_t i = 0; i < model->size_layers; i++) {
        activations->size_activations += model->layers[i].size_neurons;
    }
    activations->size_activations += data->rows * data->cols;
    activations->size_activations *= data->nImages;
}

void initialise_activations(Activations *activations, Model *model, InputData *input_data) {
    activations->activations.resize(activations->size_activations);

    for (size_t idx_pixel = 0; idx_pixel < (size_t)input_data->nImages * input_data->rows * input_data->cols; idx_pixel++) {
        activations->activations[idx_pixel] = (float)input_data->images[idx_pixel] / 255.0f;
    }

    float *inputs = activations->activations.data();
    float *outputs = activations->activations.data() + input_data->rows * input_data->cols;

    for (size_t idx_layer = 0; idx_layer < model->size_layers; idx_layer++) {
        Layer *layer = &model->layers[idx_layer];
        layer->activations_input = idx_layer == 0 ? inputs : model->layers[idx_layer - 1].activations_output;
        layer->activations_output = idx_layer == 0 ? outputs : outputs + model->layers[idx_layer - 1].size_neurons * input_data->nImages;
    }
}

int arg_max(float *probs, size_t size) {
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

void predict(Model *model, Activations *activations, InputData *data) {
    float *output_probs = model->layers[model->size_layers - 1].activations_output;

    for (size_t idx_image = 0; idx_image < (size_t)data->nImages; idx_image++) {
        size_t offset = idx_image * SIZE_CLASSES;
        int predicted_digit = arg_max(output_probs + offset, SIZE_CLASSES);
        float confidence = output_probs[offset + predicted_digit];

        printf("Image %zu: Predicted digit = %d, Confidence = %.4f",
               idx_image, predicted_digit, confidence);
        if (!data->labels.empty()) {
            unsigned char true_label = data->labels[idx_image];
            printf(", True label = %d", true_label);
            if (predicted_digit == true_label) {
                printf(" (Correct)");
            } else {
                printf(" (Incorrect)");
            }
        }
        printf("\n");
    }
}

void add_layer(Model *model, size_t size_inputs, size_t size_neurons,
               void (*activation_forward)(float *activations, size_t num_classes, size_t size_batch),
               void (*activation_backward)(const float *inputs, float *gradients, const long *labels, size_t num_features, size_t size_batch),
               float (*generate_number)(size_t, size_t)) {
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

void allocate_parameters_memory(Model *model) {
    model->parameters.resize(model->size_parameters);
    float *parameters = model->parameters.data();
    size_t offset = 0;

    for (size_t idx_layer = 0; idx_layer < model->size_layers; idx_layer++) {
        Layer *layer = &model->layers[idx_layer];
        layer->weights = parameters + offset;
        offset += layer->size_inputs * layer->size_neurons;
        layer->biases = parameters + offset;
        offset += layer->size_neurons;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <path_to_mnist_images_file> [path_to_mnist_labels_file] [path_to_models_dir]\n", argv[0]);
        printf("\nExample: %s t10k-images.idx3-ubyte t10k-labels.idx1-ubyte ./models\n", argv[0]);
        return 1;
    }

    const char *images_path = argv[1];
    const char *labels_path = (argc > 2) ? argv[2] : NULL;
    const char *models_path = (argc > 3) ? argv[3] : "./models";

    // Read MNIST images
    InputData data;
    read_mnist_images(images_path, &data);
    printf("Loaded %d images\n", data.nImages);

    // Read labels if provided (for reference)
    if (labels_path) {
        read_mnist_labels(labels_path, &data.labels, &data.nImages);
        printf("Loaded %zu labels\n", data.labels.size());
    }

    // Create and initialize model structure
    Model model;

    // Add layers to match the training model
    add_layer(&model, data.cols * data.rows, SIZE_HIDDEN, relu_forward, relu_backward, NULL);
    add_layer(&model, SIZE_HIDDEN, SIZE_OUTPUT, softmax_forward, loss_softmax_backward, NULL);

    printf("Model created with %zu layers\n", model.size_layers);
    for (size_t i = 0; i < model.size_layers; i++) {
        Layer *layer = &model.layers[i];
        printf("Layer %zu: %zu inputs, %zu neurons\n", i, layer->size_inputs, layer->size_neurons);
    }
    printf("Number of parameters: %zu\n", model.size_parameters);

    // Allocate memory for model parameters
    allocate_parameters_memory(&model);

    // Load model from latest file
    if (!load_model(&model, models_path)) {
        fprintf(stderr, "Failed to load model from %s\n", models_path);
        return 1;
    }

    // Prepare activations
    Activations activations;
    calculate_size(&activations, &model, &data);
    initialise_activations(&activations, &model, &data);

    // Run inference
    printf("\nRunning inference on %d images...\n\n", data.nImages);
    model_forward(&model, &activations, &data);

    // Output predictions
    predict(&model, &activations, &data);

    return 0;
}
