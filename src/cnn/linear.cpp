#include "linear.hpp"

Linear_P::Linear_P(
    PLayerType layer_type,
    std::string layer_name,
    types::vector2d<double>& weights,
    vector<double>& bias,
    int batch_size)
    : Layer(PLayerType::LINEAR, layer_name),
    weights_(weights),
    bias_(bias),
    batch_size_(batch_size)
    {
        CONSUMED_LEVEL++;
    }
Linear_P::~Linear_P() {}

void Linear_P::forward(types::vector2d<Ciphertext<DCRTPoly>>& x_cts, double3d& x_pts,
    vector<double>& y_pts) {
    std::cout << layer_name_ << " forward" << std::endl;

    
    // decrypt the input cihpertext
    // and add to the x_pts
    y_pts.clear();
    int encrypted_rows = x_cts.size();
    int encrypted_batches = x_cts[0].size();
    int input_n = x_pts.size();
    int input_h = x_pts[0].size();
    int input_w = x_pts[0][0].size();
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2)
    #endif
    for (int i = 0; i < encrypted_rows; i++){
        for (int j = 0; j < encrypted_batches; j++){
            Plaintext ptxt;
            CRYPTOCONTEXT->Decrypt(KEYPAIR.secretKey, x_cts[i][j], &ptxt);
            vector<double> row_vec = ptxt->GetRealPackedValue();

            int channel_idx_start = j * batch_size_ / input_w;
            //int channel_idx_end = std::min((j + 1) * batch_size_ / input_w, static_cast<size_t>(input_n));
            int col_idx = i + ENCRYPTED_HEIGHT_START;

            for (size_t k = 0; k < row_vec.size(); k++){
                if (channel_idx_start + static_cast<int>(k) / input_w >= input_n){
                    break;
                }
                int row_idx = k % input_w;
                int channel_idx = std::min(channel_idx_start + k / input_w, static_cast<size_t>(input_n));
                x_pts[channel_idx][col_idx][row_idx] = row_vec[k];
            }
        }
    }
    vector<double> flattened_input;

    for (int i = 0; i < input_n; i++){
        for (int j = 0; j < input_h; j++){
            for (int k = 0; k < input_w; k++){
                flattened_input.push_back(x_pts[i][j][k]);
            }
        }
    }
    y_pts.resize(weights_.size());

    for (size_t i = 0; i < weights_.size(); i++){
        double sum = 0;
        // y_cts[i] = CRYPTOCONTEXT->EvalAdd(y_cts[i], CRYPTOCONTEXT->MakeCKKSPackedPlaintext(bias_[i]))
        for (size_t j = 0; j < weights_[i].size(); j++){
            sum += weights_[i][j] * flattened_input[j];
        }
        y_pts[i] = sum + bias_[i];
    }

}

void Linear_P::forward(vector<double>& input, vector<double>& output){
    std::cout << layer_name_ << " forward" << std::endl;
    output.resize(weights_.size());
    for (size_t i = 0; i < weights_.size(); i++){
        double sum = 0;
        for (size_t j = 0; j < weights_[i].size(); j++){
            sum += weights_[i][j] * input[j];
        }
        output[i] = sum + bias_[i];
    }
}


void GoldenLinear_3d_input(double3d& input, types::double2d& weights, vector<double>& bias, vector<double>& output){
    int input_n = input.size();
    int input_h = input[0].size();
    int input_w = input[0][0].size();

    vector<double> flattened_input;

    for (int i = 0; i < input_n; i++){
        for (int j = 0; j < input_h; j++){
            for (int k = 0; k < input_w; k++){
                flattened_input.push_back(input[i][j][k]);
            }
        }
    }
    output.resize(weights.size());

    for (size_t i = 0; i < weights.size(); i++){
        double sum = 0;
        for (size_t j = 0; j < weights[i].size(); j++){
            sum += weights[i][j] * flattened_input[j];
        }
        output[i] = sum + bias[i];
    }

}

void GoldenLinear(vector<double>& input, types::double2d& weights, vector<double>& bias){
    vector<double> output;
    output.resize(weights.size());
    for (size_t i = 0; i < weights.size(); i++){
        double sum = 0;
        for (size_t j = 0; j < weights[i].size(); j++){
            sum += weights[i][j] * input[j];
        }
        output[i] = sum + bias[i];
    }

    input.clear();

    for(size_t i = 0; i < output.size(); i++){
        input.push_back(output[i]);
    }
}