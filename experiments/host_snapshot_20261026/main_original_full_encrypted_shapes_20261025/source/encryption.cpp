#include "encryption.hpp"

using namespace lbcrypto;
using types::double3d;
using types::double2d;
using std::vector;


void Partial_Encrypt_Sparse(types::double3d &image_3d, uint32_t numSlots, 
    uint32_t depth,CryptoContext<DCRTPoly> cryptocontext, 
    KeyPair<lbcrypto::DCRTPoly> Keypair, int channel, int max_channel, int height_start, int height_end,
    int width_start, int width_end, std::vector<Ciphertext<DCRTPoly>> &x_ctxt){
        //int image_height = image_3d[0].size();
        //int image_width = image_3d[0][0].size();
        //int image_size = image_height * image_width;

        // use the smallest ciphertext size 

    
        for(int i = height_start; i <= height_end; i++){
            for(int j = width_start; j <= width_end; j++){
                std::vector<double> x_vec(numSlots, 0);
                for(int c = 0; c < channel; c++){
                    x_vec[c] = image_3d[c][i][j];
                }
                Plaintext x_ptxt = cryptocontext->MakeCKKSPackedPlaintext(x_vec, 1, depth - 1, nullptr, max_channel);
                std::cout << "Plaintext: " << x_ptxt << std::endl;
                x_ctxt.push_back(cryptocontext->Encrypt(Keypair.secretKey, x_ptxt));
            }
        }

        return;

    }


void Partial_Encrypt(types::double3d &image_3d, uint32_t numSlots, 
        uint32_t depth, CryptoContext<DCRTPoly> cryptocontext, 
        KeyPair<lbcrypto::DCRTPoly> Keypair, int channel, 
        int enc_height, int enc_width,
        int height_start, int height_end, 
        int width_start, int width_end,
        Ciphertext<DCRTPoly> &x_ctxt){

    // create a map indicate the indexing relationship of feature map and ciphertext
    // index_map[i][j] = index of feature map [i][j] on ciphertext
    
    INDEX_MAP.resize(image_3d[0].size());
    for (int i = 0; i < static_cast<int>(INDEX_MAP.size()); i++){
        INDEX_MAP[i].resize(image_3d[0][0].size());
    }



    vector<double> x_vec;
    for (int c = 0; c < channel; c++){
        for (int h = 0; h < enc_height; h++){
            for (int w = 0; w < enc_width; w++){
                if (isInRange(h, height_start, height_end) && 
                    isInRange(w, width_start, width_end)){
                    x_vec.push_back(image_3d[c][h][w]);
                    std::cout << image_3d[c][h][w] << " ";
                }
                else{
                    x_vec.push_back(0);
                    std::cout << "0 ";
                }
            }
        }
    }
    std::cout << std::endl;
    std::cout << "finished generated x_vec" << std::endl;

    Plaintext x_ptxt = cryptocontext->MakeCKKSPackedPlaintext(x_vec, 1,
         depth - 1, nullptr, numSlots);
    
    std::cout << "Plaintext: " << x_ptxt << std::endl;

    x_ctxt = cryptocontext->Encrypt(Keypair.secretKey, x_ptxt);
}

/**
 * @brief Encrypt the input image
 * MCSC: multiple channel single row
 */
void Encrypt_MCSR(types::double3d image3d, uint32_t numSlots, int depth,
int max_channel,  CryptoContext<DCRTPoly> cryptocontext, 
KeyPair<lbcrypto::DCRTPoly> Keypair, std::vector<Ciphertext<DCRTPoly>> &x_ctxt){
    // ensure we can pack all channel in one ciphertext
    int channel = image3d.size();
    int height = image3d[0].size();
    int width = image3d[0][0].size();
    if (max_channel * width > static_cast<int>(numSlots)){
        std::cerr << "Cannot pack all channels in one ciphertext, please use a larger parameter set" << std::endl;
        return;
    }

    for (int i = 0; i < height; i++){
        std::vector<double> x_vec;
        for (int c = 0; c < channel; c++){
            for (int j = 0; j < width; j++){
                x_vec.push_back(image3d[c][i][j]);
            }
        }
        Plaintext x_ptxt = cryptocontext->MakeCKKSPackedPlaintext(x_vec, 1, depth, nullptr, numSlots);
        std::cout << "Plaintext: " << x_ptxt << std::endl;
        x_ctxt.push_back(cryptocontext->Encrypt(Keypair.secretKey, x_ptxt));

    }
}


/**
 * @brief Encrypt the input image
 * MCSC: multiple channel single row partially encrypted
 */
void Encrypt_MCSR_P(types::double3d& image3d, uint32_t numSlots, 
      CryptoContext<DCRTPoly> cryptocontext, int enc_height_start, int enc_height_end, int enc_width_start, int enc_width_end,
    KeyPair<lbcrypto::DCRTPoly> Keypair, types::vector2d<Ciphertext<DCRTPoly>> &x_ctxt){
        // ensure we can pack all channel in one ciphertext
        int channel = image3d.size();
        int width = image3d[0][0].size();
        x_ctxt.clear();


        x_ctxt.resize(enc_height_end - enc_height_start + 1);

        for (int i = enc_height_start; i <= enc_height_end; i++){
            std::vector<double> x_vec;
            for (int c = 0; c < channel; c++){
                for (int j = 0; j < width; j++){//224+2x3 = 230
                    if(isInRange(j, enc_width_start, enc_width_end)){
                        x_vec.push_back(image3d[c][i][j]);
                        image3d[c][i][j] = 0;
                    }
                    else{
                        x_vec.push_back(0);
                    }
                }
            }
            // 512
            int channel_per_cts = numSlots / width; 
            int values_per_cts = channel_per_cts * width;
            for (int j = 0; j < static_cast<int>(x_vec.size()); j += values_per_cts){
                int end = std::min(j + values_per_cts, static_cast<int>(x_vec.size()));
                std::vector<double> one_batch(x_vec.begin() + j, x_vec.begin() + end);
                Plaintext x_ptxt = cryptocontext->MakeCKKSPackedPlaintext(one_batch);
                std::cout << "Plaintext[" << i - enc_height_start << "]: " << x_ptxt << std::endl;
                x_ctxt[i - enc_height_start].push_back(cryptocontext->Encrypt(Keypair.secretKey, x_ptxt));
            }
        }
    }

/**
 * @brief Encrypt the input image
 * MCSC: multiple channel single row partially encrypted 
 * only reserve some slots for edge region
 */
void Encrypt_MCSR_P_COMPACT(types::double3d& image3d, uint32_t numSlots, 
      CryptoContext<DCRTPoly> cryptocontext, int enc_height_start, int enc_height_end, int enc_width_start, int enc_width_end,
    KeyPair<lbcrypto::DCRTPoly> Keypair, types::vector2d<Ciphertext<DCRTPoly>> &x_ctxt){
        // ensure we can pack all channel in one ciphertext
        int channel = image3d.size();
        int width = image3d[0][0].size();
        int encrypted_width = enc_width_end - enc_width_start + 1;
        int encrypted_height = enc_height_end - enc_height_start + 1;
        x_ctxt.clear();
        x_ctxt.resize(encrypted_height);

        for (int i = enc_height_start; i <= enc_height_end; i++){
            std::vector<double> x_vec;
            for (int c = 0; c < channel; c++){
                for (int j = 0; j < width; j++){
                    if(isInRange(j, enc_width_start, enc_width_end)){
                        x_vec.push_back(image3d[c][i][j]);
                        image3d[c][i][j] = 0;
                    }
                }
            }
            int channel_per_cts = numSlots / encrypted_width;
            int values_per_cts = channel_per_cts * encrypted_width;
            for (int j = 0; j < static_cast<int>(x_vec.size()); j += values_per_cts){
                int end = std::min(j + values_per_cts, static_cast<int>(x_vec.size()));
                std::vector<double> one_batch(x_vec.begin() + j, x_vec.begin() + end);
                Plaintext x_ptxt = cryptocontext->MakeCKKSPackedPlaintext(one_batch);
                std::cout << "Plaintext[" << i - enc_height_start << "]: " << x_ptxt << std::endl;
                x_ctxt[i - enc_height_start].push_back(cryptocontext->Encrypt(Keypair.secretKey, x_ptxt));
            }
        }
    }

std::vector<Ciphertext<DCRTPoly>> Encrypt_baseline(double3d& image3d, uint32_t numSlots, 
    int image_size, int channels, CryptoContext<DCRTPoly> cryptocontext, KeyPair<DCRTPoly> Keypair){
    std::vector<Ciphertext<DCRTPoly>> x_ctxt;
    for (int c = 0; c < channels; c++) {
        std::vector<double> x_vec;
        for (int i = 0; i < image_size; i++) {
            for (int j = 0; j < image_size; j++) {
                x_vec.push_back(image3d[c][i][j]);
            }
        }
        Plaintext x_ptxt = cryptocontext->MakeCKKSPackedPlaintext(x_vec);
        std::cout << "Plaintext: " << x_ptxt << std::endl;
        x_ctxt.push_back(cryptocontext->Encrypt(Keypair.secretKey, x_ptxt));
    }
    return x_ctxt;
}

void Gen_random_cts2d(uint32_t numSlots, uint32_t valid_size, int dim1, int dim2, 
CryptoContext<DCRTPoly> cryptocontext, KeyPair<lbcrypto::DCRTPoly> Keypair, 
types::vector2d<Ciphertext<DCRTPoly>> &x_ctxt){
    std::random_device rd;
    std::mt19937 gen(rd());  // Mersenne Twister
    std::uniform_real_distribution<> dis(0.0, 1.0);
    x_ctxt.clear();
    x_ctxt.resize(dim1);
    for (int i = 0; i < dim1; i++){
        for (int j = 0; j < dim2; j++){
            std::vector<double> random_data(numSlots, 0);
            for (uint32_t k = 0; k < valid_size; k++){
                random_data[k] = dis(gen); // Generate random
            }
            Plaintext x_ptxt = cryptocontext->MakeCKKSPackedPlaintext(random_data);
            std::cout << "Plaintext: " << x_ptxt << std::endl;
            x_ctxt[i].push_back(cryptocontext->Encrypt(Keypair.secretKey, x_ptxt));
        }
    }
}

    