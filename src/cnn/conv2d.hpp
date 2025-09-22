#pragma once 

#include "layer.hpp"


    class Conv2d : public Layer {
        public:
            Conv2d(
                PLayerType layer_type,
                std::string layer_name,
                types::double3d& filters,
                std::vector<double>& biases,
                int stride,
                int padding,
                uint32_t batch_size
                // no padding 
            );
            ~Conv2d();

            void forward(vector<Ciphertext<DCRTPoly>>& x_cts, vector<Ciphertext<DCRTPoly>>& y_cts) override;

            void padding(double3d& x_pts, vector<Ciphertext<DCRTPoly>>& x_cts, int padding);

        private:
            types::double3d filters_;
            std::vector<double> biases_;
            int stride_;
            int padding_;
            uint32_t batch_size_;
    };

    class Conv2d_C : public Layer {
        public:
            Conv2d_C(
                PLayerType layer_type,
                std::string layer_name,
                types::double4d& filters,
                int stride,
                int input_height,
                int input_width,
                uint32_t batch_size
            );
            ~Conv2d_C();

            void forward(std::vector<Ciphertext<DCRTPoly>>& x_cts, 
                std::vector<Ciphertext<DCRTPoly>>& y_cts);
        
        private:
            types::double4d filters_;
            int stride_;
            int input_height_;
            int input_width_;
            uint32_t batch_size_;
    };



    class Conv2d_P : public Layer {
        public:
            Conv2d_P(
                PLayerType layer_type,
                std::string layer_name,
                types::double3d& filters,
                std::vector<double>& biases,
                int stride,
                int padding,
                uint32_t batch_size,
                PLayerType next_pool_layer_type
                // no padding 
            );
            ~Conv2d_P();

            void forward(types::vector2d<Ciphertext<DCRTPoly>>& x_cts, double3d& x_pts,
                types::vector2d<Ciphertext<DCRTPoly>>& y_cts, double3d& y_pts) override;

            void padding(double3d& x_pts, vector<Ciphertext<DCRTPoly>>& x_cts, int padding);

        private:
            types::double3d filters_;
            std::vector<double> biases_;
            int stride_;
            int padding_;
            uint32_t batch_size_;
            PLayerType next_pool_layer_type_;
    };

    class Conv2dBN_P : public Layer {
        public:
            Conv2dBN_P(
                PLayerType layer_type,
                std::string layer_name,
                types::double4d& filters,
                int stride,
                int padding,
                uint32_t batch_size,
                std::vector<double> & gamma,
                std::vector<double> & beta,
                std::vector<double> & mean,
                std::vector<double> & var,
                double epsilon,
                std::vector<double> & bias,
                PLayerType next_pool_layer_type
            );
            ~Conv2dBN_P();

            Plaintext GenPlainFilter(int out_channel, int height, int output_width_idx, int input_w, int cts_idx, int channels_per_cts);

            void forward(types::vector2d<Ciphertext<DCRTPoly>>& x_cts, double3d& x_pts,
                types::vector2d<Ciphertext<DCRTPoly>>& y_cts, double3d& y_pts) override;

        private:
            types::double4d filters_;
            int stride_;
            int padding_;
            uint32_t batch_size_;
            std::vector<double> gamma_;
            std::vector<double> beta_;
            std::vector<double> mean_;
            std::vector<double> var_;
            double epsilon_;
            std::vector<double> bias_;
            PLayerType next_pool_layer_type_;
    };
    


    double3d GoldenConv2d(double3d input, types::double4d filters, vector<double> bias, int stride, int padding);

    void GoldenBN(double3d& input, vector<double>& gamma, vector<double> beta, vector<double> running_mean, vector<double> running_var,
        double epsilon);
    
    bool isEncrypted_h(int val, int filter_size, int padding, int start, int end);

    bool isEncrypted(int oh, int ow, int fh, int fw, int padding);

    std::vector<std::pair<int, int>> generate_anchor(int height, int width, int stride, int kernel_size);

    void update_index_maps(int height, int width, int stride, int kernel_size, bool initial);

    