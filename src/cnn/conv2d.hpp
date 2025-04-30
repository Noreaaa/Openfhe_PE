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


    class Conv2d_P : public Layer {
        public:
            Conv2d_P(
                PLayerType layer_type,
                std::string layer_name,
                types::double3d& filters,
                std::vector<double>& biases,
                int stride,
                int padding,
                uint32_t batch_size
                // no padding 
            );
            ~Conv2d_P();

            void forward(vector<Ciphertext<DCRTPoly>>& x_cts,
                vector<Ciphertext<DCRTPoly>>& y_cts, double3d& x_pts, double3d& y_pts) override;

            void forward(types::vector2d<Ciphertext<DCRTPoly>>& x_cts, double3d& x_pts,
                types::vector2d<Ciphertext<DCRTPoly>>& y_cts, double3d& y_pts) override;

            void padding(double3d& x_pts, vector<Ciphertext<DCRTPoly>>& x_cts, int padding);

        private:
            types::double3d filters_;
            std::vector<double> biases_;
            int stride_;
            int padding_;
            uint32_t batch_size_;
    };

    class Conv2dBN_P : public Layer {
        public:
            Conv2dBN_P(
                types::double2d& filters,
                std::vector<double>& biases,
                const std::pair<int, int>& stride = {1, 1},
                const int & padding = 0,
                uint32_t batch_size,
                std::vector<double> & gamma,
                std::vector<double> & beta,
                std::vector<double> & mean,
                std::vector<double> & var,
                std::vector<double> & epsilon
            );
            ~Conv2dBN_P();

            void forward(types::vector2d<Ciphertext<DCRTPoly>>& x_cts, double3d& x_pts,
                types::vector2d<Ciphertext<DCRTPoly>>& y_cts, double3d& y_pts) override;

        private:
            types::double3d filters_;
            std::vector<double> biases_;
            std::pair<int, int> stride_;
            int padding_;
            uint32_t batch_size_;
            std::vector<double> gamma_;
            std::vector<double> beta_;
            std::vector<double> mean_;
            std::vector<double> var_;
            std::vector<double> epsilon_;
    };
    


    void GoldenConv2d(double3d& input, double3d& filters, int stride, int padding);

    bool isEncrypted_h(int val, int filter_size, int padding);

    bool isEncrypted(int oh, int ow, int fh, int fw, int padding);