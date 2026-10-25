#pragma once
#include <stdexcept>

struct EncryptedRegion {
    int hs, he, ws, we;
};

// Conservative receptive-field overlap mapping, with inclusive input/output bounds.
inline EncryptedRegion MapEncryptedConvRegion(EncryptedRegion input, int height, int width,
                                              int kh, int kw, int stride, int padding) {
    if (height<=0 || width<=0 || kh<=0 || kw<=0 || stride<=0 || padding<0 ||
        input.hs<0 || input.he<input.hs || input.he>=height ||
        input.ws<0 || input.we<input.ws || input.we>=width ||
        height+2*padding<kh || width+2*padding<kw)
        throw std::invalid_argument("invalid encrypted convolution region");
    EncryptedRegion output{-1,-1,-1,-1};
    for (int h=0;h<(height+2*padding-kh)/stride+1;++h)
        if (h*stride+kh-1>=input.hs+padding && h*stride<=input.he+padding) {
            if(output.hs<0)output.hs=h;
            output.he=h;
        }
    for (int w=0;w<(width+2*padding-kw)/stride+1;++w)
        if (w*stride+kw-1>=input.ws+padding && w*stride<=input.we+padding) {
            if(output.ws<0)output.ws=w;
            output.we=w;
        }
    if(output.hs<0 || output.ws<0)throw std::invalid_argument("encrypted region has no convolution output");
    return output;
}
