#include "globals.hpp"

using std::size_t;
using namespace lbcrypto;

CryptoContext<DCRTPoly> CRYPTOCONTEXT;
std::shared_ptr<lbcrypto::BinFHEContext> CCLWE;
KeyPair<lbcrypto::DCRTPoly> KEYPAIR;
int CURRENT_HEIGHT;
int CURRENT_WIDTH;
int CURRENT_CHANNEL;
int ENCRYPTED_HEIGHT_START;
int ENCRYPTED_HEIGHT_END;
int ENCRYPTED_WIDTH_START;
int ENCRYPTED_WIDTH_END;
uint32_t NUMSLOTS;

std::vector<std::vector<int>> INDEX_MAP;
std::size_t CONSUMED_LEVEL;
std::vector<double> POLY_ACT_COEFFS;
double POLY_ACT_HIGHEST_DEG_COEFF;
double CURRENT_POOL_MUL_COEFF;
bool SHOULD_MUL_ACT_COEFF;
bool SHOULD_MUL_POOL_COEFF;
EActivationType ACTIVATION_TYPE;
double ROUND_THRESHOLD;
std::set<int> USE_ROTATION_STEPS;
int INPUT_C, INPUT_H, INPUT_W, OUTPUT_C, OUTPUT_H, OUTPUT_W,
    INPUT_UNITS, OUTPUT_UNITS;

std::vector<int> INPUT_HW_IDX;
std::vector<int> OUTPUT_HW_IDX;
std::vector<std::vector<int>> KERNEL_HW_ROTATION_STEP;
std::vector<int> FLATTEN_ROTATION_STEP;
std::vector<int> OUTPUT_MASKED_MAP;

int FILTER_N, FILTER_H, FILTER_W, IMAGE_C, IMAGE_H, IMAGE_W;
int STRIDE, PADDING;
std::pair<int, int> TOP_LEFT;
std::pair<int, int> TOP_RIGHT;
std::pair<int, int> BOTTOM_LEFT;
std::pair<int, int> BOTTOM_RIGHT;
int TOP, RIGHT, BOTTOM, LEFT;
size_t SLOT_COUNT;
std::vector<int> FILTER_SIZES;

int CURRENT_LEVEL;
int MAX_LEVEL;
int CONV_COUNT;
bool WITHIN;

