# Full ResNet binary-mask propagation

Input center pixels are 1; other pixels are 0. Each convolution uses a positive unit spatial kernel, followed by thresholding at >0. This propagates potential sensitive-input dependence without trained-weight cancellation, BN offsets or ReLU value gating. Pointwise BN/activations preserve the dependency mask. All channels share the same mask, allowing an exact single-spatial-plane representation for this structural propagation.

Architecture follows exported parameter kernel sizes: conv1 7x7 stride 2 pad 3; replacement maxpool_0 convolution 3x3 stride 2 pad 1; two residual blocks in each of layer1..layer4, each with two 3x3 convolutions. The first block of stages 2..4 downsamples with stride 2 and a parallel 1x1 stride-2 projection. Residual outputs use the union of both masks. There are 18 main-path convolutions in this modified model (projection convolutions not counted); this is not standard ResNet naming with its final FC counted as layer 18.

Validated all mask outputs at convolution indices 2..8 against the recorded encrypted rectangles for all eight areas (56 exact mask comparisons). Later layers are structural simulations, not measured homomorphic runs. Global average pooling yields a sensitive 1x1 output for every initial region; structurally, the following dense classifier outputs are sensitive too.

| Initial area | First 100% spatial coverage at convolution |
|---|---:|
| 10x10 | 15 |
| 20x20 | 14 |
| 30x30 | 14 |
| 40x40 | 14 |
| 50x50 | 13 |
| 60x60 | 13 |
| 70x70 | 13 |
| 80x80 | 12 |
