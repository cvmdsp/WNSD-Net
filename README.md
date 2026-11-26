# WNSD-Net: A Wavelet-Domain Noise Suppression and Discrimination Network for SAR Ship Detection

### 1. SNS: Speckle Noise Suppression Module
Performs Gaussian smoothing and feature aggregation on the **high-frequency wavelet sub-bands**, suppressing speckle noise while preserving and enhancing ship edges and contours:
1. A 2D discrete wavelet transform decomposes the input features into high-frequency sub-bands (LH / HL / HH) and a low-frequency sub-band (LL);
2. **LL passes through an identity connection** to preserve the main structural information;
3. The three high-frequency sub-bands are added element-wise, then a 3×3 convolution extracts edge structural features, followed by a **Gaussian convolution block** for smoothing and denoising;
4. The result is fed into the GFA module for channel-spatial aggregation.

### 2. GFA: Gaussian Feature Aggregation Module
The key design balancing "denoising" and "edge preservation". All high-frequency sub-bands **share the same set of GFA parameters** (reducing parameter count):
- Original high-frequency and Gaussian-smoothed features are concatenated along the channel dimension; global average pooling followed by 1D/2D convolutions extracts local and global channel information;
- A channel dependency matrix is constructed via matrix multiplication; row-wise aggregation produces a channel weight vector, fused with learnable coefficients α, β and activated by sigmoid;
- The weights are multiplied back into the original and smoothed features, followed by element-wise multiplication and a **residual connection** — the multiplicative operation suppresses noise responses while the residual preserves contour details (ablation: removing the residual connection drops AP_L from 63.3% to 28.2%).

### 3. SDFE: Scattering Discrimination Feature Extraction Module
Extracts discriminative scattering features from the **low-frequency wavelet sub-band**, enhancing the discrimination between ship targets and background noise:
1. **Directional convolution pair**: a pair of directional convolutions whose output channels are each half of the input, enhancing horizontal/vertical scattering responses (side-lobes are distributed along specific directions) while suppressing direction-independent noise components;
2. **Mamba module**: a structured state-space model (SSM) that captures long-range spatial dependencies of scattering structures (layer normalization + 2D selective scan + residual);
3. **SFI (Scattering Feature Interaction) module**: features from the two directions are concatenated and convolved, then split into G groups; adaptive average pooling is applied along the width and height respectively, interaction is performed via 1×1 convolutions, and horizontal/vertical weights are produced to re-weight the fused features in a direction-aware manner, achieving scattering information fusion and enhancement.

## ⚙️ Requirements
```bash
pip install torch torchvision ultralytics timm PyWavelets
```
## 🚀 Training & Evaluation
### Experimental setup (consistent with the paper)

| Item | Configuration |
| --- | --- |
| GPU | NVIDIA Quadro RTX 8000 |
| Input size | 640 × 640 |
| Batch size | 16 |
| Optimizer | SGD (lr=0.01, momentum=0.937, weight decay=0.0005) |
| Training epochs | 300 |
| Wavelet basis | **Haar** (ablation shows it outperforms Db2/Db3/Db4 with minimal over-smoothing and best edge preservation) |

### Datasets

- **SSDD** (SAR Ship Detection Dataset)
- **HRSID** (High-Resolution SAR Images Dataset)

Both use the officially recommended training/testing splits.

### Usage

```bash
# Training
yolo detect train data=ssdd.yaml model=wnsd.yaml epochs=300 imgsz=640 batch=16

# Evaluation
yolo detect val model=wnsd.pt data=ssdd.yaml


