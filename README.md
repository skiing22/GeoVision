# GeoVision
GeoVision: AI-Driven Disaster Mapping and Damage Assessment
# GeoVision: AI-Driven Disaster Mapping and Damage Assessment

GeoVision is an AI-powered framework designed for disaster mapping and damage analysis. Using high-resolution satellite imagery and cutting-edge deep learning models, the project provides accurate and timely insights for disaster management. Initially focused on flood mapping, GeoVision leverages advanced segmentation techniques to identify and quantify flood-affected areas efficiently.

## Features
- Automated disaster mapping using Sentinel-1 satellite imagery.
- Advanced segmentation models: U-Net with MobileNet backbone and DeepLabv3+ with ResNet backbone.
- Comparative analysis of pre- and post-disaster imagery for accurate damage assessment.
- Supports customization for other disaster scenarios (e.g., wildfires, earthquakes).

## Installation

### Prerequisites
Ensure you have Python 3.8+ installed. The project also requires a GPU-enabled environment for training models.

### Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/GeoVision.git
   cd GeoVision
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation
1. Place raw satellite images in the `data/raw/` folder.
2. Run the preprocessing script to prepare data for training:
   ```bash
   python scripts/data_preprocessing.py
   ```

### Model Training
Train the models using the prepared dataset:
   ```bash
   python scripts/model_training.py
   ```

### Model Evaluation
Evaluate the trained models on test data:
   ```bash
   python scripts/model_evaluation.py
   ```

### Prediction
Use the trained models for predictions:
   ```bash
   python scripts/predict.py --input_path=data/raw/new_image.tif --output_path=results/predicted_mask.png
   ```

## Data Description
- **Input Data**: Sentinel-1 GRD imagery, pre- and post-disaster.
- **Processed Data**: Patches of 256x256 pixels for training and validation.

## Model Architectures
### 1. DeepLabv3+
- Backbone: ResNet
- Key Features: Atrous convolutions and ASPP for multi-scale context.

### 2. U-Net
- Backbone: MobileNet
- Key Features: Encoder-decoder architecture with skip connections for precise localization.

## Results
| Metric          | U-Net     | DeepLabv3+ |
|-----------------|-----------|------------|
| Loss            | 0.2006    | 0.1453     |
| Dice Coefficient| 0.5353    | 0.6599     |
| Jaccard Index   | 0.3700    | 0.4968     |

DeepLabv3+ outperformed U-Net in accuracy and boundary precision.

## Future Enhancements
- Integration with Sentinel-2 multispectral data.
- Real-time disaster analysis using cloud platforms.
- Extending support to other disaster scenarios.

## Contribution
Contributions are welcome! Please fork the repository and create a pull request for any feature or improvement.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- Sentinel-1 data from [Copernicus Open Access Hub](https://scihub.copernicus.eu/).
- DeepLabv3+ and U-Net models implemented using [Keras](https://keras.io/).
