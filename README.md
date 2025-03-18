# Deforestation Analyser

A powerful tool for analyzing deforestation and forest fires using satellite imagery. This application uses advanced computer vision techniques to detect and measure areas affected by deforestation and forest fires.

## 🔴 Live Monitoring Dashboard

Access our real-time deforestation monitoring dashboard:
[Live Dashboard](https://sensational-macaron-123f61.netlify.app/)

## Features

- Upload and analyze before/after satellite images
- Detect deforested areas and measure their size
- Identify regions affected by forest fires
- Generate detailed analysis reports with metrics
- Interactive visualization of affected areas
- Support for various image formats and resolutions
- Real-time monitoring through live dashboard

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/DeforestationAnalyser.git
cd DeforestationAnalyser
```

2. Create and activate a conda environment:
```bash
conda env create -f environment.yml
conda activate deforest
```

3. Install the package:
```bash
pip install -e .
```

## Usage

1. Run the application:
```bash
python run.py
```

2. Open your web browser and navigate to `http://localhost:8501`

3. Upload your satellite images:
   - Select a "before" image
   - Select an "after" image
   - Choose analysis mode (Deforestation or Forest Fire)
   - Adjust settings if needed
   - Run the analysis

## Analysis Modes

### Deforestation Analysis
- Detects areas where vegetation has been removed
- Measures total deforested area in m² and hectares
- Identifies individual deforested regions
- Calculates percentage of land affected

### Forest Fire Analysis
- Identifies areas affected by forest fires
- Measures the extent of fire damage
- Highlights active fire zones
- Provides burn severity assessment

## Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster processing)
- Satellite imagery in common formats (JPEG, PNG, TIFF)

## Project Structure

```
DeforestationAnalyser/
├── deforestation_ui.py    # Streamlit user interface
├── deforestation.py       # Core detection algorithms
├── run.py                # Application runner
├── setup.py              # Package configuration
├── requirements.txt      # Python dependencies
├── environment.yml       # Conda environment
├── ALGORITHM.md          # Algorithm documentation
└── README.md            # This file
```

## Algorithm

For detailed information about the algorithms used in this project, please see [ALGORITHM.md](ALGORITHM.md).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with PyTorch and Streamlit
- Uses advanced computer vision techniques for analysis
- Inspired by the need for better deforestation monitoring tools 