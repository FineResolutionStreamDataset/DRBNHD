# Spatially Multi-Resolution Benchmarks for Stream Temperature Prediction and Downscaling

## 📊 Data Sources

**Geomorphological Features** (segment slope, elevation, mean width)
- [GIS Features of the Geospatial Fabric for National Hydrologic Modeling](https://www.sciencebase.gov/catalog/item/5e29b87fe4b0a79317cf7df5)
- [Updated dataset v1.1](https://catalog.data.gov/dataset/gis-features-of-the-geospatial-fabric-for-the-national-hydrologic-model-version-1-1)

**Meteorological Features** (solar radiation, precipitation, evapotranspiration)
- [Development of gridded surface meteorological data for ecological applications and modelling](https://rmets.onlinelibrary.wiley.com/doi/10.1002/joc.3413)
- [PRMS-IV, the Precipitation-Runoff Modeling System, Version 4](https://pubs.usgs.gov/publication/tm6B7)

## 💾 Dataset Downloads

The datasets on GitHub are incomplete. Please download the complete versions from Google Drive:

- **DRB_NHD_on_NHM_all_2024**: [Download from Google Drive](https://drive.google.com/file/d/1KS507tSReqiQ3RSxvNUCNOiH2oBduxa1/view?usp=drive_link)
- **DRB_NHM_20230928**: [Download from Google Drive](https://drive.google.com/file/d/18EBbCC0x7jJfkxuLGIHTOHNjr7WoSfrj/view?usp=drive_link)

## 🚀 Getting Started

### Training SpatioTemporal Models
- **Low-resolution DRB dataset**: Run `coarse.py`
- **High-resolution DRB dataset**: Run `base.py`

### Basic Downscaling Methods
Run the following scripts for different downscaling approaches:
- `remap.py`
- `direct_upscaling.py`
- `progress_upscaling.py`

## 🔧 Model Architecture

All spatiotemporal models are located in the `multiscale.MODEL` module with references. 

**Note**: The models maintain their original GitHub implementations, with minor modifications to the first and final layers of the forward function to accommodate our specific use case.
