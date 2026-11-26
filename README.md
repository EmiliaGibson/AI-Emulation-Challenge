# AI Emulation Challenge: Global-Scale Land Ecosystem Forecasting

A graph-based deep learning approach for global-scale forest and land ecosystem prediction, based on Google's WeatherNext2 Functional Generative Networks (FGNs) framework.

## Overview

The project investigates the use of graph-based neural networks and spherical spatial representations to emulate complex land ecosystem dynamics and predict future forest and ecosystem states across geographically distributed sites.

Inspired by the approach taken by Google's WeatherNext 2, the framework explores how functional generative modelling and spatially structured neural networks can be adapted to climate and terrestrial ecosystem forecasting.

The central objective is to learn the underlying spatiotemporal relationships governing ecosystem behaviour from historical data and use these relationships to generate predictions at previously unseen geographical locations.

## Methodology

### 1. Spherical spatial representation

To represent global spatial relationships, the model uses a spherical mesh constructed from a refined icosahedron.

This provides a quasi-uniform discretisation of the Earth's surface, avoiding the spatial distortions associated with conventional latitude–longitude representations.


### 2. Graph-based neural network architecture

The forecasting model follows a grid-to-mesh-to-grid architecture.

The architecture consists of three principal components:

**Grid-to-mesh encoder**

Maps information from the original geographical grid onto the spherical mesh. This allows spatially distributed input features to be represented on a graph suitable for message passing and attention-based processing.

**Mesh processor**

Learns spatial interactions and dependencies between locations on the spherical mesh.

The processor uses graph-based neural network layers and local multi-head self-attention, restricting attention to spatially neighbouring nodes. This enables the model to capture non-local ecosystem relationships while maintaining computational efficiency on large spatial graphs.

**Mesh-to-grid decoder**

Maps the processed mesh representations back to the geographical grid, producing predictions for the target ecosystem variables.

### 3. Spatiotemporal forecasting

The model is trained to learn temporal evolution from historical ecosystem and climate information.

By conditioning predictions on previous states and associated environmental features, the framework aims to capture the temporal dependencies governing ecosystem dynamics.

The forecasting framework is designed to support autoregressive prediction, allowing successive future states to be generated from previous model outputs.


## Model architecture

The main components of the architecture are:

```text
       Historical climate and
       ecosystem information
                  |
                  v
          Grid-to-mesh
             encoder
                  |
                  v
          Spherical mesh
             processor
                  |
          Graph Transformer
          / Local attention
          / Message passing
                  |
                  v
          Mesh-to-grid
             decoder
                  |
                  v
        Predicted ecosystem
             variables
```

## Dataset and evaluation

The project addresses the IEEE BigData Cup 2026 AI Emulation Challenge: **Global-Scale Land Ecosystem Forecasting**


Evaluation is performed using the challenge-provided training and validation data, with predictions generated in the required test-site ordering and format.

## Technologies

The implementation uses:

- **Python** – Core programming language.
- **PyTorch** – Deep learning framework.
- **PyTorch Geometric** – Graph representation and message passing.
- **NumPy** – Data processing and numerical operations.
- **Spherical meshes** – Global spatial representation.
- **Graph Transformers** – Spatial attention and feature processing.


## Acknowledgements

The work is inspired by Google's WeatherNext 2 and its Functional Generative Network approach to medium-range weather forecasting.

**References**

- Google DeepMind, *WeatherNext 2*.
- IEEE BigData Cup 2026, *AI Emulation Challenge: Global-Scale Land Ecosystem Forecasting*.
