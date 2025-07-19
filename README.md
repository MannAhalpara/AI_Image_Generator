# Image Generation Studio

This repository contains Python scripts to generate images using popular pre-trained models: Stable Diffusion and DALL-E Mini. Users can choose to generate images with one or both models, providing prompts and various generation parameters.

## Features

* **Stable Diffusion Integration:** Generate high-quality images using the Stable Diffusion v1.5 model.
* **DALL-E Mini Integration:** Create images with the DALL-E Mini (now Craiyon) model.
* **User-Friendly Interface:** Simple command-line interface for inputting prompts and selecting models.
* **Organized Outputs:** Generated images are saved into categorized output directories (`outputs/stable_diffusion` and `outputs/dalle_mini`).

## Setup

### Prerequisites

* Python 3.8+
* `pip` (Python package installer)
* A GPU is highly recommended for faster generation, especially for Stable Diffusion.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/MannAhalpara/AI_Image_Generator.git](https://github.com/MannAhalpara/AI_Image_Generator.git)
    cd your-new-repo-name
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To start the image generation process, run the `main.py` script:

```bash
python main.py
