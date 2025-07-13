import torch
from diffusers import StableDiffusionPipeline
import gc
import os

def generate_stable_diffusion(prompt, num_images=1, steps=50, output_dir="outputs/stable_diffusion"):
    """Generate images using Stable Diffusion model."""
    try:
        print("Loading Stable Diffusion model...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        
        # Move to GPU if available, else CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        print(f"Using device: {device}")

        # Generate images
        print("Generating Stable Diffusion images...")
        images = pipe(
            prompt,
            num_inference_steps=steps,
            num_images_per_prompt=num_images
        ).images

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Save images
        output_paths = []
        for idx, img in enumerate(images):
            output_path = f"{output_dir}/sd_{idx}_{prompt[:20].replace(' ', '_')}.png"
            img.save(output_path)
            output_paths.append(output_path)
            print(f"Saved Stable Diffusion image: {output_path}")

        # Clean up
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return output_paths

    except Exception as e:
        print(f"Error in Stable Diffusion generation: {str(e)}")
        return []