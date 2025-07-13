import torch
from min_dalle import MinDalle
from PIL import Image
import gc
import os

def generate_dalle_mini(prompt, grid_size=1, seed=42, output_dir="outputs/dalle_mini"):
    """Generate images using DALL-E Mini model."""
    try:
        print("Loading DALL-E Mini model...")
        model = MinDalle(
            models_root='./pretrained',
            dtype=torch.float16,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            is_mega=True,
            is_reusable=True
        )

        # Generate image
        print("Generating DALL-E Mini images...")
        image = model.generate_image(
            text=prompt,
            seed=seed,
            grid_size=grid_size,
            temperature=1.0,
            top_k=32,
            supercondition_factor=16,
            is_verbose=True
        )

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Save grid and single tile
        grid_output = f"{output_dir}/dalle_grid_{prompt[:20].replace(' ', '_')}.png"
        image.save(grid_output)
        print(f"Saved DALL-E Mini grid: {grid_output}")

        tile_size = image.width // grid_size
        single = image.crop((0, 0, tile_size, tile_size))
        tile_output = f"{output_dir}/dalle_tile0_{prompt[:20].replace(' ', '_')}.png"
        single.save(tile_output)
        print(f"Saved DALL-E Mini tile: {tile_output}")

        # Clean up
        del image, single, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return [grid_output, tile_output]

    except Exception as e:
        print(f"Error in DALL-E Mini generation: {str(e)}")
        return []