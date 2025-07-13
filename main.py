import os
from stable_diffusion import generate_stable_diffusion
from dall_e import generate_dalle_mini

def setup_directories():
    """Create output directories if they don't exist."""
    os.makedirs("outputs/stable_diffusion", exist_ok=True)
    os.makedirs("outputs/dalle_mini", exist_ok=True)

def get_user_choice():
    """Get user's choice of image generator."""
    while True:
        print("\nChoose an image generator:")
        print("1. Stable Diffusion")
        print("2. DALL-E Mini")
        print("3. Both")
        choice = input("Enter your choice (1-3): ").strip()
        if choice in ['1', '2', '3']:
            return choice
        print("Invalid choice! Please enter 1, 2, or 3.")

def get_positive_int(prompt, default):
    """Get a positive integer input with a default value."""
    while True:
        try:
            value = input(f"{prompt} (default: {default}): ").strip()
            if value == "":
                return default
            value = int(value)
            if value > 0:
                return value
            print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")

def main():
    """Main function to handle terminal inputs and generate images."""
    setup_directories()

    print("\n=== Image Generation Studio ===")
    prompt = input("Unleash Your Imagination! Enter your image prompt: ").strip()
    if not prompt:
        print("Prompt cannot be empty! Using default prompt.")
        prompt = ""

    choice = get_user_choice()

    sd_outputs = []
    dalle_outputs = []

    if choice in ['1', '3']:  # Stable Diffusion or Both
        num_images = get_positive_int("Enter number of images for Stable Diffusion", 1)
        steps = get_positive_int("Enter number of inference steps for Stable Diffusion", 50)
        print("\nGenerating Stable Diffusion images...")
        sd_outputs = generate_stable_diffusion(prompt, num_images, steps)

    if choice in ['2', '3']:  # DALL-E Mini or Both
        grid_size = get_positive_int("Enter grid size for DALL-E Mini", 1)
        seed = get_positive_int("Enter random seed for DALL-E Mini", 42)
        print("\nGenerating DALL-E Mini images...")
        dalle_outputs = generate_dalle_mini(prompt, grid_size, seed)

    # Summary
    print("\n=== Generation Summary ===")
    print(f"Prompt: {prompt}")

if __name__ == "__main__":
    main()