import skimage
from skimage import transform as sk_transform
from skimage.io import imread, imsave
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter # Ensure ImageEnhance and ImageFilter are available
import random
import os 
import glob

def generate_random_displacement(max_displacement):
    """Generates a small random displacement."""
    return random.uniform(-max_displacement, max_displacement)

def advanced_warp_image(input_image_path, output_image_path,
                        grid_granularity=(5, 7), # Number of control points (rows, cols)
                        max_displacement_ratio=0.05, # Max displacement as a ratio of image dimension
                        affine = True,
                        noise = True,
                        lines = False,
                        photocopy = False,
                        photocopy_cycles = 5):

    """
    Applies a warping effect to an image 
    
    Args to add types of warping:
        
        affine (Boolean): PiecewiseAffineTransform 
        noise (Boolean): Noise (pixels) 
        lines (Boolean): Lines
        photocopy (Boolean): Calls
        

    Attributes for the args are hardcoded, change them below if necessary
    """
    try:
        # 1. Load the image using scikit-image
        image_sk = imread(input_image_path)
        if image_sk.shape[-1] == 4: # Handle RGBA by converting to RGB for simplicity in some effects
            image_sk = skimage.color.rgba2rgb(image_sk)
        
        rows, cols = image_sk.shape[0], image_sk.shape[1]

        # PiecewiseAffineTransform 

        # Define source control points (a regular grid)
        src_cols = np.linspace(0, cols, grid_granularity[1])
        src_rows = np.linspace(0, rows, grid_granularity[0])
        src_rows, src_cols = np.meshgrid(src_rows, src_cols)
        src = np.dstack([src_cols.flat, src_rows.flat])[0]

        # Define destination control points (perturbed source points)
        # Max displacement based on the smaller image dimension
        max_disp = min(rows, cols) * max_displacement_ratio
        
        dst = np.zeros_like(src)
        for i in range(len(src)):
            # Uniformly perturbs points, we can increase the stability of the edges by reducing the displacement for the edge points
            # Current version perturbs all points.
            disp_x = generate_random_displacement(max_disp)
            disp_y = generate_random_displacement(max_disp)
            
            # Ensure destination points stay within image boundaries (optional, warp can handle out-of-bounds)
            dst_x = np.clip(src[i, 0] + disp_x, 0, cols)
            dst_y = np.clip(src[i, 1] + disp_y, 0, rows)
            dst[i] = (dst_x, dst_y)

        # Estimate the transformation
        tform = sk_transform.PiecewiseAffineTransform()
        tform.estimate(src, dst)
        
        if affine:

            # Apply the transformation
            # The output of warp is float64 in range [0, 1] by default
            warped_sk = sk_transform.warp(image_sk, tform.inverse, output_shape=(rows, cols), mode='edge')

            # Convert warped scikit-image (float, 0-1 range) to Pillow Image (uint8, 0-255 range)
            warped_pil = Image.fromarray((warped_sk * 255).astype(np.uint8))
        else: #Otherwise no affine transformation, cstill need to onvert to Pillow Image if other transformations are applied
            warped_pil = image_sk.copy()
        
        if warped_sk.dtype == np.float64 or warped_sk.dtype == np.float32:
            warped_pil = Image.fromarray((warped_sk * 255).astype(np.uint8))
        elif warped_sk.dtype == np.uint8:
            warped_pil = Image.fromarray(warped_sk)
            
        draw = ImageDraw.Draw(warped_pil)
        width, height = warped_pil.size

        # Add uniform noise clusters
        if noise:
            num_noise_dots = int(width * height * random.uniform(0.01, 0.02)) # Uniform Noise across 0.5% to 1.5% of total pixels
            for _ in range(num_noise_dots):
                x = random.randint(0, width - 1)
                y = random.randint(0, height - 1)
                # Make noise color slightly varying around mid-gray or based on pixel neighborhood
                noise_shade = random.randint(50, 200)
                noise_color = (noise_shade, noise_shade, noise_shade)
                if warped_pil.mode == 'RGBA':
                    noise_color += (random.randint(100, 255),) # Add alpha if RGBA
                draw.point((x, y), fill=noise_color)

        #Add arbitrary lines
        if lines:
            num_lines = random.randint(3, 7)
            for _ in range(num_lines):
                start_x = random.randint(0, width - 1)
                start_y = random.randint(0, height - 1)
                end_x = random.randint(0, width - 1)
                end_y = random.randint(0, height - 1)
                line_shade = random.randint(70, 180)
                line_color = (line_shade, line_shade, line_shade)
                if warped_pil.mode == 'RGBA':
                    line_color += (random.randint(100,200),)
                draw.line([(start_x, start_y), (end_x, end_y)], fill=line_color, width=random.randint(1, 2))
                
                
        if photocopy:
            simulate_photocopy_scan_cycles(warped_pil, output_image_path, photocopy_cycles, input_image_path)
        else:
            # Save the final image 
            warped_pil.save(output_image_path)
            print(f"Perturbing '{input_image_path}' and saving as '{output_image_path}'")

    except FileNotFoundError:
        print(f"Error: Input image '{input_image_path}' not found.")
    except Exception as e:
        print(f"Some other error occurred: {e}")
        import traceback
        traceback.print_exc()
        
        
        
def simulate_photocopy_scan_cycles(input_image, output_image_path, cycles, image_name,
                                   contrast_range=(1.1, 1.4),      # How much to boost contrast
                                   brightness_range=(0.95, 1.05),   # brightness fluctuation
                                   blur_radius_range=(0.3, 0.8),    # blur per cycle
                                   noise = True, #pixellated noise
                                   noise_intensity_range=(0.00025, 0.0005), # Proportion of pixels for pixellated noise
                                   max_rotation_angle_deg=1,    # Max rotation in degrees
                                   grayscale_all_cycles=False,       # Conver to greyscale (all current images are already greyscale)
                                   jpeg_compression_quality_range=(50, 90) # Simulate lossy compression
                                  ):
    """
    Simulates the effect of iteratively photocopying and scanning an image.

    Args:
        input_image_path (str): 
        output_image_path (str): P
        cycles (int): Number of photocopy/scan iterations.
        contrast_range (tuple): Min/max factor for contrast enhancement per cycle.
        brightness_range (tuple): Min/max factor for brightness adjustment per cycle.
        blur_radius_range (tuple): Min/max radius for Gaussian blur per cycle.
        noise_intensity_range (tuple): Min/max proportion of pixels to add speckle noise to.
        max_rotation_angle_deg (float): Max degrees for random slight rotation per cycle.
        grayscale_all_cycles ((Boolean) ): If True, converts image to grayscale in each cycle.
        jpeg_compression_quality_range (tuple): Min/max JPEG quality for simulated compression loss.
    """
    try:
        # 1. Load initial image 
        #current_image_pil = Image.open(input_image_path)
        current_image_pil = input_image

        # Check for valid image mode (e.g., RGB or L)
        if current_image_pil.mode == 'RGBA':
            # Create white background 
            background = Image.new('RGB', current_image_pil.size, (255, 255, 255))
            background.paste(current_image_pil, mask=current_image_pil.split()[3]) # Paste using alpha channel as mask
            current_image_pil = background
        elif current_image_pil.mode == 'P': # Palette mode
             current_image_pil = current_image_pil.convert('RGB')


        original_width, original_height = current_image_pil.size

        for i in range(cycles):
            #print(f"Processing cycle {i+1}/{cycles}...")

            #  Apply Grayscale
            if grayscale_all_cycles and current_image_pil.mode != 'L':
                current_image_pil = current_image_pil.convert('L')
            # If already L, or grayscale_all_cycles is False but image became L, ensure it's RGB for color enhancers if needed
            # For simplicity, if it goes to 'L', subsequent PIL enhancers usually handle it or it stays 'L'.
            # If grayscale is true, we expect it to be 'L'. If not, it should be 'RGB'.

            # Ensure RGB if not grayscale and not already RGB 
            if not grayscale_all_cycles and current_image_pil.mode != 'RGB':
                 current_image_pil = current_image_pil.convert('RGB')


            # 1. Contrast Enhancement 
            contrast_factor = random.uniform(contrast_range[0], contrast_range[1])
            enhancer = ImageEnhance.Contrast(current_image_pil)
            current_image_pil = enhancer.enhance(contrast_factor)

            # 2. Brightness Adjustment 
            brightness_factor = random.uniform(brightness_range[0], brightness_range[1])
            enhancer = ImageEnhance.Brightness(current_image_pil)
            current_image_pil = enhancer.enhance(brightness_factor)

            # 3. Gaussian Blur 
            blur_radius = random.uniform(blur_radius_range[0], blur_radius_range[1])
            if blur_radius > 0: # Apply blur only if radius is positive
                current_image_pil = current_image_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))

            # 4.  Rotation 
            angle = random.uniform(-max_rotation_angle_deg, max_rotation_angle_deg)
            # Rotate, fill background with white, allow expansion, then crop back to original size to simulate placing on a scanner bed.
            if angle != 0:
                # Use white background for rotation fill
                fill_color = 'white' if current_image_pil.mode == 'L' else (255, 255, 255)
                rotated_image = current_image_pil.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=fill_color)

                # Crop back to original dimensions, centered
                w_rot, h_rot = rotated_image.size
                x_offset = (w_rot - original_width) / 2
                y_offset = (h_rot - original_height) / 2
                current_image_pil = rotated_image.crop((x_offset, y_offset, x_offset + original_width, y_offset + original_height))


            # add noise 
            if noise:
                noise_intensity = random.uniform(noise_intensity_range[0], noise_intensity_range[1])
                num_noise_dots = int(original_width * original_height * noise_intensity)
                draw = ImageDraw.Draw(current_image_pil)
                for _ in range(num_noise_dots):
                    x = random.randint(0, original_width - 1)
                    y = random.randint(0, original_height - 1)
                    # Noise color
                    noise_shade = random.randint(0, 100) # Darker speckles
                    if current_image_pil.mode == 'L':
                        dot_color = noise_shade
                    else: # RGB
                        dot_color = (noise_shade, noise_shade, noise_shade)
                    draw.point((x, y), fill=dot_color)
                del draw # Release the drawing context

            # 6. Simulate JPEG-like compression artifacts

            if cycles > 0 : # Avoid if no degradation cycles are specified beyond initial load
                temp_output_path = os.path.join(os.path.dirname(output_image_path), f"_temp_cycle_{i}.jpg")
                quality = random.randint(jpeg_compression_quality_range[0], jpeg_compression_quality_range[1])
                current_image_pil.save(temp_output_path, "JPEG", quality=quality)
                current_image_pil = Image.open(temp_output_path)
                os.remove(temp_output_path) # Clean up temporary file

                # If the image was grayscale, opening JPEG might make it RGB. Convert back.
                if grayscale_all_cycles and current_image_pil.mode != 'L':
                    current_image_pil = current_image_pil.convert('L')
                elif not grayscale_all_cycles and current_image_pil.mode != 'RGB' and current_image_pil.mode != 'L':
                    # If after JPEG load ttype is non-standard, try to normalize
                    current_image_pil = current_image_pil.convert('RGB')


        # Save image
        current_image_pil.save(output_image_path)
        print(f"Perturbed image saved as '{output_image_path}' after {cycles} cycles.")


    except Exception as e:
        print(f"Some error occured for image:  '{image_name}': {e}")
        import traceback
        traceback.print_exc()        
        
        
        
def main():
    
    INPUT_FOLDER = "Circuit Problems"  # Replace with the path to your input folder
                                  

    OUTPUT_FOLDER = "perturbed_images" # Replace with your desired output folder path
                                      
    
    # perturbed image filename suffix
    FILENAME_SUFFIX = "_perturbed"
    
    # valid image types, all should be png by default
    IMAGE_EXTENSIONS = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif"]
    
    RANDOM_SEED = 1
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    """
    Main function
    """
    print(f"Input folder: {os.path.abspath(INPUT_FOLDER)}")
    print(f"Output folder: {os.path.abspath(OUTPUT_FOLDER)}")
    print(f"Filename suffix: '{FILENAME_SUFFIX}'")
    print(f"Processing extensions: {', '.join(IMAGE_EXTENSIONS)}")
    print("-" * 30)


    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"Ensured output folder '{OUTPUT_FOLDER}' exists.")

    processed_count = 0
    found_files = []


    for ext_pattern in IMAGE_EXTENSIONS:
        search_path = os.path.join(INPUT_FOLDER, ext_pattern)
        found_files.extend(glob.glob(search_path))

    if not found_files:
        print(f"No images found in '{INPUT_FOLDER}' with the specified extensions.")
        return

    print(f"Found {len(found_files)} image(s) to process.")
    
    try:

        for input_image_path in found_files:
    
            base_name = os.path.basename(input_image_path)
    
    
            name_part, extension_part = os.path.splitext(base_name)
    
    
            output_filename = f"{name_part}{FILENAME_SUFFIX}{extension_part}"
    
    
            output_image_path = os.path.join(OUTPUT_FOLDER, output_filename)
    
    
            advanced_warp_image(input_image_path, output_image_path)
            
            
            processed_count += 1
            
            
            
            display = False
            if display:
                
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                original_img_display = imread(input_image_path)
                warped_img_display = imread(output_image_path)
                axes[0].imshow(original_img_display)
                axes[0].set_title("Original")
                axes[0].axis('off')
                axes[1].imshow(warped_img_display)
                axes[1].set_title("Perturbed")
                axes[1].axis('off')
                plt.show()
            
    except ImportError as ie:
            print(f"ImportError: {ie}. Libraries not imported correctly.")
            print(ie)
    except Exception as e:
            print(f"Some other error occurred: {e}")
            import traceback
            traceback.print_exc()

    print("-" * 30)
    print(f"Processing complete. {processed_count} image(s) perturbed.")

if __name__ == "__main__":
    # This makes the script runnable from the command line
    main()
