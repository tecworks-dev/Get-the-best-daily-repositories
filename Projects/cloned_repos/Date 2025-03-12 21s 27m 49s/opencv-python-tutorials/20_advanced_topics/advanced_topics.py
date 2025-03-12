#!/usr/bin/env python3
"""
Advanced Topics in OpenCV
This script demonstrates advanced techniques and optimizations using OpenCV
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from multiprocessing import Pool, cpu_count

class GPUAcceleration:
    def __init__(self):
        """Initialize GPU acceleration class"""
        # Check CUDA availability
        self.has_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
        
        if self.has_cuda:
            print("CUDA is available. GPU acceleration enabled.")
            self.gpu_stream = cv2.cuda_Stream()
        else:
            print("CUDA is not available. Using CPU.")
    
    def to_gpu(self, image):
        """
        Transfer image to GPU memory
        
        Args:
            image: Input image
        
        Returns:
            GPU matrix or original image if CUDA is not available
        """
        if self.has_cuda:
            return cv2.cuda_GpuMat(image)
        return image
    
    def from_gpu(self, gpu_mat):
        """
        Transfer image from GPU memory
        
        Args:
            gpu_mat: GPU matrix or original image
        
        Returns:
            CPU image
        """
        if self.has_cuda and isinstance(gpu_mat, cv2.cuda_GpuMat):
            return gpu_mat.download()
        return gpu_mat
    
    def gaussian_blur(self, image, kernel_size=(5, 5), sigma=0):
        """
        GPU-accelerated Gaussian blur
        
        Args:
            image: Input image
            kernel_size: Size of Gaussian kernel
            sigma: Standard deviation
        
        Returns:
            Blurred image
        """
        if self.has_cuda:
            gpu_image = self.to_gpu(image)
            gpu_result = cv2.cuda.createGaussianFilter(
                gpu_image.type(), gpu_image.type(),
                kernel_size, sigma).apply(gpu_image)
            return self.from_gpu(gpu_result)
        return cv2.GaussianBlur(image, kernel_size, sigma)
    
    def canny_edge(self, image, threshold1=50, threshold2=150):
        """
        GPU-accelerated Canny edge detection
        
        Args:
            image: Input image
            threshold1: First threshold
            threshold2: Second threshold
        
        Returns:
            Edge image
        """
        if self.has_cuda:
            gpu_image = self.to_gpu(image)
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gpu_gray = cv2.cuda.cvtColor(gpu_image, cv2.COLOR_BGR2GRAY)
            else:
                gpu_gray = gpu_image
            
            # Apply Canny
            gpu_edges = cv2.cuda.createCannyEdgeDetector(threshold1, threshold2).detect(gpu_gray)
            return self.from_gpu(gpu_edges)
        
        # CPU implementation
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return cv2.Canny(gray, threshold1, threshold2)
    
    def benchmark(self, image, iterations=10):
        """
        Benchmark GPU vs CPU performance
        
        Args:
            image: Input image
            iterations: Number of iterations for benchmarking
        
        Returns:
            Dictionary with benchmark results
        """
        results = {}
        
        # Benchmark Gaussian blur
        cpu_times = []
        gpu_times = []
        
        for _ in range(iterations):
            # CPU timing
            start = time.time()
            _ = cv2.GaussianBlur(image, (5, 5), 0)
            cpu_times.append(time.time() - start)
            
            # GPU timing (if available)
            if self.has_cuda:
                start = time.time()
                _ = self.gaussian_blur(image, (5, 5), 0)
                gpu_times.append(time.time() - start)
        
        results['gaussian_blur'] = {
            'cpu_avg': np.mean(cpu_times),
            'gpu_avg': np.mean(gpu_times) if self.has_cuda else None,
            'speedup': np.mean(cpu_times) / np.mean(gpu_times) if self.has_cuda else None
        }
        
        # Benchmark Canny edge detection
        cpu_times = []
        gpu_times = []
        
        for _ in range(iterations):
            # CPU timing
            start = time.time()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            _ = cv2.Canny(gray, 50, 150)
            cpu_times.append(time.time() - start)
            
            # GPU timing (if available)
            if self.has_cuda:
                start = time.time()
                _ = self.canny_edge(image, 50, 150)
                gpu_times.append(time.time() - start)
        
        results['canny_edge'] = {
            'cpu_avg': np.mean(cpu_times),
            'gpu_avg': np.mean(gpu_times) if self.has_cuda else None,
            'speedup': np.mean(cpu_times) / np.mean(gpu_times) if self.has_cuda else None
        }
        
        return results

class CustomFilters:
    @staticmethod
    def guided_filter(guide, src, radius=10, eps=1e-6):
        """
        Apply guided filter
        
        Args:
            guide: Guidance image
            src: Source image
            radius: Filter radius
            eps: Regularization parameter
        
        Returns:
            Filtered image
        """
        # Convert to float32
        guide = guide.astype(np.float32)
        src = src.astype(np.float32)
        
        # Get dimensions
        height, width = src.shape[:2]
        
        # Mean of guide image
        mean_guide = cv2.boxFilter(guide, -1, (radius, radius))
        
        # Mean of source image
        mean_src = cv2.boxFilter(src, -1, (radius, radius))
        
        # Mean of guide * source
        mean_guide_src = cv2.boxFilter(guide * src, -1, (radius, radius))
        
        # Variance of guide
        var_guide = cv2.boxFilter(guide * guide, -1, (radius, radius)) - mean_guide * mean_guide
        
        # Covariance of guide and source
        cov_guide_src = mean_guide_src - mean_guide * mean_src
        
        # Linear coefficients
        a = cov_guide_src / (var_guide + eps)
        b = mean_src - a * mean_guide
        
        # Mean of linear coefficients
        mean_a = cv2.boxFilter(a, -1, (radius, radius))
        mean_b = cv2.boxFilter(b, -1, (radius, radius))
        
        # Final output
        output = mean_a * guide + mean_b
        
        return output.astype(np.uint8)
    
    @staticmethod
    def non_local_means(image, h=10, search_window=21, block_size=7):
        """
        Apply non-local means denoising
        
        Args:
            image: Input image
            h: Filter strength
            search_window: Size of search window
            block_size: Size of block for computing weight
        
        Returns:
            Denoised image
        """
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(
                image, None, h, h, search_window, block_size)
        else:
            return cv2.fastNlMeansDenoising(
                image, None, h, search_window, block_size)
    
    @staticmethod
    def anisotropic_diffusion(image, num_iter=15, delta_t=0.14, kappa=15, option=1):
        """
        Apply anisotropic diffusion filter
        
        Args:
            image: Input image
            num_iter: Number of iterations
            delta_t: Time step
            kappa: Conductance parameter
            option: Diffusion function (1 or 2)
        
        Returns:
            Filtered image
        """
        # Convert to float32
        img = image.astype(np.float32)
        
        # Handle multi-channel images
        if len(img.shape) == 3:
            result = np.zeros_like(img)
            for i in range(img.shape[2]):
                result[:, :, i] = CustomFilters.anisotropic_diffusion(
                    img[:, :, i], num_iter, delta_t, kappa, option)
            return result
        
        # Create copy of image
        diff_img = img.copy()
        
        # Iterate
        for _ in range(num_iter):
            # Calculate gradients
            north = np.roll(diff_img, -1, axis=0)
            south = np.roll(diff_img, 1, axis=0)
            east = np.roll(diff_img, 1, axis=1)
            west = np.roll(diff_img, -1, axis=1)
            
            # Calculate differences
            diff_north = north - diff_img
            diff_south = south - diff_img
            diff_east = east - diff_img
            diff_west = west - diff_img
            
            # Calculate diffusion function
            if option == 1:
                # Perona-Malik diffusion function 1
                cn = np.exp(-(diff_north / kappa) ** 2)
                cs = np.exp(-(diff_south / kappa) ** 2)
                ce = np.exp(-(diff_east / kappa) ** 2)
                cw = np.exp(-(diff_west / kappa) ** 2)
            else:
                # Perona-Malik diffusion function 2
                cn = 1.0 / (1.0 + (diff_north / kappa) ** 2)
                cs = 1.0 / (1.0 + (diff_south / kappa) ** 2)
                ce = 1.0 / (1.0 + (diff_east / kappa) ** 2)
                cw = 1.0 / (1.0 + (diff_west / kappa) ** 2)
            
            # Update image
            diff_img = diff_img + delta_t * (
                cn * diff_north + cs * diff_south + ce * diff_east + cw * diff_west)
        
        return diff_img.astype(np.uint8)

class FrequencyDomainProcessing:
    @staticmethod
    def dft(image):
        """
        Apply Discrete Fourier Transform
        
        Args:
            image: Input image
        
        Returns:
            Magnitude spectrum and phase spectrum
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Expand image to optimal size
        rows, cols = gray.shape
        optimal_rows = cv2.getOptimalDFTSize(rows)
        optimal_cols = cv2.getOptimalDFTSize(cols)
        padded = cv2.copyMakeBorder(gray, 0, optimal_rows - rows, 0, optimal_cols - cols,
                                  cv2.BORDER_CONSTANT, value=0)
        
        # Apply DFT
        dft = cv2.dft(np.float32(padded), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        # Compute magnitude and phase spectra
        magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
        phase = cv2.phase(dft_shift[:, :, 0], dft_shift[:, :, 1])
        
        # Convert magnitude to logarithmic scale for better visualization
        magnitude = 20 * np.log(magnitude + 1)
        
        # Normalize magnitude for display
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        phase = cv2.normalize(phase, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return magnitude, phase, dft_shift
    
    @staticmethod
    def inverse_dft(dft_shift):
        """
        Apply Inverse Discrete Fourier Transform
        
        Args:
            dft_shift: Shifted DFT result
        
        Returns:
            Reconstructed image
        """
        # Shift back
        dft_ishift = np.fft.ifftshift(dft_shift)
        
        # Inverse DFT
        img_back = cv2.idft(dft_ishift)
        
        # Compute magnitude
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        
        # Normalize
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return img_back
    
    @staticmethod
    def frequency_filter(image, filter_type='lowpass', cutoff=30):
        """
        Apply frequency domain filtering
        
        Args:
            image: Input image
            filter_type: Type of filter ('lowpass', 'highpass', 'bandpass')
            cutoff: Cutoff frequency
        
        Returns:
            Filtered image
        """
        # Get DFT
        magnitude, phase, dft_shift = FrequencyDomainProcessing.dft(image)
        
        # Get dimensions
        rows, cols = magnitude.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create filter mask
        mask = np.zeros((rows, cols, 2), np.float32)
        
        if filter_type == 'lowpass':
            # Low-pass filter: keep frequencies inside circle
            center = [crow, ccol]
            x, y = np.ogrid[:rows, :cols]
            mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= cutoff ** 2
            mask[mask_area] = 1
        elif filter_type == 'highpass':
            # High-pass filter: keep frequencies outside circle
            center = [crow, ccol]
            x, y = np.ogrid[:rows, :cols]
            mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 > cutoff ** 2
            mask[mask_area] = 1
        elif filter_type == 'bandpass':
            # Band-pass filter: keep frequencies between inner and outer circles
            center = [crow, ccol]
            x, y = np.ogrid[:rows, :cols]
            inner_radius = cutoff // 2
            outer_radius = cutoff
            mask_area = np.logical_and(
                (x - center[0]) ** 2 + (y - center[1]) ** 2 >= inner_radius ** 2,
                (x - center[0]) ** 2 + (y - center[1]) ** 2 <= outer_radius ** 2
            )
            mask[mask_area] = 1
        
        # Apply filter
        filtered_dft = dft_shift * mask
        
        # Inverse DFT
        filtered_image = FrequencyDomainProcessing.inverse_dft(filtered_dft)
        
        return filtered_image, mask[:, :, 0]

class ParallelProcessing:
    @staticmethod
    def process_tile(args):
        """
        Process a single tile
        
        Args:
            args: Tuple containing (tile, function, args, kwargs)
        
        Returns:
            Processed tile
        """
        tile, func, args, kwargs = args
        return func(tile, *args, **kwargs)
    
    @staticmethod
    def parallel_process(image, func, tile_size=(100, 100), *args, **kwargs):
        """
        Process image in parallel using tiling
        
        Args:
            image: Input image
            func: Processing function
            tile_size: Size of tiles
            args: Additional arguments for func
            kwargs: Additional keyword arguments for func
        
        Returns:
            Processed image
        """
        # Get dimensions
        height, width = image.shape[:2]
        
        # Create tiles
        tiles = []
        positions = []
        
        for y in range(0, height, tile_size[0]):
            for x in range(0, width, tile_size[1]):
                # Get tile
                tile_height = min(tile_size[0], height - y)
                tile_width = min(tile_size[1], width - x)
                tile = image[y:y+tile_height, x:x+tile_width]
                
                # Store tile and position
                tiles.append(tile)
                positions.append((y, x, tile_height, tile_width))
        
        # Process tiles in parallel
        with Pool(processes=cpu_count()) as pool:
            processed_tiles = pool.map(
                ParallelProcessing.process_tile,
                [(tile, func, args, kwargs) for tile in tiles]
            )
        
        # Reconstruct image
        result = np.zeros_like(image)
        
        for (y, x, h, w), tile in zip(positions, processed_tiles):
            result[y:y+h, x:x+w] = tile
        
        return result

def create_sample_images():
    """Create sample images for testing"""
    # Create a simple image with shapes
    img1 = np.zeros((512, 512, 3), dtype=np.uint8)
    img1.fill(255)  # White background
    
    # Add some shapes
    cv2.rectangle(img1, (100, 100), (200, 200), (0, 0, 255), -1)  # Red rectangle
    cv2.circle(img1, (350, 250), 100, (0, 255, 0), -1)  # Green circle
    cv2.line(img1, (100, 400), (400, 400), (255, 0, 0), 10)  # Blue line
    
    # Create a noisy image
    img2 = img1.copy()
    noise = np.random.normal(0, 25, img2.shape).astype(np.uint8)
    img2 = cv2.add(img2, noise)
    
    # Create a blurry image
    img3 = cv2.GaussianBlur(img1, (21, 21), 5)
    
    return img1, img2, img3

def display_results(results, title, cols=3):
    """Display results using matplotlib"""
    n = len(results)
    rows = (n + cols - 1) // cols
    
    plt.figure(figsize=(15, 5 * rows))
    plt.suptitle(title, fontsize=16)
    
    for i, (name, img) in enumerate(results.items()):
        plt.subplot(rows, cols, i+1)
        
        # Handle different color spaces
        if len(img.shape) == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap='gray')
            
        plt.title(name)
        plt.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def main():
    """Main function"""
    print("Advanced Topics in OpenCV")
    print("=======================")
    
    # Create or load sample images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), 'images')
    
    # Create sample images
    img1, img2, img3 = create_sample_images()
    
    # Save sample images if needed
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    cv2.imwrite(os.path.join(images_dir, 'advanced_sample1.jpg'), img1)
    cv2.imwrite(os.path.join(images_dir, 'advanced_sample2.jpg'), img2)
    cv2.imwrite(os.path.join(images_dir, 'advanced_sample3.jpg'), img3)
    
    while True:
        print("\nSelect a demo:")
        print("1. GPU Acceleration")
        print("2. Custom Filters")
        print("3. Frequency Domain Processing")
        print("4. Parallel Processing")
        print("q. Quit")
        
        choice = input("\nEnter your choice: ")
        
        if choice == '1':
            print("\nGPU Acceleration Demo")
            
            # Initialize GPU acceleration
            gpu = GPUAcceleration()
            
            # Apply GPU-accelerated operations
            start_time = time.time()
            gpu_blur = gpu.gaussian_blur(img2, (15, 15), 0)
            gpu_time = time.time() - start_time
            
            # Apply CPU operations for comparison
            start_time = time.time()
            cpu_blur = cv2.GaussianBlur(img2, (15, 15), 0)
            cpu_time = time.time() - start_time
            
            # Display results
            results = {
                'original': img2,
                f'GPU blur ({gpu_time:.4f}s)': gpu_blur,
                f'CPU blur ({cpu_time:.4f}s)': cpu_blur
            }
            
            display_results(results, "GPU vs CPU Acceleration")
            
            # Benchmark
            print("\nBenchmarking GPU vs CPU performance...")
            benchmark_results = gpu.benchmark(img1)
            
            for operation, result in benchmark_results.items():
                print(f"\n{operation.upper()}:")
                print(f"  CPU average time: {result['cpu_avg']:.6f} seconds")
                if result['gpu_avg'] is not None:
                    print(f"  GPU average time: {result['gpu_avg']:.6f} seconds")
                    print(f"  Speedup: {result['speedup']:.2f}x")
                else:
                    print("  GPU not available")
            
        elif choice == '2':
            print("\nCustom Filters Demo")
            
            # Apply custom filters
            guided = CustomFilters.guided_filter(img1, img2, radius=10, eps=1e-6)
            nlm = CustomFilters.non_local_means(img2, h=10)
            anisotropic = CustomFilters.anisotropic_diffusion(img2, num_iter=15)
            
            # Display results
            results = {
                'original': img2,
                'guided_filter': guided,
                'non_local_means': nlm,
                'anisotropic_diffusion': anisotropic
            }
            
            display_results(results, "Custom Filters")
            
        elif choice == '3':
            print("\nFrequency Domain Processing Demo")
            
            # Apply DFT
            magnitude, phase, dft_shift = FrequencyDomainProcessing.dft(img1)
            
            # Apply frequency filters
            lowpass, lowpass_mask = FrequencyDomainProcessing.frequency_filter(
                img1, filter_type='lowpass', cutoff=30)
            
            highpass, highpass_mask = FrequencyDomainProcessing.frequency_filter(
                img1, filter_type='highpass', cutoff=30)
            
            bandpass, bandpass_mask = FrequencyDomainProcessing.frequency_filter(
                img1, filter_type='bandpass', cutoff=50)
            
            # Display results
            results = {
                'original': img1,
                'magnitude_spectrum': magnitude,
                'phase_spectrum': phase,
                'lowpass_filter': lowpass,
                'lowpass_mask': lowpass_mask * 255,
                'highpass_filter': highpass,
                'highpass_mask': highpass_mask * 255,
                'bandpass_filter': bandpass,
                'bandpass_mask': bandpass_mask * 255
            }
            
            display_results(results, "Frequency Domain Processing", cols=3)
            
        elif choice == '4':
            print("\nParallel Processing Demo")
            
            # Define processing function
            def process_func(tile, kernel_size=(15, 15)):
                return cv2.GaussianBlur(tile, kernel_size, 0)
            
            # Process image in parallel
            start_time = time.time()
            parallel_result = ParallelProcessing.parallel_process(
                img1, process_func, tile_size=(128, 128), kernel_size=(15, 15))
            parallel_time = time.time() - start_time
            
            # Process image sequentially for comparison
            start_time = time.time()
            sequential_result = process_func(img1, kernel_size=(15, 15))
            sequential_time = time.time() - start_time
            
            # Display results
            results = {
                'original': img1,
                f'parallel ({parallel_time:.4f}s)': parallel_result,
                f'sequential ({sequential_time:.4f}s)': sequential_result
            }
            
            display_results(results, "Parallel vs Sequential Processing")
            
            # Print speedup
            speedup = sequential_time / parallel_time
            print(f"\nParallel processing speedup: {speedup:.2f}x")
            
        elif choice.lower() == 'q':
            break
        
        else:
            print("\nInvalid choice. Please try again.")
    
    print("\nDemo completed. Thank you!")

if __name__ == "__main__":
    main()