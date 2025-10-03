import numpy as np
import cv2


def create_gauss_kernel(size, sigma):

    kernel = np.zeros((size, size))
    center = size // 2 
    
    print(f"Создание матрицы {size}x{size}, Sigma={sigma}")
    print(f"Центр матрицы: ({center}, {center})")
    
    for i in range(size):
        for j in range(size):
            x = i
            y = j
            distance_sq = (x - center)**2 + (y - center)**2
            kernel[i, j] = (1 / (2 * np.pi * sigma**2)) * np.exp((-1) * (distance_sq / (2 * sigma**2)))
    
    return kernel

def normalize_kernel(kernel):

    kernel_sum = np.sum(kernel)
    normalized = kernel / kernel_sum
    return normalized


def apply_gauss_filter(image, kernel):

    filtered_image = image.copy()
    
    kernel_size = kernel.shape[0]

    padding = kernel_size // 2
    
    height, width = image.shape
    
    print(f"Размер изображения: {height}x{width}")
    print(f"Размер ядра: {kernel_size}x{kernel_size}")
    print(f"Обрабатываемые пиксели: [{padding}:{height-padding}, {padding}:{width-padding}]")
    

    for y in range(padding, height - padding):
        for x in range(padding, width - padding):

            start_y = y - padding
            end_y = y + padding + 1
            start_x = x - padding
            end_x = x + padding + 1
            
            okno = image[start_y:end_y, start_x:end_x]
            
            new_value = 0
            for k in range(kernel_size):
                for l in range(kernel_size):
                    new_value += okno[k, l] * kernel[k, l]
            
            filtered_image[y, x] = new_value
    
    return filtered_image


print("ЗАДАНИЕ 1")

sizes = [3, 5, 7]
sigma = 1.0

gaussian_kernels = {}

for size in sizes:
    kernel = create_gauss_kernel(size, sigma)
    gaussian_kernels[size] = kernel
    
    print(f"\nМатрица Гаусса {size}x{size} (до нормирования):")
    for row in kernel:
        print(" ".join([f"{val:.3f}" for val in row]))

    print(f"Сумма элементов: {np.sum(kernel):.3f}")

    print("-" * 40)



print("ЗАДАНИЕ 2")

normalized_kernels = {}

for size in sizes:
    original_kernel = gaussian_kernels[size]
    normalized_kernel = normalize_kernel(original_kernel)
    normalized_kernels[size] = normalized_kernel
    
    print(f"\nНормированная матрица {size}x{size}:")
    for row in normalized_kernel:
        print(" ".join([f"{val:.3f}" for val in row]))
    print(f"Сумма элементов: {np.sum(normalized_kernel):.3f}")
    
    print("-" * 40)



print("ЗАДАНИЕ 3-4-5")

pickme = cv2.imread("C:\\1.jpg")
pickme_gray = cv2.cvtColor(pickme, cv2.COLOR_BGR2GRAY)

opencv_filtered = cv2.GaussianBlur(pickme_gray, (5, 5), 5)

kernel = create_gauss_kernel(5, 5)
kernel = normalize_kernel(kernel)
my_filtered_pickme = apply_gauss_filter(pickme_gray, kernel)


cv2.imshow('Original', pickme_gray)
cv2.imshow('MY GAUSS', my_filtered_pickme)
cv2.imshow('OpenCV GAUSS', opencv_filtered)


cv2.waitKey(0)
cv2.destroyAllWindows()