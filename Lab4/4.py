import numpy as np
import cv2

def create_gauss_kernel(size, sigma):
    kernel = np.zeros((size, size))
    center = size // 2 
    
    for i in range(size):
        for j in range(size):
            x = i
            y = j
            kernel[i, j] = (1 / (2 * np.pi * sigma**2)) * np.exp((-1) * ((x - center)**2 + (y - center)**2/ (2 * sigma**2)))
    
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

def sobel_operator(image):
    height, width = image.shape
    


    Gx_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    

    Gx = np.zeros((height, width), dtype=np.float32)
    Gy = np.zeros((height, width), dtype=np.float32)
    magnitude = np.zeros((height, width), dtype=np.float32)
    angle = np.zeros((height, width), dtype=np.float32)
    

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            
            gx_val = 0
            gy_val = 0
            
            for i in range(3):
                for j in range(3):
                    gx_val += image[y + i - 1, x + j - 1] * Gx_kernel[i, j]
                    gy_val += image[y + i - 1, x + j - 1] * Gy_kernel[i, j]
            
            Gx[y, x] = gx_val
            Gy[y, x] = gy_val
            
            #длина градиента
            magnitude[y, x] = np.sqrt(gx_val**2 + gy_val**2)
            
            #угол градиента
            if gx_val != 0:
                angle_rad = np.arctan2(gy_val, gx_val)
                angle_deg = np.degrees(angle_rad)
                
                if angle_deg < 0:
                    angle_deg += 180
                angle[y, x] = angle_deg
    
    return magnitude, angle

def non_maximum_suppression(magnitude, angle):
    height, width = magnitude.shape
    suppressed = np.zeros((height, width), dtype=np.float32)
    
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            current_angle = angle[y, x]
            current_magnitude = magnitude[y, x]
            
            direction = get_direction(current_angle)
            

            if direction == 0: 
                neighbor1 = magnitude[y, x - 1]
                neighbor2 = magnitude[y, x + 1]
            elif direction == 1:  # 22.5
                neighbor1 = magnitude[y - 1, x + 1]
                neighbor2 = magnitude[y + 1, x - 1]
            elif direction == 2:  # 45
                neighbor1 = magnitude[y - 1, x]
                neighbor2 = magnitude[y + 1, x]
            elif direction == 3:  # 67.5
                neighbor1 = magnitude[y - 1, x - 1]
                neighbor2 = magnitude[y + 1, x + 1]
            elif direction == 4:  # 90
                neighbor1 = magnitude[y, x - 1]
                neighbor2 = magnitude[y, x + 1]
            elif direction == 5:  # 112.5
                neighbor1 = magnitude[y - 1, x + 1]
                neighbor2 = magnitude[y + 1, x - 1]
            elif direction == 6:  # 135
                neighbor1 = magnitude[y - 1, x]
                neighbor2 = magnitude[y + 1, x]
            elif direction == 7:  # 157.5
                neighbor1 = magnitude[y - 1, x - 1]
                neighbor2 = magnitude[y + 1, x + 1]
            else:
                neighbor1 = 0
                neighbor2 = 0
            
            if current_magnitude >= neighbor1 and current_magnitude >= neighbor2:
                suppressed[y, x] = current_magnitude
            else:
                suppressed[y, x] = 0
    
    return suppressed

def get_direction(angle):
    if angle < 0:
        angle += 180
    
    tg = np.tan(np.radians(angle)) if angle != 90 else float('inf')
    

    if 0 <= angle < 90: 
        if tg < -2.414:
            return 0
        elif -2.414 <= tg < -0.414:
            return 1
        elif -0.414 <= tg < 0.414:
            return 2
        else:  # 0.414 <= tg
            return 3
    else:  # 90 <= angle < 180
        if tg > 2.414:
            return 4
        elif 0.414 < tg <= 2.414:
            return 5
        elif -0.414 <= tg <= 0.414:
            return 6
        else:  # tg < -0.414
            return 7

def double_threshold_filtering(suppressed_magnitude, low_ratio=0.01, high_ratio=0.10):
    height, width = suppressed_magnitude.shape
    result = np.zeros((height, width), dtype=np.uint8)
    
    max_grad = np.max(suppressed_magnitude)
    

    low_level = max_grad * low_ratio
    high_level = max_grad * high_ratio
    
    print(f"Максимальный градиент: {max_grad:.2f}")
    print(f"Нижний порог: {low_level:.2f}")
    print(f"Верхний порог: {high_level:.2f}")
    

    strong_edges = np.zeros((height, width), dtype=bool)
    weak_edges = np.zeros((height, width), dtype=bool)
    
    for y in range(height):
        for x in range(width):
            grad_value = suppressed_magnitude[y, x]
            
            if grad_value >= high_level:
                strong_edges[y, x] = True
                result[y, x] = 255
            elif grad_value >= low_level:
                weak_edges[y, x] = True
                result[y, x] = 128 
            else:
                result[y, x] = 0
    
    final_result = result.copy()
    
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if weak_edges[y, x]:
                has_strong_neighbor = False
                
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue 
                        
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            if strong_edges[ny, nx]:
                                has_strong_neighbor = True
                                break
                    if has_strong_neighbor:
                        break
                
                if has_strong_neighbor:
                    final_result[y, x] = 255
                else:
                    final_result[y, x] = 0
    
    return final_result

def process_image(image_path):
    print("=== ЗАДАНИЕ 1 ===")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка: не удалось загрузить изображение {image_path}")
        return
    

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('1. Original', gray_image)
    cv2.waitKey(0)
    

    kernel_size = 5
    sigma = 15
    kernel = create_gauss_kernel(kernel_size, sigma)
    kernel = normalize_kernel(kernel)
    filtered_image = apply_gauss_filter(gray_image, kernel)
    

    cv2.imshow('2. Gauss', filtered_image)
    cv2.waitKey(0)
    

    print("\n=== ЗАДАНИЕ 2 ===")
    
    magnitude, angle = sobel_operator(filtered_image)
    
    magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    magnitude_normalized = magnitude_normalized.astype(np.uint8)
    
    angle_normalized = cv2.normalize(angle, None, 0, 255, cv2.NORM_MINMAX)
    angle_normalized = angle_normalized.astype(np.uint8)

    cv2.imshow('3. Magnitude (dlina grad)', magnitude_normalized)
    cv2.waitKey(0)
    

    cv2.imshow('4. Angle', angle_normalized)
    cv2.waitKey(0)
    

    print("\n=== ЗАДАНИЕ 3 ===")
    

    suppressed = non_maximum_suppression(magnitude, angle)
    

    suppressed_normalized = cv2.normalize(suppressed, None, 0, 255, cv2.NORM_MINMAX)
    suppressed_normalized = suppressed_normalized.astype(np.uint8)


    cv2.imshow('5. Supress nemax', suppressed_normalized)
    cv2.waitKey(0)


    print("\n=== ЗАДАНИЕ 4 ===")

    

    final_edges = double_threshold_filtering(suppressed)

    

    cv2.imshow('6. Double threshold', final_edges)
    cv2.waitKey(0)
    
    


    cv2.imwrite('1_original.jpg', gray_image)
    cv2.imwrite('2_gaussian.jpg', filtered_image)
    cv2.imwrite('3_gradient_magnitude.jpg', magnitude_normalized)
    cv2.imwrite('4_gradient_angle.jpg', angle_normalized)
    cv2.imwrite('5_non_max_suppression.jpg', suppressed_normalized)
    cv2.imwrite('6_final_edges.jpg', final_edges)
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = "C:\\1.jpg" 
    process_image(image_path)