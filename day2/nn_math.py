import numpy as np

# relu function
def relu(x):
    return np.maximum(0, x)

# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 

# 2-D convlolution
def conv2d(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1
    output = np.zeros((output_height, output_width))
    
    for i in range(output_height):
        for j in range(output_width):
            region = image[i:i+kernel_height, j:j+kernel_width]
            output[i, j] = np.sum(region * kernel)
    
    return output

print(sigmoid(np.array([-2,-1, 0, 1, 2])))
print(relu(np.array([-2,-1, 0, 1, 2])))
image = np.array([[1, 2, 3, 4,5],
                  [6, 7, 8, 9,10],
                  [11,12,13,14,15],
                  [16,17,18,19,20],
                  [-21,-22,-23,-24,-25]])
kernel = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]])
print(conv2d(image, kernel))


