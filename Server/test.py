import tensorflow as tf
import numpy as np
import cv2
import time
import tensorflow as tf
 
 
print(tf.__version__)
print(tf.test.gpu_device_name())
print(tf.config.experimental.set_visible_devices)
print('GPU:',tf.config.list_physical_devices('GPU'))
print('CPU:',tf.config.list_physical_devices(device_type='CPU'))
print(tf.config.list_physical_devices('GPU'))
print(tf.test.is_gpu_available())

 
 


gpus = tf.config.experimental.list_physical_devices('GPU')

#输出可用的GPU数量
print("Num GPUs Available: ", len(gpus))
#查询GPU设备


if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_visible_devices(devices=gpu,device_type='GPU')
            tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU Setting up!")
    except RuntimeError as e:
        print(e)

tf_lite_path = './mobilenetDetv1.tflite'

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=tf_lite_path, num_threads=24)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# print('Input Details:', input_details)
# print('Output Details:', output_details)

# Load and preprocess the image
img = cv2.imread('TEST.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
img = cv2.resize(img, dsize=(300, 300))     # Resize to (224, 224) for MobileNetV2

# Ensure the image is of type uint8
input_data = np.expand_dims(img, axis=0)  # Add a batch dimension

# Check if the input tensor expects uint8 data and convert if necessary
if input_details[0]['dtype'] == np.uint8:
    input_data = np.array(input_data, dtype=np.uint8)  # Ensure the image is uint8

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], input_data)

# Measure latency by invoking the interpreter multiple times for accurate measurement
num_runs = 10
total_time = 0

for _ in range(num_runs):
    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()
    
    total_time += (end_time - start_time) * 1000  # Convert to milliseconds

# Get the output from the model
output_data = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

# Calculate average inference time
average_time = total_time / num_runs

print(f'Average inference time over {num_runs} runs: {average_time:.2f} ms')
print('Output Data:', output_data)
