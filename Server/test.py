import tensorflow as tf
import numpy as np
import cv2
import time
import tensorflow as tf
from PIL import Image
import cupy as cp
 
 
print(tf.__version__)
print(tf.test.gpu_device_name())
print(tf.config.experimental.set_visible_devices)
print('GPU:',tf.config.list_physical_devices('GPU'))
print('CPU:',tf.config.list_physical_devices(device_type='CPU'))
print(tf.config.list_physical_devices('GPU'))
print(tf.test.is_gpu_available())



print("*"*50)
models = {
          0:tf.lite.Interpreter(model_path=".\deconv_fin_munet.tflite"),
          1:tf.lite.Interpreter(model_path=".\mobilenet_v2_1.0_224_quant.tflite"),
          2:tf.lite.Interpreter(model_path=".\mobilenet_v1_1.0_224_quant.tflite"),
          3:tf.lite.Interpreter(model_path=".\ssd_mobilenet_v1_1_metadata_1.tflite"),
          4:tf.lite.Interpreter(model_path=".\mobilenetDetv1.tflite"),
          5:tf.lite.Interpreter(model_path=".\efficientclass-lite0.tflite"),
          6:tf.lite.Interpreter(model_path=".\inception_v1_224_quant.tflite"),  
          7:tf.lite.Interpreter(model_path=".\mobilenetClassv1.tflite"),        
          8:tf.lite.Interpreter(model_path=".\deeplabv3.tflite"),
          9:tf.lite.Interpreter(model_path=".\model_metadata.tflite"),
          10:tf.lite.Interpreter(model_path=".\mnist.tflite")}

model_name ={0:"deconv_fin_munet",
             1:"mobilenet_v2_1.0_224_quant",
             2:"mobilenet_v1_1.0_224_quant",
             3:"ssd_mobilenet_v1_1_metadata_1",
             4:"mobilenetDetv1",
             5:"efficientclass-lite0",
             6:"inception_v1_224_quant",
             7:"mobilenetClassv1",
             8:"deeplabv3",
             9:"model_metadata",
             10:"mnist"}
 
for i in range(11):
    models[i].allocate_tensors()
    print(f"{model_name[i]} Model Loaded Successfully!")

# def preprocess_image(image, model_input_details):
#     """
#     Preprocesses the image according to the model's input requirements.

#     Args:
#     image (PIL.Image): The input image.
#     model_input_details (dict): A dictionary containing the model's input details.

#     Returns:
#     numpy.ndarray: The preprocessed image tensor.
#     """
#     # Extract input details for the first model (assuming it's the one you're using)
#     input_dict = model_input_details[0]
#     input_shape = input_dict['shape']
    
#     # Resize image to match the input dimensions required by the model
#     image = image.resize((input_shape[2], input_shape[1]), Image.Resampling.LANCZOS)

#     # Convert image to numpy array and adjust type accordingly
#     image = np.array(image)
#     if input_dict['dtype'] == np.float32:
#         image = image.astype(np.float32)
#         # Normalize image if there are no quantization parameters, or adjust according to them
#         if 'quantization_parameters' in input_dict and input_dict['quantization_parameters']['scales'].size > 0:
#             scale = input_dict['quantization_parameters']['scales'][0]
#             zero_point = input_dict['quantization_parameters']['zero_points'][0]
#             image = (image - zero_point) * scale
#         else:
#             image = image / 255.0
#     else:
#         image = image.astype(np.uint8)
#         if 'quantization_parameters' in input_dict and input_dict['quantization_parameters']['scales'].size > 0:
#             scale = input_dict['quantization_parameters']['scales'][0]
#             zero_point = input_dict['quantization_parameters']['zero_points'][0]
#             image = ((image - zero_point) * scale).astype(np.uint8)

#     # Add batch dimension
#     image = np.expand_dims(image, axis=0)

#     return image
def preprocess_image(image_path, model_input_details):
    """
    Preprocesses the image according to the model's input requirements using OpenCV for speed.

    Args:
    image_path (str): The path to the input image.
    model_input_details (dict): A dictionary containing the model's input details.

    Returns:
    numpy.ndarray: The preprocessed image tensor.
    """
    input_dict = model_input_details[0]
    input_shape = input_dict['shape']

    # Load image using OpenCV (loads as BGR)
    image = cv2.imread(image_path)

    # Resize the image using OpenCV if necessary
    if image.shape[:2] != (input_shape[1], input_shape[2]):
        image = cv2.resize(image, (input_shape[2], input_shape[1]))

    # Convert BGR to RGB, as TensorFlow models often expect RGB input
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # If the model expects float32 input, normalize and convert
    if input_dict['dtype'] == np.float32:
        image = image.astype(np.float32)
        if 'quantization_parameters' in input_dict and input_dict['quantization_parameters']['scales'].size > 0:
            scale = input_dict['quantization_parameters']['scales'][0]
            zero_point = input_dict['quantization_parameters']['zero_points'][0]
            image = (image - zero_point) * scale
        else:
            # Normalize image to range [0, 1]
            image /= 255.0
    else:
        image = image.astype(np.uint8)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image



gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))



if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_visible_devices(devices=gpu,device_type='GPU')
            tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU Setting up!")
    except RuntimeError as e:
        print(e)

print("*"*50)



# print('Input Details:', input_details)
# print('Output Details:', output_details)

# Load and preprocess the image
# img = cv2.imread('Server\TEST.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
# img = cv2.resize(img, dsize=(300, 300))     # Resize to (224, 224) for MobileNetV2



latency = []
for i in range(11):
    
    input_details = models[i].get_input_details()
    output_details = models[i].get_output_details()
    start_time = time.time()
    timer1 = time.perf_counter()
    # Preprocess the image
    img = preprocess_image('TEST.jpg', input_details)

    timer2 = time.perf_counter()
    print(f"Preprocessing Time: {(timer2 - timer1) * 1000} ms")
    # Set the image tensor as the input to the model
    models[i].set_tensor(input_details[0]['index'], img)
    
    # Run inference
    models[i].invoke()
    # Get the output tensor for the current model
    output = models[i].get_tensor(output_details[0]['index'])
    # Get the inference time
    latency.append(time.time() - start_time)

for i in range(11):
    print(f'{model_name[i]}: {latency[i] * 1000 } ms')



