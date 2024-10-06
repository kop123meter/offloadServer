# Description: This file contains the server code that receives data from the client, runs inference on the data using the TFLite models, and sends the output back to the client.
# Author: Ze Li
# Date: 2024-09-01

# We need to modify server address in bitmapcollector.kt 

# Import necessary libraries
import socket
import tensorflow as tf
import numpy as np
import base64 # To decode base64 images from client
import io # To convert bytes to image   
from PIL import Image, ImageFile 
import time
import threading
import cv2

from tensorflow.keras.mixed_precision import experimental as mixed_precision


HOST = '0.0.0.0'
PORT = 4545

ImageFile.LOAD_TRUNCATED_IMAGES = True

MAX_NUM_THREADS = 10



#Using GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_visible_devices(devices=gpu,device_type='GPU')
            tf.config.experimental.set_memory_growth(gpu, True)
            print(tf.test.is_gpu_available())
            print("GPU Setting up!")
    except RuntimeError as e:
        print(e)


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



def load_models():
    # Load the TFLite model and allocate tensors.
    return {
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

    
    #print("TensorFlow Lite supports set_num_threads:", hasattr(interpreter, 'set_num_threads'))
    # input_details = interpreter.get_input_details()
    
    
    # print("*"*50)


def preprocess_image(image, model_input_details):
    """
    Preprocesses the image according to the model's input requirements.

    Args:
    image (PIL.Image): The input image.
    model_input_details (dict): A dictionary containing the model's input details.

    Returns:
    numpy.ndarray: The preprocessed image tensor.
    """
    # Extract input details for the first model (assuming it's the one you're using)
    input_dict = model_input_details[0]
    input_shape = input_dict['shape']
    
    # Resize image to match the input dimensions required by the model
    image = image.resize((input_shape[2], input_shape[1]), Image.Resampling.LANCZOS)

    # Convert image to numpy array and adjust type accordingly
    image = np.array(image)
    if input_dict['dtype'] == np.float32:
        image = image.astype(np.float32)
        # Normalize image if there are no quantization parameters, or adjust according to them
        if 'quantization_parameters' in input_dict and input_dict['quantization_parameters']['scales'].size > 0:
            scale = input_dict['quantization_parameters']['scales'][0]
            zero_point = input_dict['quantization_parameters']['zero_points'][0]
            image = (image - zero_point) * scale
        else:
            image = image / 255.0
    else:
        image = image.astype(np.uint8)
        if 'quantization_parameters' in input_dict and input_dict['quantization_parameters']['scales'].size > 0:
            scale = input_dict['quantization_parameters']['scales'][0]
            zero_point = input_dict['quantization_parameters']['zero_points'][0]
            image = ((image - zero_point) * scale).astype(np.uint8)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image

def preprocess_image_cv2(image, model_input_details):
    """
    Preprocesses the image according to the model's input requirements using OpenCV for speed.

    Args:
    image (numpy.ndarray): The input image loaded using OpenCV (BGR format).
    model_input_details (dict): A dictionary containing the model's input details.

    Returns:
    numpy.ndarray: The preprocessed image tensor.
    """
    input_dict = model_input_details[0]
    input_shape = input_dict['shape']

    # Resize the image if necessary to match model input dimensions
    if image.shape[:2] != (input_shape[1], input_shape[2]):
        image = cv2.resize(image, (input_shape[2], input_shape[1]), interpolation=cv2.INTER_LANCZOS4)

    # Convert BGR to RGB since OpenCV loads images in BGR format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # If the model expects float32 input, convert and normalize the image
    if input_dict['dtype'] == np.float32:
        image = image.astype(np.float32)
        if 'quantization_parameters' in input_dict and input_dict['quantization_parameters']['scales'].size > 0:
            scale = input_dict['quantization_parameters']['scales'][0]
            zero_point = input_dict['quantization_parameters']['zero_points'][0]
            # Apply quantization scaling
            image = (image - zero_point) * scale
        else:
            # Normalize the image to [0, 1] if no quantization parameters
            image /= 255.0
    else:
        # If the model expects uint8, make sure the image is in uint8 format
        image = image.astype(np.uint8)
        if 'quantization_parameters' in input_dict and input_dict['quantization_parameters']['scales'].size > 0:
            scale = input_dict['quantization_parameters']['scales'][0]
            zero_point = input_dict['quantization_parameters']['zero_points'][0]
            # Apply quantization scaling
            image = ((image - zero_point) * scale).astype(np.uint8)

    # Add batch dimension (1, H, W, C)
    image = np.expand_dims(image, axis=0)

    return image

# Handle message from client
def handle_client(client_socket):
    '''
    This function handles the client connection and processes the data received from the client.Data
    Data format: model_index:data_length:image_data
    '''
    models = load_models()
    for interpreter in models.values():
        interpreter.allocate_tensors()
    print("Modle Load Successful!")
    try:
        # Receive data from the client
        data = b""
        while True:
            packet = client_socket.recv(4096)
            if not packet:
                break
            # Check if the client has finished sending data
            if b"finish" in packet:
                data += packet.replace(b"finish", b"")
                break
            data += packet
        data = data.decode('utf-8')
        print("Received data from client")

        startProcessingTime = time.perf_counter()

        client_socket.send("RECEIVED".encode('utf-8'))
        # Split the data into model index and image data
        model_index_data, data_length ,image_data = data.split(":", 2)
        print("data_length:", data_length)  # This is the length of the image data
        print("actual length:", len(image_data))  # This is the model index
       

        

        # Convert the model index to an integer
        model_index = int(model_index_data)

        # Validate model index
        if model_index not in models:
            print(f"Invalid model index: {model_index}")
            client_socket.send("Invalid model index".encode('utf-8'))
            return
        
       
        
      
        

        # Get the appropriate model interpreter
        interpreter = models[model_index]
        current_model_name = model_name[model_index]
        print("Current Model Name: ", current_model_name)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        testStartTimer = time.perf_counter()

        if isinstance(image_data, bytes):
            image_data = image_data.decode('utf-8')
        
        image_data = image_data.strip()
        print("Base64 string length before padding:", len(image_data))
        
        # Check if the model expects a base64 image
        missing_padding = len(image_data) % 4
        if missing_padding:
            image_data += '=' * (4 - missing_padding)

        print("Base64 string length after padding:", len(image_data))
        print("Remainder when divided by 4 (should be 0):", len(image_data) % 4)
      

        try:
            image_data = base64.b64decode(image_data)
        except Exception as e:
            print(f"Error decoding image: {e}")
            client_socket.send(f"Error decoding image: {e}".encode('utf-8'))
            return

        # Check if the image data is None
        if image_data is None:
            print("Decoded image data is None.")
            client_socket.send("Decoded image data is None.".encode('utf-8'))
            return

        # Open the image
        try:
            image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            print(f"Error opening image: {e}")
            client_socket.send(f"Error opening image: {e}".encode('utf-8'))
            return

        
        # Preprocess the image
        testEndTimer = time.perf_counter() 
        print(f"Image preprocessing time for decode: {(testEndTimer - testStartTimer) * 1000 } ms")

        timer1 = time.perf_counter()
        input_data = preprocess_image(image, input_details)
        #input_data = preprocess_image_cv2(np.array(image), input_details)
        timer2 = time.perf_counter()
        print(f"Image preprocessing time for preprocessing data: {(timer2 - timer1) * 1000} ms")
        
        # Run inference
        
        timer3 = time.perf_counter()
    
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        timer4 = time.perf_counter()
        print(f"Model processing time: {(timer4 - timer3) * 1000} ms")
        endProcessingTime = time.perf_counter()

        # Calculate processing latency
        processing_latency = (endProcessingTime - startProcessingTime) * 1000  # Convert to milliseconds

        # Get the result from the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        result = output_data.tolist()

        # Send a confirmation message that the server has received and processed the request
        #sclient_socket.send("Processing complete. Sending result...".encode('utf-8'))

        # Send the processing latency and result back to the client

        #TODO
        # Current Best Network Latency is still high 
        # Maybe we can compress result data
        result_str = f"{processing_latency}:{result}"
        client_socket.send(result_str.encode('utf-8'))

        print(f"Total processing latency: {processing_latency} ms")

    except Exception as e:
        print(f"Error handling client: {e}")
    finally:
        client_socket.close()







def startServer():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(5)  # allow 5 connections to be queued
    print("Server is running")
    print("Waiting for client to send data")
    client_count =0

    while True:
        conn, addr = s.accept()
        print('Connected by', addr)
        # handle_client(conn)
        print("*" * 50, client_count, "*" *50)
        # Start a new thread to handle the client
        client_thread = threading.Thread(target=handle_client, args=(conn,))
        client_thread.start()
        client_count = client_count + 1



if __name__ == "__main__":
   startServer()
