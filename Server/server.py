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

HOST = '0.0.0.0'
PORT = 4545

ImageFile.LOAD_TRUNCATED_IMAGES = True




#Using GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)



# Load the TFLite model and allocate tensors.
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

for interpreter in models.values():
    interpreter.allocate_tensors()


# Get input and output tensors.
def preprocess_image(image, input_shape):
    image = image.resize((input_shape[2], input_shape[1]))
    image = np.array(image).astype(np.float32)
    image = np.expand_dims(image, axis=0)  # expand dimensions to fit expected input shape
    return image


# Handle message from client
def handle_client(client_socket):
    try:
        # Receive data from the client
        data = b""
        while True:
            packet = client_socket.recv(4096)
            if not packet:
                break
            # Check if the client has finished sending data
            if b"finish" in packet:
                break
            data += packet
        data = data.decode('utf-8')
        print("Received data from client")

        startProcessingTime = time.perf_counter()

        client_socket.send("RECEIVED".encode('utf-8'))
        # Split the data into model index and image data
        model_index_data, image_data = data.split(":", 1)

        

        # Convert the model index to an integer
        model_index = int(model_index_data)

        # Validate model index
        if model_index not in models:
            print(f"Invalid model index: {model_index}")
            client_socket.send("Invalid model index".encode('utf-8'))
            return
        

        # Get the appropriate model interpreter
        interpreter = models[model_index]
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Check if the model expects a base64 image
        missing_padding = len(image_data) % 4
        if missing_padding:
            image_data += '=' * (4 - missing_padding)

        print(len(image_data) % 4) 
        # Decode the base64 image data
        image_data = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_data))
        input_data = preprocess_image(image, input_details[0]['shape'])

        # Run inference
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        endProcessingTime = time.perf_counter()

        # Calculate processing latency
        processing_latency = (endProcessingTime - startProcessingTime) * 1000  # Convert to milliseconds

        # Get the result from the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        result = output_data.tolist()

        # Send a confirmation message that the server has received and processed the request
        #sclient_socket.send("Processing complete. Sending result...".encode('utf-8'))

        # Send the processing latency and result back to the client
        result_str = f"{processing_latency}:{result}"
        client_socket.send(result_str.encode('utf-8'))

        print(f"Model processing latency: {processing_latency} ms")

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

    while True:
        conn, addr = s.accept()
        print('Connected by', addr)
        handle_client(conn)

        # Start a new thread to handle the client
        # client_thread = threading.Thread(target=handle_client, args=(conn,))
        # client_thread.start()



if __name__ == "__main__":
    startServer()