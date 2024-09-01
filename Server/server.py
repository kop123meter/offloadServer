# Description: This file contains the server code that receives data from the client, runs inference on the data using the TFLite models, and sends the output back to the client.
# Author: Ze Li
# Date: 2024-09-01

# Import necessary libraries
import socket
import tensorflow as tf
import numpy as np
import base64 # To decode base64 images from client
import io # To convert bytes to image   
from PIL import Image


HOST = '0.0.0.0'
PORT = 4545




#Using GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)



# Load the TFLite model and allocate tensors.
models = {"deeplabv3":tf.lite.Interpreter(model_path=".\deeplabv3.tflite"),
          "mnist":tf.lite.Interpreter(model_path=".\mnist.tflite")}

for interpreter in models.values():
    interpreter.allocate_tensors()


# Get input and output tensors.
def preprocess_image(image, input_shape):
    image = image.resize((input_shape[2], input_shape[1]))
    image = np.array(image).astype(np.float32)
    image = np.expand_dims(image, axis=0)  # expand dimensions to fit expected input shape
    return image

# Run inference
def run_inference(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

# Handle message from client
def handle_message(client_socket):
    try:
        data_buffer = client_socket.recv(1024 * 1024) # 1MB buffer to receive data
        if not data_buffer:
            return
        
        print("Received data from client")
        print(f"Data: {data_buffer}")
        # Get model name and image from message
        # The message format is: {model_name}::{image as bytes}
        data_str = data_buffer.decode("utf-8")
        model_name, base64_images = data_str.split("::")

        # Make sure the model name is valid
        if model_name not in models:
            print(f"Model {model_name} not found")
            return
        
        # Get the interpreter and image data
        interpreter = models[model_name]
        # Decode the base64 image
        image_data = base64.b64decode(base64_images)
        image = Image.open(io.BytesIO(image_data)) # Convert bytes to image
        # Preprocess the image
        input_shape = interpreter.get_input_details()[0]['shape']
        image = preprocess_image(image, input_shape)
        # Run inference
        output = run_inference(interpreter, image).tolist()
        print(f"Output: {output}")
        # Send the output back to the client
        client_socket.send("ACK".encode("utf-8"))   # Send acknowledgment to client
        client_socket.sendall(str(output).encode("utf-8")) # Send the output back to the client

    except:
        print("Error receiving data")
        return
    finally:
        client_socket.close()


def startServer():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(1)
    conn, addr = s.accept()
    print('Connected by', addr)

    print("Server is running")
    print("Waiting for client to send data")
    while True:
        handle_message(conn)


if __name__ == "__main__":
    startServer()