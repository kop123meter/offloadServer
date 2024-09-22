import tensorflow as tf
import timeit
 
 
# def cpu_run():
#     with tf.device('/cpu:0'):
#         cpu_a = tf.random.normal([10000, 1000])
#         cpu_b = tf.random.normal([1000, 2000])
#         c = tf.matmul(cpu_a, cpu_b)
#     return c
 
 
# def gpu_run():
 
#     with tf.device('/gpu:0'):
#         gpu_a = tf.random.normal([10000, 1000])
#         gpu_b = tf.random.normal([1000, 2000])
#         c = tf.matmul(gpu_a, gpu_b)
#     return c
 
 
# import numpy as np
# import tensorflow as tf
# import cv2

# # 加载量化模型
# interpreter = tf.lite.Interpreter(model_path='mobilenetDetv1.tflite',num_threads=30,experimental_delegates=[tf.lite.experimental.load_delegate('tensorflowlite_gpu.dll')])
# interpreter.allocate_tensors()

# # 获取输入和输出张量的信息
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# print("Input dtype:", input_details[0]['dtype'])
# print("Output dtype:", output_details[0]['dtype'])

# # 加载和预处理图像（假设输入图像的范围是 [0, 255]）
# img = cv2.imread('TEST.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.resize(img, (300, 300))

# # 输入数据类型转换为 uint8
# input_data = np.expand_dims(img, axis=0).astype(np.uint8)

# # 设置输入张量
# interpreter.set_tensor(input_details[0]['index'], input_data)


# start_time = timeit.default_timer()
# # 执行推理
# interpreter.invoke()

# end_time = timeit.default_timer()
# print('推理时间:', (end_time - start_time) * 1000)
# # 获取输出数据
# output_data = interpreter.get_tensor(output_details[0]['index'])
# print('推理结果:', output_data)
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# import time

# # 加载 MobileNetV1 模型
# model = tf.keras.applications.MobileNet(weights='imagenet')

# # 加载并预处理图像
# img_path = 'TEST.jpg'  # 替换为实际图像路径
# img = image.load_img(img_path, target_size=(224, 224))
# img_array = image.img_to_array(img)
# img_array = np.expand_dims(img_array, axis=0)

# # 使用 MobileNetV1 的预处理函数将图像转换为模型输入格式
# img_array = tf.keras.applications.mobilenet.preprocess_input(img_array)

# # 进行预测
# time1 = time.time()
# predictions = model.predict(img_array)
# time2 = time.time()
# print('推理时间:', (time2 - time1) * 1000)
# # 解码预测结果，返回Top-5的预测类别
# decoded_predictions = tf.keras.applications.mobilenet.decode_predictions(predictions, top=5)

# # 输出预测结果
# for i, (imagenet_id, label, score) in enumerate(decoded_predictions[0]):
#     print(f"{i + 1}: {label} ({score * 100:.2f}%)")
try:
    # Windows: 使用 'tensorflowlite_gpu.dll'
    # Linux: 使用 'libtensorflowlite_gpu_delegate.so'
    gpu_delegate = tf.lite.experimental.load_delegate('tensorflowlite_gpu.dll')
    interpreter = tf.lite.Interpreter(model_path='./mobilenetDetv1.tflite', experimental_delegates=[gpu_delegate])
    print("Using GPU for inference.")
except Exception as e:
    print(f"Could not load GPU delegate: {e}")