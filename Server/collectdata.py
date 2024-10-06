import socket
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from threading import Thread
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Initialize data structures
rewards = []
tris_counts = []
# AI_dict = {}  # {model_index: [latencies]}
# AI_info_dict = {}  # {model_index: device_index}
total_latencies = []
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']  # Added white color for visibility on many backgrounds
devs = ["CPU", "GPU", "NNAPI", "SERVER"]
devs = ["CPU", "GPU", "NNAPI", "SERVER"]

folder_name = 'exp_output_csv'

if not os.path.exists(folder_name):
        os.makedirs(folder_name)



# Server to receive data from clients
def socket_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', 6767))
    server_socket.listen(5)
    print("Server listening on port 6767")

    while True:
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr}")

        while True:
            data = client_socket.recv(1024).decode()
            if not data:
                break

            try:
                data_type, received_data = data.split('/', 1)
                # if data_type == "reward":
                #     reward, current_tris = received_data.split(",",1)
                #     rewards.append(float(reward))
                #     tris_counts.append(float(current_tris))
                # elif data_type == "time":
                #     time, device_index, model_index = received_data.split(",",2)
                #     device_index = int(device_index)  # Ensure device index is an integer
                #     if model_index not in AI_dict:
                #         AI_dict[model_index] = []
                #     AI_info_dict[model_index] = device_index
                #     AI_dict[model_index].append(float(time))
                
                if data_type == "msg":
                    reward, current_tris, total_time = received_data.split(",",2)
                    rewards.append(float(reward))
                    tris_counts.append(float(current_tris))
                    total_latencies.append(float(total_time))
                    print(f"Reward: {reward}, Triangle Count: {current_tris}, Total Latency: {total_time}")
                elif data_type == "save":
                    data = {
                        'Reward': rewards,
                        'Triangle Count': tris_counts,
                        'Total Latency': total_latencies
                    }
                    df = pd.DataFrame(data)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    file_name = f"{timestamp}_offloadOutput.csv"
                    file_path = os.path.join(folder_name, file_name)
                    df.to_csv(file_path, index=False)
                    print(f"Data saved to {file_path}")
                    


            
                
            except ValueError:
                print("Invalid data received")

        client_socket.close()
        print(f"Connection closed with {addr}")

# Update the plot dynamically
def update_plot(i):
    ax1.clear()
    ax2.clear()
    ax3.clear()

    ax1.set_title('Reward')
    ax1.plot(rewards, label='Reward', color='r')
    ax1.legend(loc='upper left')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Reward')
    y_min, y_max = ax1.get_ylim()
    ax1.set_yticks(np.arange(y_min, y_max + 0.5, 0.5))

    ax2.set_title('Quality')
    ax2.plot(tris_counts, label='Quality', color='b')
    ax2.legend(loc='upper left')
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Triangle Count')

    ax3.set_title('Average Latency')
    ax3.plot(total_latencies, label='Average Latency', color='g')
    ax3.legend(loc='upper left')
    ax3.set_xlabel('Time Points')
    ax3.set_ylabel('Average Time (ms)')
    y_min, y_max = ax3.get_ylim()
    # ax3.set_yticks(np.arange(y_min, y_max + 200, 200))

    plt.tight_layout()

if __name__ == '__main__':
    # Start the server in a separate thread
    server_thread = Thread(target=socket_server)
    server_thread.start()

    # Set up the figure and axis
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

    # Start animation for dynamic plotting
    ani = animation.FuncAnimation(fig, update_plot, interval=10)
    plt.show()

    # Ensure the server thread ends gracefully
    server_thread.join()