import socket
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from threading import Thread

# Initialize data structures
rewards = []
tris_counts = []
AI_dict = {}  # {model_index: [latencies]}
AI_info_dict = {}  # {model_index: device_index}

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
                if data_type == "reward":
                    reward, current_tris = received_data.split(",",1)
                    rewards.append(float(reward))
                    tris_counts.append(float(current_tris))
                elif data_type == "time":
                    time, device_index, model_index = received_data.split(",",2)
                    print(f"time: {time} device: {device_index} model_index :{model_index}")
                    if model_index not in AI_dict:
                        AI_dict[model_index] = []
                        AI_info_dict[model_index] = device_index
                    AI_dict[model_index].append(float(time))

            except ValueError:
                print("Invalid data received")

        client_socket.close()
        print(f"Connection closed with {addr}")

# Update the plot dynamically
def update_plot(i):
    ax1.clear()
    ax2.clear()
    ax3.clear()

    # Plot rewards and triangle counts
    ax1.set_title('Reward')
    ax1.plot(rewards, label='Reward', color='r')
    ax1.legend(loc='upper left')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Reward')

    ax2.set_title('Triangle Count')
    ax2.plot(tris_counts, label='Triangle Count', color='b')
    ax2.legend(loc='upper left')
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Triangle Count')

    # Plot response times for each AI model index
    ax3.set_title('Response Times by AI Model')
    for model_index, times in AI_dict.items():
        color = 'g' if AI_info_dict[model_index] == '3' else 'y'  # Green if device index is 3, magenta otherwise
        print(f"{model_index}:::::::::::::::{AI_info_dict[model_index]} ::::::::::    {color}")
        
        ax3.plot(times, label=f'Model {model_index} (Device {AI_info_dict[model_index]})', color=color)
    ax3.legend(loc='upper left')
    ax3.set_xlabel('Time Points')
    ax3.set_ylabel('Response Time (ms)')

    plt.tight_layout()

if __name__ == '__main__':
    # Start the server in a separate thread
    server_thread = Thread(target=socket_server)
    server_thread.start()

    # Set up the figure and axis
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

    # Start animation for dynamic plotting
    ani = animation.FuncAnimation(fig, update_plot, interval=1000)
    plt.show()

    # Ensure the server thread ends gracefully
    server_thread.join()
