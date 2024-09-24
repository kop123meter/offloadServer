import socket
import threading
import time
from datetime import datetime
# from bayesian_client_new import BayesianOptimizationRunner
import os
import json
import numpy as np # type: ignore
from GPyOpt.methods import BayesianOptimization # type: ignore

import colorama
from colorama import Fore, Style

class TwoClientsServer:

    MAX_ITER = 15 # 15
    EXPLORATION_n = 5
    NUM_TASKS = 5
    PORT = 1909  


    def __init__(self):
        self.iterations = self.EXPLORATION_n + self.MAX_ITER + 1
        current_time = datetime.now()
        self.output_dir = os.path.join("./experiments", current_time.strftime("%Y-%m-%d %H-%M-%S"))
        os.makedirs(self.output_dir, exist_ok=True)
        self.stop_event = threading.Event()
        self.server_socket = None
        self.client_threads = []
        
        
    def run(self):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('0.0.0.0', self.PORT))
            self.server_socket.listen(5)

           
            while not self.stop_event.is_set():
                print("SERVER: Waiting for the Python client...")

                python_client = PythonClient(num_tasks=self.NUM_TASKS,
                                         max_iter=self.MAX_ITER,
                                         host='192.168.1.3',
                                         port=self.PORT,
                                         output_dir=self.output_dir,
                                         stop_event=self.stop_event)
            
                self.client_threads.append(python_client)
                python_client.start()
                client_socket2, addr2 = self.server_socket.accept()
                print(f"SERVER: Python client connected: {addr2[0]} \n        now Waiting for Android client...")
                
                print(python_client.is_alive)
                client_socket1, addr1 = self.server_socket.accept()
                print(f"SERVER: Android client connected: {addr1[0]}")

                python_handler = ClientHandler(client_socket2, client_socket1, is_android=False, output_dir=self.output_dir, stop_event=self.stop_event)
                python_handler.start()
                self.client_threads.append(python_handler)

                android_handler = ClientHandler(client_socket1, client_socket2, is_android=True, output_dir=self.output_dir, stop_event=self.stop_event)
                android_handler.start()
                self.client_threads.append(android_handler)
            
                while android_handler.is_done == False:
                    time.sleep(1)

        except Exception as e:
            print(f"SERVER: Exception: {e}")
       

    def stop(self):
        self.stop_event.set()
        if self.server_socket:
            self.server_socket.close()
        for thread in self.client_threads:
            thread.join()
        
        red_printer = printer("red")
        red_printer.print("SERVER: All sockets closed and threads terminated.")


class PythonClient(threading.Thread):
    def __init__(self, num_tasks, max_iter, host, port, output_dir, stop_event):
        super().__init__()
        self.runner = BayesianOptimizationRunner(num_tasks, max_iter, host, port, output_dir)
        self.stop_event = stop_event

    def run(self):
        time.sleep(1)  # wait for the server to be ready to accept 
        self.runner.main()


class ClientHandler(threading.Thread):
    def __init__(self, client_socket, other_client_socket, is_android=False, output_dir=None, stop_event=None):
        super().__init__()
        self.client_socket = client_socket
        self.other_client_socket = other_client_socket
        self.is_android = is_android
        self.output_dir = output_dir
        self.stop_event = stop_event
        self.is_done = False
        
        if is_android:
            color_printer = printer('green')
        else:
            color_printer = printer('yellow')
        self.print = color_printer.print
    
        red_printer = printer('red')
        self.printRed = red_printer.print
                        
    def run(self):
        try:
            if self.is_android:
                self.android_client()
            else:
                self.python_client()
        except Exception as e:
            self.printRed(f"SERVER: Exception in {'Android' if self.is_android else 'Python'}ClientHandler: {e}")
        finally:
            self.print(f"SERVER: {'Android' if self.is_android else 'Python'} handler is Done")
            self.client_socket.close()
            self.is_done = True
            self.other_client_socket.close()
            
            
    def python_client(self):
        while not self.stop_event.is_set():
            input_data = self.client_socket.recv(1024).decode().strip()
            if not input_data:
                break
            self.print(f"SERVER: Received from Python: {input_data}")
            
            if "status/" in input_data:
                pass
            else:
                self.other_client_socket.sendall((input_data + '\n').encode())
            
    def android_client(self):
        while not self.stop_event.is_set():
            input_data = self.client_socket.recv(1024).decode().strip()
            if not input_data:
                self.printRed("SERVER: android client 'not input_data'")
                break
            self.print(f"SERVER: Received from Android: {input_data}")
            
            if "thermal/" in input_data:
                filename = 'thermal_data.csv'
                filepath = os.path.join('./', self.output_dir, filename)
                with open(filepath, "a") as out:
                    out.write(input_data[len('thermal/'):] + '\n')
                    
            elif "delegate/" in input_data:
                self.other_client_socket.sendall((input_data + '\n').encode()) 
            else:
                self.other_client_socket.sendall((input_data + '\n').encode())


class BayesianOptimizationRunner:
    def __init__(self, num_tasks, max_iter, host, port, output_dir=None):
        self.NUM_TASKS = num_tasks
        self.MAX_ITER = max_iter
        self.HOST = host
        self.PORT = port
        self.counter = 0
        self.output_dir = output_dir
        self.client_socket = None

        color_printer = printer('blue')
        self.print = color_printer.print



    # def quick_test(self, X):
    #     for i in range(3):
    #         print(f"OPTIMIZER:  x{i+1}: {X[0][i]}")
    #         print(f"OPTIMIZER:  4x{i+1}: {X[0][i] * 4}")
    #         print(f"OPTIMIZER:  ROUND{i+1}: {round(X[0][i] * 4)}")
    #     X_list = X.tolist()
    #     print(f"OPTIMIZER:  input: {X_list}")
    #     return X[0][2]

    def objective(self, X):
        self.counter += 1

        current_time = datetime.now()
        timecur = current_time.strftime("%H:%M:%S.%f")

        translatedU = self.translate_delegate_usage(X[0][:3]) + [X[0][3]]
        X_list = [translatedU]
        nontranslated_X_list = X.tolist()

        data = {"python_client": X_list[0]}
        json_data = json.dumps(data)

        self.client(json_data + '\n')

        self.print(f"OPTIMIZER:  {self.counter}: {nontranslated_X_list[0]} is sent, waiting for the reward ...")
        self.print(f"OPTIMIZER:  delegate is translated to {X_list[0]}")

        received_data = ''
        while 'reward/' not in received_data:
            received_data = self.client()

        self.client("status/reward_received\n")
        reward = float(received_data.split('reward/')[1])
        self.print(f"OPTIMIZER:  Obj -> Received reward : {reward}")

        with open(f"{self.output_dir}/Bayesian_OUTPUT.csv", "a") as out:
            out.write(f"{timecur},{nontranslated_X_list[0]},{X_list[0]},{reward}\n")

        return -reward

    def translate_delegate_usage(self, x):
        scaled_values = (x * self.NUM_TASKS).astype(int)
        remainder = self.NUM_TASKS - np.sum(scaled_values)
        sorted_indices = np.argsort(x)[::-1]

        for i in range(remainder):
            scaled_values[sorted_indices[i]] += 1

        return scaled_values.tolist()

    class JavaRewardBayesianOptimization(BayesianOptimization):
        def __init__(self, domain, constraints, objective):
            super().__init__(
                f=objective,
                domain=domain,
                constraints=constraints,
                acquisition_type='EI'
            )

    def run(self):
        space = [
            {'name': 'var_1', 'type': 'continuous', 'domain': [0, 1]},
            {'name': 'var_2', 'type': 'continuous', 'domain': [0, 1]},
            {'name': 'var_3', 'type': 'continuous', 'domain': [0, 1]},
            {'name': 'var_4', 'type': 'continuous', 'domain': [0.2, 1]}
        ]
        
        constraints = [
                    {'name': 'constr_1', 'constraint': 'x[:,0] + x[:,1] + x[:,2] - 1'},
                    {'name': 'constr_2', 'constraint': '-x[:,0] - x[:,1] - x[:,2] + 0.999'}
                    ]
        
        current_time = datetime.now()
        if self.output_dir is None:
            self.output_dir = current_time.strftime("%H_%M")
            os.makedirs(self.output_dir, exist_ok=True)

        self.print("OPTIMIZER: waiting for Android activation msg...")
        received_data = self.client()
        self.print("OPTIMIZER:  first Received: " + str(received_data))

        if 'activate' in received_data:
            act_msg, num_tasks = received_data.split(':',1)
            num_tasks = int(float(num_tasks))
            print(f"OPTIMIZER:  num_tasks: {num_tasks}")
            self.setNumTasks(num_tasks)
            self.print(f"OPTIMIZER:  num_tasks: {self.getNumTasks()}")
            time.sleep(10)
            with open(f"{self.output_dir}/Bayesian_OUTPUT.csv", "a") as out:
                out.write("time,,cu,,gu,,nu,,tris,,,ct,,gt,,nt,,ttris,,reward\n")

            with open(f"{self.output_dir}/Candidate_time.csv", "a") as out:
                out.write("time\n")

            problem = self.JavaRewardBayesianOptimization(
                domain=space,
                constraints=constraints,
                objective=self.objective
            )

            self.print("OPTIMIZER:  Initial exploration is finished!")
            problem.run_optimization(max_iter=self.MAX_ITER)

            best_input = problem.x_opt
            best_reward = problem.fx_opt

            self.print(f"OPTIMIZER:  Best input combination for iteration count #{self.MAX_ITER}: {best_input}")
            self.print(f"OPTIMIZER:  Best reward: {best_reward}")

            with open(f"{self.output_dir}/Bayesian_OUTPUT.csv", "a") as out:
                out.write(f"{best_input},{best_reward}\n")

            translatedU = self.translate_delegate_usage(best_input[:3]) + [best_input[3], best_reward * -1]
            X_list2 = [translatedU]

            data = {"python_client": X_list2[0]}
            json_data = json.dumps(data)
            self.client(json_data + '\n')

            self.print(f"OPTIMIZER:  delegate is translated to {X_list2[0]}")
            received_data = self.client()
            reward = float(received_data.split('reward/')[1])
            self.print(f"OPTIMIZER:  Obj -> Received reward : {reward}")

            with open(f"{self.output_dir}/Bayesian_OUTPUT.csv", "a") as out:
                out.write(f"{best_input[0]},{X_list2[0]}\n")

            # problem.plot_convergence(filename=f"{self.output_dir}/convergence_plot")
            problem.save_report(f"{self.output_dir}/report.txt")

    def client(self, msg_to_send=None):
        if msg_to_send is not None:
            self.client_socket.sendall(msg_to_send.encode())
            return None
        else:
            received_data = self.client_socket.recv(1024).decode()
            return received_data

    def main(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as self.client_socket:
            self.client_socket.connect((self.HOST, self.PORT))
            self.run()
            self.print(f"OPTIMIZER:  Done!")
    
    def setNumTasks(self, num_tasks):
        self.NUM_TASKS = num_tasks
    
    def getNumTasks(self):
        return self.NUM_TASKS


class printer:
    def __init__(self,color='black') -> None:
        self.color = color
        self.color_dict = {
            "red": Fore.RED,
            "green": Fore.GREEN,
            "yellow": Fore.YELLOW,
            "blue": Fore.BLUE,
            "magenta": Fore.MAGENTA,
            "cyan": Fore.CYAN,
            "white": Fore.WHITE,
            "black": Fore.LIGHTBLACK_EX
        }

    def print(self, text):

        color = self.color
        if color.lower() in self.color_dict:
            print(self.color_dict[color.lower()] + text + Style.RESET_ALL)
        else:
            print("Color not recognized. Available colors: red, green, yellow, blue, magenta, cyan, white, black")


if __name__ == "__main__":
    colorama.init(autoreset=True)
    server = TwoClientsServer()
    
    try:
        server.run()
    except KeyboardInterrupt:
        server.stop()
        red_printer = printer("red")
        red_printer.print("SERVER: Shutting down due to keyboard interrupt.")