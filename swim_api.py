import socket
import time


def get_system_state():
    state = []#rt, tp, rate, dimer, server
    host = "127.0.0.1"
    port = 4242
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    conn = s.connect((host, port))

    s.sendall(b'get_basic_rt')
    data = s.recv(1024)
    response_time_base = str(data.decode("utf-8"))

    s.sendall(b'get_opt_rt')
    data = s.recv(1024)
    response_time_opt = str(data.decode("utf-8"))

    response_time = (float(response_time_base) + float(response_time_opt)) / 2.0
    print (" Response time", response_time)
    state.append(response_time/0.05)



    s.sendall(b'get_basic_throughput')
    data = s.recv(1024)
    throughput_base = str(data.decode("utf-8"))

    s.sendall(b'get_opt_throughput')
    data = s.recv(1024)
    throughput_opt = str(data.decode("utf-8"))

    throughput = (float(throughput_base) + float(throughput_opt)) / 2.0
    print (" Throughput", throughput)
    state.append(throughput/6)



    s.sendall(b'get_arrival_rate')
    data = s.recv(1024)
    arrival_rate = float(str(data.decode("utf-8")))
    print(" Rate", arrival_rate)
    state.append(arrival_rate/13)


    s.sendall(b'get_dimmer')
    data = s.recv(1024)
    dimmer_value = float(str(data.decode("utf-8")))
    print (" current dimmer ", str(dimmer_value))
    state.append(dimmer_value)


    s.sendall(b'get_active_servers')
    data = s.recv(1024)
    server_in_use = int(str(data.decode("utf-8")))
    print(" active_server", server_in_use)
    state.append(server_in_use/3)


    return state

def perform_action(state, action):
    print(action)
    dimmer = state[-2]
    server = state[-1] * 3
    print(server)
    host = "127.0.0.1"
    port = 4242
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    conn = s.connect((host, port))
    s.sendall(b'get_max_servers')
    data = s.recv(1024)
    max_server = int(str(data.decode("utf-8")))
    done = False

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    conn = s.connect((host, port))
    if action[0]=="add":
        if server == max_server:
            done = True
        else:
            s.sendall(b'add_server')
            data = s.recv(1024)
    elif action[0]=="remove":
        if server < 2:
            done = True
        else:
            s.sendall(b'remove_server')
            data = s.recv(1024)
    print("at 1")
    time.sleep(2)


    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    conn = s.connect((host, port))
    print("at 2")
    if action[1] > 0:
        if float(dimmer) + 0.5 <= 1:
            s.sendall(b'set_dimmer ' + str.encode(str(float(dimmer) + 0.5)))
            data = s.recv(1024)

        else:
            done = True
    elif action[1] < 0:
        if float(dimmer) - 0.5 >= 0:
            s.sendall(b'set_dimmer ' + str.encode(str(float(dimmer) - 0.5)))
            data = s.recv(1024)
        else:
            done = True

    time.sleep(2)
    return done


