import socket


def get_system_state():
    state = []#rt, tp, rate, dimer, server
    host = "127.0.0.1"
    port = 4242
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    conn = s.connect((host, port))
    s.sendall(b'get_basic_rt')
    data = s.recv(1024)
    response_time_base = str(data.decode("utf-8"))
    s.close()

    conn = s.connect((host, port))
    s.sendall(b'get_opt_rt')
    data = s.recv(1024)
    response_time_opt = str(data.decode("utf-8"))
    s.close()


    response_time = (float(response_time_base) + float(response_time_opt)) / 2.0
    print (" Response time", response_time)
    state.append(response_time)


    conn = s.connect((host, port))
    s.sendall(b'get_basic_throughput')
    data = s.recv(1024)
    throughput_base = str(data.decode("utf-8"))
    s.close()


    conn = s.connect((host, port))
    s.sendall(b'get_opt_throughput')
    data = s.recv(1024)
    throughput_opt = str(data.decode("utf-8"))
    s.close()

    throughput = (float(throughput_base) + float(throughput_opt)) / 2.0
    print (" Throughput", throughput)
    state.append(throughput)


    conn = s.connect((host, port))
    s.sendall(b'get_arrival_rate')
    data = s.recv(1024)
    arrival_rate = float(str(data.decode("utf-8")))
    state.append(arrival_rate)
    s.close()

    conn = s.connect((host, port))
    s.sendall(b'get_dimmer')
    data = s.recv(1024)
    dimmer_value = float(str(data.decode("utf-8")))
    print (" current dimmer ", str(dimmer_value))
    state.append(dimmer_value)
    s.close()

    conn = s.connect((host, port))
    s.sendall(b'get_active_servers')
    data = s.recv(1024)
    server_in_use = int(str(data.decode("utf-8")))
    state.append(server_in_use)
    s.close()

    return state

def perform_action(state, action):
    print(action)
    dimmer = state[-2]
    server = state[-1]
    host = "127.0.0.1"
    port = 4242
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    conn = s.connect((host, port))
    s.sendall(b'get_max_servers')
    data = s.recv(1024)
    s.close()
    max_server = int(str(data.decode("utf-8")))
    done = False
    if action[0]=="add":
        if server== max_server:
            done = True
        else:
            conn = s.connect((host, port))
            s.sendall(b'add_server')
            data = s.recv(1024)
            s.close()
            done = False
    elif action[0]=="remove":
        if server == 0:
            done = True
        else:
            conn = s.connect((host, port))
            s.sendall(b'remove_server')
            data = s.recv(1024)
            s.close()

    if action[1] > 0:
        if float(dimmer) + 0.25 <= 1:
            print(str.encode(str(float(dimmer) + 0.25)))
            conn = s.connect((host, port))
            s.sendall(b'set_dimmer ' + str.encode(str(float(dimmer) + 0.25)))
            data = s.recv(1024)
            s.close()
        else:
            done = True
    elif action[1] < 0:
        if float(dimmer) - 0.25 >= 0:
            print(str.encode(str(float(dimmer) - 0.25)))
            conn = s.connect((host, port))
            s.sendall(b'set_dimmer' + str.encode(str(float(dimmer) - 0.25)))
            data = s.recv(1024)
            s.close()
        else:
            done = True
    return done


