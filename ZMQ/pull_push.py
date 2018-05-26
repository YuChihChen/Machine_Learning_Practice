import zmq
import time
import communicationtools as ct
from multiprocessing import Process


def server(name_, port_list_):
    worker = ct.Communication(name_, zmq.PUSH, 'bind', port_list_)
    i = 0
    while True:
        worker.send_message("{} send the results_{}".format(name_, i))
        i += 1
        if i >= 5:
            break


def client(port_list_):
    asker = ct.Communication('client', zmq.PULL, 'connect', port_list_, ['127.0.0.1'])
    time.sleep(20)
    while True:
        asker.receive_message()
        time.sleep(1)


def main():
    ps1 = Process(target=server, args=('server_1', [5566],))
    ps2 = Process(target=server, args=('server_2', [5588],))
    pc = Process(target=client, args=([5566, 5588], ))
    ps1.start()
    ps2.start()
    pc.start()
    ps1.join()
    ps2.join()
    pc.join()


if __name__ == '__main__':
    main()