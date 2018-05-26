import zmq
import time
import communicationtools as ct
from multiprocessing import Process


def server(port_list_):
    worker = ct.Communication('worker', zmq.REP, 'bind', port_list_)
    i = 0
    while True:
        worker.receive_message()
        worker.send_message("I got the task_{}".format(i))
        time.sleep(30)
        i += 1


def client(port_list_):
    asker = ct.Communication('client', zmq.REQ, 'connect', port_list_, ['127.0.0.1'])
    i = 0
    while True:
        asker.send_message("Please do the task_{}".format(i))
        asker.receive_message()
        i += 1


def main():
    ps1 = Process(target=server, args=([5566],))
    ps2 = Process(target=server, args=([5588],))
    pc = Process(target=client, args=([5566, 5588], ))
    ps1.start()
    ps2.start()
    pc.start()
    ps1.join()
    ps2.join()
    pc.join()


if __name__ == '__main__':
    main()


