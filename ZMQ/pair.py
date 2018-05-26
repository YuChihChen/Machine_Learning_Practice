import zmq
import communicationtools as ct
from multiprocessing import Process


def pair_server():
    server = ct.Communication('server', zmq.PAIR, 'bind', [5566])
    while True:
        server.send_message("We are pair. I am a server")
        msg = server.receive_message()


def pair_client():
    port = "5556"
    client = ct.Communication('client', zmq.PAIR, 'connect', [5566], ['127.0.0.1'])
    while True:
        client.send_message("We are pair. I am a client")
        msg = client.receive_message()


def main():
    ps = Process(target=pair_server)
    pc = Process(target=pair_client)
    ps.start()
    pc.start()
    ps.join()
    pc.join()


if __name__ == '__main__':
    main()