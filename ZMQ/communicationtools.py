import zmq
import time
import datetime as dt
import socket as st


class Communication:
    def __init__(self, socket_name_, socket_type_, bind_connect_, port_list_=None, address_list_=None):
        context = zmq.Context()
        self.name = socket_name_
        self.socket = context.socket(socket_type_)
        if bind_connect_ == 'bind' and address_list_ is None:
            self.socket.bind("tcp://127.0.0.1:{}".format(port_list_[0]))
        elif bind_connect_ == 'connect' and address_list_ is not None:
            for address in address_list_:
                for port in port_list_:
                    self.socket.connect("tcp://{}:{}".format(address, port))
        else:
            raise ValueError('mode({}) and address({}) did not match'.format(bind_connect_, address_list_))

    def send_message(self, message_):
        self.socket.send_string(message_)
        print('{}_{} send a message === {} === at {}'
              .format(self.name, st.gethostname(), message_, dt.datetime.now()))
        time.sleep(1)

    def send_object(self, object_):
        self.socket.send_pyobj(object_)
        print('{}_{} send an object at {}'.format(self.name, st.gethostname(), dt.datetime.now()))
        time.sleep(2)

    def receive_message(self):
        message = self.socket.recv_string()
        print('{}_{} receive a message === {} === at {}'
              .format(self.name, st.gethostname(), message, dt.datetime.now()))
        return message

    def receive_object(self):
        obj = self.socket.recv_pyobj()
        print('{}_{} receive an object at {}'.format(self.name, st.gethostname(), dt.datetime.now()))
        return obj






