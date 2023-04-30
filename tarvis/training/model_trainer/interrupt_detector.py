import random
import signal


class InterruptDetector:
    def __init__(self):
        self.__is_interrupted = False

    def start(self):
        signal.signal(signal.SIGINT, self.__set_interrupted)
        signal.signal(signal.SIGTERM, self.__set_interrupted)

    def __set_interrupted(self, signum, frame):
        self.__is_interrupted = True

    is_interrupted = property(fget=lambda self: self.__is_interrupted)