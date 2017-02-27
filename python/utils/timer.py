from time import time


class Timer:
    @staticmethod
    def now():
        return time()

    def __init__(self):
        self.timers = dict()

    def start(self, *args):
        start_time = time()
        for arg in args:
            self.timers[arg] = start_time

        return [start_time] * len(args)

    def end(self, *args):
        end_time, times = time(), []
        for arg in args:
            if arg in self.timers:
                times.append(end_time - self.timers[arg])
                del self.timers[arg]
            else:
                times.append(0.0)

        return times if len(times) > 1 else times[0]
