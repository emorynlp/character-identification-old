

class MultiprocessingUtils(object):
    @staticmethod
    def get_array_split_indices(array_length, thread_count):
        array_length += 1
        indices = range(0, array_length, array_length / thread_count)[:thread_count]
        indices.append(array_length)

        return indices

    @staticmethod
    def start_all_processes(processes):
        for process in processes:
            process.start()

    @staticmethod
    def wait_for_all_processes(processes):
        for process in processes:
            process.join()
