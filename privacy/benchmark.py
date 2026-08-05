import time


def measure_latency(func, *args, **kwargs):
    """
    Measure execution time of a function.
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()

    return result, end - start