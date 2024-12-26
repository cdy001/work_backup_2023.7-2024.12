import time


def time_cost(func):
    def print_time_cost(*args, **kargs):
        time_start = time.time()
        result = func(*args, **kargs)
        time_end = time.time()
        print(f"function {func.__name__} cost time: {time_end - time_start: .4f}s")
        return result
    return print_time_cost


def time_cost_async(func):
    async def print_time_cost(*args, **kargs):
        time_start = time.time()
        result = await func(*args, **kargs)
        time_end = time.time()
        print(f"function {func.__name__} cost time: {time_end - time_start: .4f}s")
        return result
    return print_time_cost