import sys
from time import perf_counter
from types import ModuleType, FunctionType
from gc import get_referents

EMOTIONS = ['Anger', 'Disgust', 'Fear', 'Happiness',
           'Sadness', 'Surprise', 'Neutrality']

def getsize(obj):
    """sum size of object & members."""
    # Custom objects know their class.
    # Function objects seem to know way too much, including modules.
    # Exclude modules as well.
    BLACKLIST = type, ModuleType, FunctionType

    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size

class TimeIt:
    """
    Context manager for timing code.
    """

    def __init__(self, message: str):
        self.message = message
        self.start_time = perf_counter()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        print(f'[{perf_counter() - self.start_time:8.2f}s] {self.message}')
        return False

def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

def clear_line():
    # Clear the line with the ANSI code
    print("\r\033[K\r", end='')
    sys.stdout.flush()

def print_error(*args, **kwargs):
    print("\033[31m", end='')
    print(*args, **kwargs)
    print("\033[0m", end='')

