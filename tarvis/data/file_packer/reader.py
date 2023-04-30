from io import BytesIO, StringIO
from tqdm import tqdm
from contextlib import contextmanager

try:
    from torch.multiprocessing import Lock
except ImportError as _:
    print("[ WARN] 'torch' could not be imported. The native Python 'multiprocessing.Lock' will be used to synchronize "
          "concurrent accesses to the file handle.")
    from multiprocessing import Lock

from tarvis.data.file_packer import FilePackHeader

import os

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError as _:
    print("[ WARN] 'imread_cv2' function will be unavailable because either OpenCV or NumPy could not be imported")
    CV2_AVAILABLE = False


class DummyLock:
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


try:
    _LOCK = Lock()
except PermissionError as _:
    print("[ WARN] Could not create Lock object. FilePack access from concurrent processes might be problematic.")

    _LOCK = DummyLock()


class FilePackReader(object):
    def __init__(self, file, base_path='', verbose=False, multiprocess_lock=True):
        if isinstance(file, str):
            self.__ifile = open(file, 'rb', buffering=0)
        else:
            self.__ifile = file

        header = FilePackHeader.read_from_file_handle(self.__ifile)
        self.base_path = header.base_path
        if base_path:  # overwrite base path if one is provided
            self.base_path = base_path

        # build a dictionary of the file paths (including directory structure) to their corresponding offsets in the
        # packed file
        if verbose:
            print("Loading and parsing file pack header...")

        self.__file_map = dict()
        offset = header.data_offset

        def populate_file_map_recursive(subdict, path_tokens, offset, byte_size):
            if len(path_tokens) > 1:
                if path_tokens[0] not in subdict:
                    subdict[path_tokens[0]] = dict()
                populate_file_map_recursive(subdict[path_tokens[0]], path_tokens[1:], offset, byte_size)
            else:
                subdict[path_tokens[0]] = (offset, byte_size)
            return

        self.__total_files = header.num_files
        for i in tqdm(range(header.num_files), disable=not verbose):
            path_tokens = [t for t in header.filenames[i].split('/') if t]
            populate_file_map_recursive(self.__file_map, path_tokens, offset, header.buffer_sizes[i])
            offset += header.buffer_sizes[i]

        if not multiprocess_lock:
            global _LOCK
            _LOCK = DummyLock()

    def __del__(self):
        self.__ifile.close()

    def __len__(self):
        return self.__total_files

    def __strip_base_path(self, path):
        if path[:len(self.base_path)] != self.base_path:
            raise IOError("The provided path '{}' does not have the expected base path as the file pack: '{}'".format(
                path, self.base_path
            ))
        return path[len(self.base_path):].lstrip('/')

    def __get_path_element(self, path):
        path_tokens = [t for t in path.split('/') if t]
        path_map = self.__file_map
        for t in path_tokens:
            path_map = path_map.get(t)
            if not path_map:
                raise IOError("No file/directory named '{}' found in file pack".format(path))
        return path_map

    def __get_file_info(self, filepath):
        file_info = self.__get_path_element(filepath)
        assert isinstance(file_info, (tuple, list))
        return file_info

    def read_bytes(self, filepath, exclude_base_path):
        if not exclude_base_path:
            filepath = self.__strip_base_path(filepath)

        file_info = self.__get_file_info(filepath)
        offset, num_bytes = file_info

        global _LOCK
        with _LOCK:
            self.__ifile.seek(offset)
            file_buffer = self.__ifile.read(num_bytes)
        return file_buffer

    def open(self, filepath, mode, exclude_base_path=False):
        assert mode in ("r", "rb"), "Files stored in a file pack are read-only"
        bytes_in = self.read_bytes(filepath, exclude_base_path)
        if mode == "rb":
            return BytesIO(bytes_in)
        else:
            return StringIO(bytes_in.decode("utf-8"))

    def cv2_imread(self, filepath, flags=cv2.IMREAD_UNCHANGED, exclude_base_path=False):
        if not CV2_AVAILABLE:
            raise IOError("OpenCV and/or NumPy could not be imported")
        image = self.read_bytes(filepath, exclude_base_path)
        image = np.frombuffer(image, np.uint8)
        return cv2.imdecode(image, flags)

    def exists(self, path, exclude_base_path=False):
        if not exclude_base_path:
            path = self.__strip_base_path(path)

        try:
            self.__get_path_element(path)
        except IOError as _:
            return False
        return True

    def isdir(self, path, exclude_base_path=False):
        if not exclude_base_path:
            path = self.__strip_base_path(path)

        return isinstance(self.__get_path_element(path), dict)

    def isfile(self, path, exclude_base_path=False):
        if not exclude_base_path:
            path = self.__strip_base_path(path)

        return isinstance(self.__get_path_element(path), tuple)

    def listdir(self, dirpath, exclude_base_dir=False):
        if not exclude_base_dir:
            dirpath = self.__strip_base_path(dirpath)

        dir_dict = self.__get_path_element(dirpath)
        assert isinstance(dir_dict, dict)
        return sorted(list(dir_dict.keys()))
        # path_tokens = [t for t in dirpath.split('/') if t]

    def walk(self, exclude_base_path=False, sort_entries=False):
        base_dirpath = '' if exclude_base_path else self.base_path

        def walk_recursive(dirpath, subdict):
            dirnames = []
            filenames = []
            for path_token, v in subdict.items():
                if isinstance(v, dict):
                    dirnames.append(path_token)
                elif isinstance(v, (tuple, list)):
                    filenames.append(path_token)
                else:
                    raise ValueError("Should not be here")

            if sort_entries:
                dirnames, filenames = sorted(dirnames), sorted(filenames)

            yield dirpath, dirnames, filenames
            for dirname in dirnames:
                yield from walk_recursive(os.path.join(dirpath, dirname), subdict[dirname])

        return walk_recursive(base_dirpath, self.__file_map)

    def print_directory_structure(self, ignore_base_path=False, directories_only=False, indentation=3):
        def print_recursive(subdict, indent):
            for k, v in subdict.items():
                indent_str = " " * indent
                if isinstance(v, dict):
                    print("{}- {}: {}".format(indent_str, k, '({})'.format(len(v))))
                    print_recursive(v, indent + indentation)
                elif isinstance(v, (tuple, list)):
                    if not directories_only:
                        print("{}- {}".format(indent_str, k))
                else:
                    raise ValueError("Should not be here")

        if ignore_base_path:
            print_recursive(self.__file_map, 0)
        else:
            print("-{}:".format(self.base_path))
            print_recursive(self.__file_map, indentation)
