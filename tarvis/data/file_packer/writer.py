from tqdm import tqdm
from tarvis.data.file_packer import FilePackHeader

import os


class FilePackWriter(object):
    def __init__(self, outfile, file_paths):
        if isinstance(outfile, str):
            self.__ofile = open(outfile, 'wb')
        else:
            self.__ofile = outfile

        self.file_paths = file_paths

    def __del__(self):
        self.__ofile.close()

    def write(self, base_path='', paths_to_save=None):
        # read all files once to know their raw byte size and dimensions
        file_buffer_sizes = []

        if not paths_to_save:
            paths_to_save = self.file_paths
        elif len(paths_to_save) != len(self.file_paths):
            raise ValueError("'paths_to_save' must have the same length as the number of files to pack, but lengths "
                             "are {} and {}".format(len(paths_to_save), len(self.file_paths)))

        print("Reading file sizes...")
        for path in tqdm(self.file_paths):
            file_buffer_sizes.append(os.path.getsize(path))

        header = FilePackHeader(1, paths_to_save, file_buffer_sizes, base_path)
        header.write(self.__ofile)

        # load files again to actually write them out to an file pack file.
        print("Writing files to pack...")
        for buffer_size, path in tqdm(zip(file_buffer_sizes, self.file_paths), total=len(self.file_paths)):
            with open(path, 'rb') as readfile:
                file = readfile.read()
            # print("Writing at ", self.__ofile.tell())
            self.__ofile.write(file)


def pack_directory_contents(top_dir, pack_file, permitted_exts=(), excluded_exts=(), skip_hidden=True,
                            follow_links=True, verbose=False):
    if permitted_exts and excluded_exts:
        raise ValueError("Either 'permitted_exts' or 'excluded_exts' can be provided, but not both")

    def ext_permitted(ext):
        if permitted_exts:
            return ext in permitted_exts
        if excluded_exts:
            return ext not in excluded_exts
        return True

    def file_permitted(filename):
        if filename.lstrip('/').startswith('.') and skip_hidden:
            return False

        extension = os.path.splitext(filename)[1][1:]
        return ext_permitted(extension)

    def dir_permitted(dirpath):
        dirname = os.path.split(dirpath)[-1].lstrip('/')
        if dirname.startswith('.') and skip_hidden:
            return False
        return True

    print("[ INFO] Recursively iterating over directory: '{}'. Depending on the number of files, this may take a while."
          " Set the '--verbose' flag to get debug output.".format(top_dir))

    file_paths = []
    for dirpath, _, filenames in os.walk(top_dir, followlinks=follow_links):
        if not dir_permitted(dirpath):
            continue

        if verbose:
            print("Searching directory: {}".format(dirpath))

        file_paths.extend([
            os.path.join(dirpath, f) for f in filenames if file_permitted(f)
        ])

    top_dir_len = len(top_dir)
    paths_to_save = [path[top_dir_len+1:] for path in file_paths]

    print("[ INFO] Found a total of {} files to pack".format(len(paths_to_save)))
    writer = FilePackWriter(pack_file, file_paths)
    writer.write(top_dir, paths_to_save)
