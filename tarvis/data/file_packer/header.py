import struct


class FilePackHeader(object):
    __MAX_FILENAME_LENGTH = 128
    __MAX_BASE_PATH_LENGTH = 256

    def __init__(self, version, filenames, buffer_sizes, base_path=''):
        self.version = version
        self.filenames = filenames
        self.num_files = len(filenames)

        if self.num_files != len(buffer_sizes):
            raise ValueError("'buffer_sizes' and 'filenames' must contain an equal number of elements "
                             "({} vs. {})".format(len(buffer_sizes), self.num_files))
        self.buffer_sizes = buffer_sizes

        self.base_path = base_path
        if len(base_path) > self.__MAX_BASE_PATH_LENGTH:
            raise ValueError("Base path cannot be longer than {} characters. The provided base path in {} "
                             "characters long".format(self.__MAX_BASE_PATH_LENGTH, len(base_path)))

        self.__padded_filenames = []
        self.data_offset = -1

    def __str__(self):
        return "Format version: {}\n" \
               "Number of files: {}\n" \
               "Base path: {}".format(self.version, self.num_files, self.base_path)

    def pretty(self):
        s = [
            "{}: {}".format(filename, buffer_size)
            for filename, buffer_size in zip(self.filenames, self.buffer_sizes)
        ]
        s = "\n".join(s)
        return str(self) + "\nIMAGE FILES: BUFFER SIZES\n" + s

    def _pad_filenames(self):
        for filename in self.filenames:
            if len(filename) > self.__MAX_FILENAME_LENGTH:
                raise ValueError("Filenames cannot be longer than {}. The filename {} has a length of {}".format(
                    self.__MAX_FILENAME_LENGTH, filename, len(filename)))
            self.__padded_filenames.append(filename.ljust(self.__MAX_FILENAME_LENGTH))

    def write(self, ofile):
        # write file version
        version = struct.pack('B', self.version)
        ofile.write(version)

        # write image count
        num_images = struct.pack('Q', self.num_files)
        ofile.write(num_images)

        # write base path
        base_path = self.base_path.ljust(self.__MAX_BASE_PATH_LENGTH)
        base_path = bytearray(base_path, encoding='utf-8')
        base_path = struct.pack('%ds' % self.__MAX_BASE_PATH_LENGTH, base_path)
        ofile.write(base_path)

        # pad filenames with spaces if not already done
        if not self.__padded_filenames:
            self._pad_filenames()

        # write file names
        filenames_block_length = self.__MAX_FILENAME_LENGTH * self.num_files
        filenames_block = ''.join(self.__padded_filenames)
        filenames_block = bytearray(filenames_block, encoding='utf-8')
        filenames_block = struct.pack('%ds' % filenames_block_length, filenames_block)
        ofile.write(filenames_block)

        # write out file buffer sizes
        buffer_sizes = struct.pack('%dQ' % self.num_files, *self.buffer_sizes)
        ofile.write(buffer_sizes)

    @classmethod
    def read_from_file_handle(cls, ifile):
        # read file format version
        version = struct.unpack('B', ifile.read(1))[0]

        # read number of images
        num_images = struct.unpack('Q', ifile.read(8))[0]

        # read base path
        base_path = struct.unpack('%ds' % cls.__MAX_BASE_PATH_LENGTH, ifile.read(cls.__MAX_BASE_PATH_LENGTH))[0]
        base_path = base_path.decode('utf-8').rstrip()

        # read image file names
        filenames_block_length = num_images * cls.__MAX_FILENAME_LENGTH
        filenames = struct.unpack('%ds' % filenames_block_length, ifile.read(filenames_block_length))[0]
        filenames = filenames.decode('utf-8')
        filenames = [filenames[i:i+cls.__MAX_FILENAME_LENGTH].rstrip() for i in
                     range(0, len(filenames), cls.__MAX_FILENAME_LENGTH)]

        # read image buffer sizes
        buffer_sizes = struct.unpack('%dQ' % num_images, ifile.read(8 * num_images))

        header = cls(version, filenames, buffer_sizes, base_path)
        header.data_offset = ifile.tell()
        return header


if __name__ == '__main__':
    header = FilePackHeader(version=1,
                            filenames=['abc.jpg', 'def.png'],
                            buffer_sizes=[10000, 20000],
                            base_path='/root/ali/')
    FILE_PATH = '/tmp/test.impack'
    with open(FILE_PATH, 'wb') as writefile:
        header.write(writefile)

    with open(FILE_PATH, 'rb') as readfile:
        header = FilePackHeader.read_from_file_handle(readfile)
        print(header.pretty())
