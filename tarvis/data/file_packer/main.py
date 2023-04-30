from argparse import ArgumentParser
from tarvis.data.file_packer import FilePackReader, pack_directory_contents

import os


def main(args):
    if args.output_file:
        assert os.path.isdir(args.input), \
            "The input path must be a directory. To inspect an existing file pack, run without the '--output-file' arg."

        if os.path.exists(args.output_file) and not args.force:
            raise ValueError("The output file path already exists. Run with '-f' to overwrite.")

        pack_directory_contents(args.input, args.output_file,
                                permitted_exts=args.permitted_exts,
                                excluded_exts=args.excluded_exts,
                                skip_hidden=not args.include_hidden,
                                follow_links=not args.ignore_links,
                                verbose=args.verbose)
    else:
        reader = FilePackReader(args.input, verbose=True)
        reader.print_directory_structure(ignore_base_path=True,
                                         directories_only=args.directories_only)


if __name__ == '__main__':
    parser = ArgumentParser()

    # Primary arguments
    parser.add_argument(
        '--input',             '-i', required=True,
        help="For creating an fpack file, this should be a directory path. For displaying the contents of an existing "
             "file, this should be a path to that file.")
    parser.add_argument(
        '--output-file',       '-o', required=False,
        help="Path to the output file. This option is only relevant for creating an fpack file.")

    # Extra options
    parser.add_argument(
        '--force',             '-f', action='store_true',
        help="If set, the output file will be overwritten if it already exists.")

    # Write options
    writer_options = parser.add_argument_group('Writer Options')
    writer_options.add_argument(
        '--include-hidden',          action='store_true',
        help="If set, hidden files and folders in the input directory will also be packed.")
    writer_options.add_argument(
        '--ignore-links',            action='store_true',
        help="If set, symlinks in the input directory will be ignored")
    writer_options.add_argument(
        '--permitted-exts',    nargs='*', required=False,
        help="A list of permitted file extensions. If this is given, files with any other extensions will be ignored.")
    writer_options.add_argument(
        '--excluded-exts',     nargs='*', required=False,
        help="A list of file extensions to exclude. Files with any other extensions will be processed.")
    writer_options.add_argument(
        '--verbose',           '-v', action='store_true',
        help="Names of directories being will be displayed as they are being processed.")

    # Read options
    reader_options = parser.add_argument_group('Reader Options')
    reader_options.add_argument(
        '--directories-only',  action='store_true',
        help="Print only the directory names when displaying the contents of an fpack file. Useful for preventing "
             "an extremely long output.")

    main(parser.parse_args())
