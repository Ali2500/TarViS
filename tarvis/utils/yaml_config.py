from io import IOBase

import argparse
import os
import yaml


class YamlConfig(dict):
    def __init__(self, d, scope):
        self.__immutable = False
        self.__scope = scope
        super(self.__class__, self).__init__()

        for k, v in d.items():
            if isinstance(v, dict):
                self.__setattr__(k, self.__class__(v, self.__scope + k + '.'))
            else:
                self.__setattr__(k, v)

        self.__immutable = True  # prevents changes being made at runtime

    def __getattr__(self, item):
        attr = self.get(item, None)
        if attr is None and not item.startswith('_' + self.__class__.__name__ + '__'):
            raise ValueError("No attribute named '%s' found in config scope '%s'" % (item, self.__scope))
        return attr

    def __setattr__(self, key, value):
        if self.__immutable and key != '_' + self.__class__.__name__ + '__immutable':
            raise ValueError("The config is immutable and cannot be modified")

        return self.__setitem__(key, value)

    def __setitem__(self, key, value):
        if self.__immutable and key != '_' + self.__class__.__name__ + '__immutable':
            raise ValueError("The config is immutable and cannot be modified")

        return super(self.__class__, self).__setitem__(key, value)

    def __str__(self):
        return self.pretty()

    def __repr__(self):
        return self.pretty()

    @property
    def scope(self):
        return self.__scope

    def pretty(self, left_margin=0):
        s = ""
        for k, v in self.items():
            if k.startswith('_' + self.__class__.__name__ + '__'):
                continue

            for i in range(left_margin):
                s += " "

            if isinstance(v, self.__class__):
                s = s + k + ":\n" + str(v.pretty(left_margin + 2))
            else:
                s = s + k + ": " + str(v) + "\n"
        return s

    def merge_with(self, opts, strict=True, verbose=False):
        self.__immutable = False
        unexpected_keys = []

        for key, val in opts.items():
            if key.startswith("_YamlConfig__"):
                continue

            if key not in self:
                if strict:
                    self.__immutable = True
                    raise ValueError("No option named '%s' exists in YamlConfig" % key)
                else:
                    unexpected_keys.append(key)
            else:
                value = self[key]
                if isinstance(value, self.__class__):
                    unexpected_keys.extend(value.merge_with(val, strict))
                else:
                    self[key] = val

        self.__immutable = True
        return unexpected_keys

    def merge_from_file(self, path, strict=True, verbose=False):
        other_cfg = self.__class__.load_from_file(path)
        return self.merge_with(other_cfg, strict=strict, verbose=verbose)

    def update_param(self, name, new_value):
        """
        Method to update the value of a given parameter.
        :param name:
        :param new_value:
        :return:
        """
        if name not in self:
            raise ValueError("No parameter named '{}' exists".format(name))
        self.__immutable = False
        self[name] = new_value
        self.__immutable = True

    def update_from_args(self, args, verbose=False, prefix="cfg"):
        """
        Update the values based on user input given via 'argparse.ArgumentParser'.
        :param args:
        :param verbose:
        :param prefix: If the arg names have some prefix attached to them, provide it here so it is parsed correctly.
        :return:
        """
        self.__immutable = False
        for arg_name, arg_value in vars(args).items():
            if arg_value is None or not arg_name.startswith(f"{prefix}."):
                continue

            n_skip = len(prefix) + 1
            arg_name = arg_name[n_skip:]

            tokens = arg_name.split(".")

            target_cfg = self
            for i, token in enumerate(tokens, 1):
                assert token in target_cfg, f"'{token}' not found in scope '{target_cfg.scope}'"

                if i < len(tokens):
                    target_cfg = target_cfg[token]
                else:
                    target_cfg.update_param(token, arg_value)

            if verbose:
                print(f"{arg_name} --> {arg_value}")

        self.__immutable = True

    def add_args_to_parser(self, parser, recursive=True, prefix="cfg", suppress_help=False):
        """
        Populates an ArgumentParser instance with argument names from the config instance.
        :param parser: Instance of argparse.ArgumentParser
        :param recursive: If True, config values in nested scoped will also be added
        :param prefix: A string prefix that will be prepended to the arg names
        :param suppress_help: bool
        :return:
        """
        assert prefix

        def str2bool(v):
            if v.lower() in ("yes", "true", "on", "t", "1"):
                return True
            elif v.lower() in ("no", "false", "off", "f", "0"):
                return False
            else:
                raise ValueError("Failed to cast '{}' to boolean type".format(v))

        parser.register('type', 'bool', str2bool)

        for key, val in self.items():
            if key.startswith(f"_{self.__class__.__name__}__"):
                continue

            if isinstance(val, self.__class__):
                if recursive:
                    # print(f"Entering into: '{key}' from scope '{self.__scope}'")
                    val.add_args_to_parser(parser, True, f"{prefix}.{key}", suppress_help)
                else:
                    continue

            if suppress_help:
                extra_args = {"help": argparse.SUPPRESS}
            else:
                extra_args = dict()

            if isinstance(val, (list, tuple)):
                parser.add_argument(f"--{prefix}.{key}", nargs='*', type=type(val[0]), required=False, **extra_args)
            elif isinstance(val, bool):
                parser.add_argument(f"--{prefix}.{key}", type='bool', required=False, **extra_args)
            else:
                # print(f"Adding arg: {prefix}, {key}")
                parser.add_argument(f"--{prefix}.{key}", type=type(val), required=False, **extra_args)

        return parser

    def d(self):
        return self.as_dict()

    def as_dict(self):
        """
        Converts the object instance to a standard Python dict
        :return: object instance parsed as dict
        """
        d = dict()
        for k, v in self.items():
            if k.startswith('_' + self.__class__.__name__ + '__'):  # ignore class variables
                continue
            if isinstance(v, self.__class__):
                d[k] = v.as_dict()
            else:
                d[k] = v

        return d

    def dump(self, f):
        """
        Saves the current config to a file
        :param f: filepath as str or file-like object
        :return: None
        """
        if isinstance(f, str):
            f = open(f, 'w')
        elif not isinstance(f, IOBase):
            raise TypeError("'f' must be a string or file-like object")

        yaml.dump(self.d(), f)
        f.close()

    @classmethod
    def load_from_file(cls, config_file_path):
        assert os.path.exists(config_file_path), "config file not found at given path: %s" % config_file_path

        pyyaml_major_version = int(yaml.__version__.split('.')[0])
        pyyaml_minor_version = int(yaml.__version__.split('.')[1])
        required_loader_arg = pyyaml_major_version >= 5 and pyyaml_minor_version >= 1

        with open(config_file_path, 'r') as readfile:
            try:
                if required_loader_arg:
                    d = yaml.load(readfile, Loader=yaml.FullLoader)
                else:
                    d = yaml.load(readfile, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                mark = exc.problem_mark
                print("Error position: ({}:{})".format(mark.line+1, mark.column+1))
                exit(0)

        yaml_config = cls(d, '')
        return yaml_config

    scope = property(fget=lambda self: self.__scope)
