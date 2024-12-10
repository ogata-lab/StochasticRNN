#
# Copyright (c) Since 2024 Ogata Laboratory, Waseda University
# Released under the MIT License.
#

import os
import json
import datetime


OK = "\033[92m"
WARN = "\033[93m"
NG = "\033[91m"
END_CODE = "\033[0m"


def print_info(msg):
    print(OK + "[INFO] " + END_CODE + msg)


def print_warn(msg):
    print(WARN + "[WARNING] " + END_CODE + msg)


def print_error(msg):
    print(NG + "[ERROR] " + END_CODE + msg)


def print_args(args):
    """Print arguments"""
    if not isinstance(args, dict):
        args = vars(args)

    keys = args.keys()
    keys = sorted(keys)

    print("================================")
    for key in keys:
        print("{} : {}".format(key, args[key]))
    print("================================")


def save_args(args, filename):
    """Dump arguments as json file"""
    with open(filename, "w") as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)


def restore_args(filename):
    """Load argument file from file"""
    with open(filename, "r") as f:
        args = json.load(f)
    return args


def check_args(args):
    """Check arguments"""

    if args.tag is None:
        tag = datetime.datetime.today().strftime("%Y%m%d_%H%M_%S")
        args.tag = tag
        print_info("Set tag = %s" % tag)

    # make log directory
    check_path(os.path.join(args.log_dir, args.tag), mkdir=True)

    # saves arguments into json file
    save_args(args, os.path.join(args.log_dir, args.tag, "args.json"))

    print_args(args)
    return args


def check_path(path, mkdir=False):
    """
    Checks that path is collect
    """
    if path[-1] == "/":
        path = path[:-1]

    if not os.path.exists(path):
        if mkdir:
            os.makedirs(path, exist_ok=True)
        else:
            raise ValueError("%s does not exist" % path)

    return path


def set_logdir(log_dir, tag):
    return check_path(os.path.join(log_dir, tag), mkdir=True)


def normalization(data, indataRange, outdataRange):
    """
    Function to normalize a numpy array within a specified range
    Args:
        data (np.array): Data array
        indataRange (float list):  List of maximum and minimum values of original data, e.g. indataRange=[0.0, 255.0].
        outdataRange (float list): List of maximum and minimum values of output data, e.g. indataRange=[0.0, 1.0].
    Return:
        data (np.array): Normalized data array
    """
    data = (data - indataRange[0]) / (indataRange[1] - indataRange[0])
    data = data * (outdataRange[1] - outdataRange[0]) + outdataRange[0]
    return data


def tensor2numpy(x):
    """
    Convert tensor to numpy array.
    """
    if x.device.type == "cpu":
        return x.detach().numpy()
    else:
        return x.cpu().detach().numpy()
