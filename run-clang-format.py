#!/usr/bin/python3
import glob
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", default="clang-format", help="Path to clang-format executable")
args = parser.parse_args()

extensions = ["cu", "cpp", "h", "cuh", "hpp"]
file_paths = []

for extension in extensions:
    file_paths.extend(glob.glob('**/*.{}'.format(extension), recursive=True))

for file_path in file_paths:
    subprocess.run([args.path, "-i", "-style=file", file_path])
