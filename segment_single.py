import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--file', required=True)
parser.add_argument('--label_base', type=int, default=1)
FLAGS = parser.parse_args()

FILE = FLAGS.file
LABEL_BASE = FLAGS.label_base
CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
FOLDER = os.path.join(CURRENT_PATH, 'data', 'single')
ply_header = []
num_point = 0


def ply_to_txt():
  base_filename = os.path.basename(FILE)
  filename, file_extension = os.path.splitext(base_filename)
  filepath = os.path.join(FOLDER, filename + '.txt')
  input_file = os.path.join(FOLDER, FILE)
  global num_point
  with open(input_file, 'r') as fid:
    lines = []
    header_done = False
    for line in fid:
      if line == '\n': continue
      nums = line.split()
      if nums[0] == 'end_header':
        ply_header.append(' '.join(nums))
        header_done = True
        continue
      if not header_done:
        ply_header.append(' '.join(nums))
        continue
      nums[-1] = str(float(nums[-1])-LABEL_BASE)
      lines.append(' '.join(nums))
      num_point += 1

  content = '\n'.join(lines)
  with open(filepath, 'w') as fid:
      fid.write(content)

if __name__ == '__main__':
  ply_to_txt()
  print(num_point)
  #print('\n'.join(ply_header))