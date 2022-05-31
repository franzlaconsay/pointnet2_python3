import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--file', required=True)
parser.add_argument('--label_base', type=int, default=0)
parser.add_argument('--category')
FLAGS = parser.parse_args()

FILE = FLAGS.file
filename, file_extension = os.path.splitext(FILE)
PLY_FILE = FILE
TXT_FILE = filename + '.txt'
TXT_FILE_SEGMENTED = filename + '_segmented.txt'
PLY_FILE_SEGMENTED = filename + '_segmented.ply'
LABEL_BASE = FLAGS.label_base
CATEGORY = FLAGS.category
ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
SINGLE_FOLDER = os.path.join(ROOT_PATH, 'data', 'single')
MODEL_PATH = os.path.join(ROOT_PATH, 'part_seg', '%s_100_k2' % CATEGORY)
LOG_DIR = 'log_single'
ply_header = []
num_point = 0


def ply_to_txt():
  input_file = os.path.join(SINGLE_FOLDER, PLY_FILE)
  output_filepath = os.path.join(SINGLE_FOLDER, TXT_FILE)
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
  with open(output_filepath, 'w') as fid:
      fid.write(content)

def txt_to_ply():
  input_file = os.path.join(SINGLE_FOLDER, TXT_FILE_SEGMENTED)
  output_filepath = os.path.join(SINGLE_FOLDER, PLY_FILE_SEGMENTED)
  global num_point
  with open(input_file, 'r') as fid:
    lines = []
    for line in fid:
      if line == '\n': continue
      nums = line.split()
      lines.append(' '.join(nums))

  content = '\n'.join(ply_header + lines)
  with open(output_filepath, 'w') as fid:
      fid.write(content)

def segment():
  py_script = os.path.join('part_seg', 'evaluate_pheno4d_single.py')
  cmd = 'python %s --filename %s --num_point %s --category %s --log_dir %s --model_path %s' % (py_script, TXT_FILE_SEGMENTED, num_point, CATEGORY, LOG_DIR, MODEL_PATH)
  print(cmd)
  os.system(cmd)

if __name__ == '__main__':
  ply_to_txt()
  segment()
  txt_to_ply()