import multiprocessing
from multiprocessing import Pool
import os
import shutil
from pathlib import Path
import time

start_time = time.time()
current_path = os.path.dirname(os.path.realpath(__file__))
dataset_folder = os.path.join(r'D:\Code\Research\PointNet2\pheno4d_mt')
txt_folder = os.path.join(current_path, 'pheno4d_kpconv')

categories = ['Maize', 'Tomato']
filepaths = []

def getFiles():
  for c in categories:
    src_folder = os.path.join(dataset_folder, c)
    filenames = os.listdir(src_folder)
    for filename in filenames:
      filepath = os.path.join(src_folder, filename)
      filepaths.append(str(filepath))

def process_filepaths(filepath):
  des_folder_maize = os.path.join(txt_folder, 'Maize')
  des_folder_tomato = os.path.join(txt_folder, 'Tomato')
  if not os.path.exists(des_folder_maize): os.makedirs(des_folder_maize)
  if not os.path.exists(des_folder_tomato): os.makedirs(des_folder_tomato)
  base_filename = os.path.basename(filepath)
  filename, file_extension = os.path.splitext(base_filename)
  des_folder = des_folder_maize if base_filename.startswith('M') else des_folder_tomato
  filepath_copy = os.path.join(des_folder, filename + '.txt')
  shutil.copy2(filepath, filepath_copy)
  create_pts(filepath_copy)
  create_seg(filepath_copy)
  os.remove(filepath_copy)

def create_pts(filepath):
  folder = os.path.dirname(filepath)
  pts_folder = os.path.join(folder, 'points')
  if not os.path.exists(pts_folder): os.makedirs(pts_folder)
  base_filename = os.path.basename(filepath)
  filename, file_extension = os.path.splitext(base_filename)
  pts_filepath = os.path.join(pts_folder, filename + '.pts')
  with open(filepath, 'r') as fid:
    lines = []
    for line in fid:
      if line == '\n': continue
      nums = line.split()
      nums = nums[:3]
      lines.append(' '.join(nums))

  content = '\n'.join(lines)
  with open(pts_filepath, 'w') as fid:
      fid.write(content)

def create_seg(filepath):
  folder = os.path.dirname(filepath)
  pts_folder = os.path.join(folder, 'points_label')
  if not os.path.exists(pts_folder): os.makedirs(pts_folder)
  base_filename = os.path.basename(filepath)
  filename, file_extension = os.path.splitext(base_filename)
  seg_filepath = os.path.join(pts_folder, filename + '.seg')
  with open(filepath, 'r') as fid:
    lines = []
    for line in fid:
      if line == '\n': continue
      nums = line.split()
      nums = str(int(float(nums[-1]))+1)
      lines.append(' '.join(nums))

  content = '\n'.join(lines)
  with open(seg_filepath, 'w') as fid:
      fid.write(content)

def run():
  for f in filepaths:
    process_filepaths(f)

def pool_handler():
  cpu = multiprocessing.cpu_count()
  p = Pool(cpu)
  p.map(process_filepaths, filepaths)

def compute_stats():
  execution_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
  print(execution_time)

if __name__ == '__main__':
  getFiles()
  pool_handler()
  #run()
  compute_stats()