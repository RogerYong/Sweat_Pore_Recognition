import os
import numpy as np
import utils

def load_pores_txt(pts_path):
  pts = []
  with open(pts_path, 'r') as f:
    for line in f:
      row, col = [int(t) for t in line.split()]
      #if (col >= 8 and col <= 312 and row >= 8 and row <= 232):
      pts.append((row - 1, col - 1))

  return pts
def compute_statistics(pores, detections):
  """Computes true detection rate (TDR), false detection rate (FDR) and the corresponding F-score for the given groundtruth and detections.

  Args:
    pores: pore groundtruth coordinates in format [N, P, 2].
      Second dimension must be np.arrays.
    detections: detection coordinates in the same format as pores, ie
      [N, D, 2] and second dimension np.arrays.

  Returs:
    f_score: F-score for the given detections and groundtruth.
    tdr: TDR for the given detections and groundtruth.
    fdr: FDR for the given detections and groundtruth.
  """
  # find correspondences between detections and pores
  total_pores = 0
  total_dets = 0
  true_dets = 0
  for i in range(len(pores)):
    # update totals
    total_pores += len(pores[i])
    total_dets += len(detections[i])
    true_dets += len(utils.find_correspondences(pores[i], detections[i]))

  # compute tdr, fdr and f score
  eps = 1e-5
  tdr = true_dets / (total_pores + eps)
  fdr = (total_dets - true_dets) / (total_dets + eps)
  f_score = 2 * (tdr * (1 - fdr)) / (tdr + (1 - fdr))

  return f_score, tdr, fdr

def main():
  # load only test files for both pores and detections
  pores = []
  detections = []
  for index in os.listdir(pores_dir):
    a, b = os.path.splitext(index);
    pores_path = os.path.join(pores_dir, a + '.txt')
    index_pores = load_pores_txt(pores_path)
    pores.append(np.array(index_pores))

    dets_path = os.path.join(dets_dir, a + '.txt')
    index_dets = load_pores_txt(dets_path)
    detections.append(np.array(index_dets))

  # compute statistics
  f_score, tdr, fdr = compute_statistics(pores, detections)
  print('TDR = {}'.format(tdr))
  print('FDR = {}'.format(fdr))
  print('F-score = {}'.format(f_score))
  return f_score, tdr, fdr

def cal(gt_dir, pred_dir):
  # load only test files for both pores and detections
  pores_dir = gt_dir
  dets_dir = pred_dir
  pores = []
  detections = []
  for index in os.listdir(pores_dir):
    a, b = os.path.splitext(index)
    pores_path = os.path.join(pores_dir, a + '.txt')
    index_pores = load_pores_txt(pores_path)
    pores.append(np.array(index_pores))

    dets_path = os.path.join(dets_dir, a + '.txt')
    index_dets = load_pores_txt(dets_path)
    detections.append(np.array(index_dets))

  # compute statistics
  f_score, tdr, fdr = compute_statistics(pores, detections)
  # print('TDR = {}'.format(tdr))
  # print('FDR = {}'.format(fdr))
  # print('F-score = {}'.format(f_score))
  return f_score, tdr, fdr

if __name__ == '__main__':

  pores_dir='location'
  dets_dir='res'
  main()



