# Recurrent plot for irregular time series following Ozken et al. (2018)
#
import numpy as np
import pandas as pd
from utils import getData
from scipy.stats import norm
import matplotlib.pyplot as plt
import sys


def vspdk(sa, sb, lambda_0, lambda_k, lambda_s):
  nspi = len(sa[0])
  nspj = len(sb[0])
  scr = np.zeros([nspi + 1, nspj + 1])
  # Initialize margins with the cost of adding a data point
  scr[:, 0] = np.arange(nspi + 1).T # * lambda_s
  scr[0, :] = np.arange(nspj + 1) # * lambda_s
  if nspi and nspj > 0:
    for i in range(1, nspi + 1):
      for j in range(1, nspj + 1):
        try:
          scr[i, j] = min(scr[i - 1, j] + lambda_s, scr[i, j - 1] + lambda_s, scr[i - 1, j - 1] + lambda_0 * np.abs(sa[0, i - 1] - sb[0, j - 1]) + lambda_k * np.abs(sa[1, i - 1] - sb[1, j - 1]))
        except:
          scr[i, j] = min(scr[i - 1, j] + lambda_s, scr[i, j - 1] + lambda_s)
    d = scr[-1, -1]
  elif nspi == 0 and nspj == 0:
    d = -1
  else:
    d = np.abs(nspi - nspj) * lambda_s
  return d


def subseq(data, dt = None, avgnum = None):
  # Split the time series into subsequences
  tmin = data[0].min()
  tspan = data[0].max() - tmin
  if dt is None: dt = avgnum * np.mean(np.diff(data[0]))
  nseq = int(tspan / dt) + 1
  seqs = []
  for i in range(nseq):
    seq = data[:, (data[0] >= tmin + i * dt) & (data[0] < tmin + (i + 1) * dt)]
    seqs.append([tmin + (i + 0.5) * dt, seq])
  final = (data[:, data[0] > tmin + nseq * dt])
  seqs.append([tmin + (nseq + 0.5) * dt, final])
  return np.array(seqs)


def getLambdaS(seq, lambda_0, lambda_k):
  # Select the value of lambda_s that gives the best Gaussian distribution of distances
  ls = []
  for l_s in np.arange(0.50, 1.51, 0.01):
    dists = []
    for i in range(len(seq) - 1):
      for j in range(i + 1, len(seq)):
        dists.append(vspdk(seq[i, 1], seq[j, 1], lambda_0, lambda_k, l_s))
    dists = np.array(dists)
    dists = dists[dists > 0.]
    mu = np.mean(dists)
    sigma = np.std(dists)
    count, bins = np.histogram(dists, 25, density = False)
    count = count / len(dists)
    center = (bins[: -1] + bins[1:]) / 2
    beta = np.sqrt(2. * np.pi * (sigma ** 2.))
    gaussian = (1. / beta) * np.exp(-((center - mu) ** 2.) / (2. * sigma ** 2.))
    p2_sel = np.corrcoef(count, gaussian)[0, 1]
    ls.append([l_s, p2_sel])
#    plt.figure()
#    plt.bar(bins[:-1], count, color = 'w', edgecolor = 'k', label = 'empirical')
#    plt.plot(center, gaussian, 'r', label = 'gaussian fit')
#    plt.show()
  ls = np.array(ls).T
  lambda_s = ls[0][np.argmax(ls[1])]
  return lambda_s


def rr(rp):
  # Calculate the recurrence rate from the recurrence plot
  n = len(rp)
  idx = np.triu_indices(n, 1)
  rr = 2. * np.sum(rp[idx]) / (n * (n - 1))
  return rr


def pl(rp):
  # Histogram of diagonal lines of length l in the recurrence plot
  n = len(rp)
  p = []
  for l in range(1, n):
    sum = 0
    for i in range(1, n):
      for j in range(1, n):
        try:
          sum += (1 - rp[i - 1, j - 1]) * (1 - rp[i + l, j + l]) * np.prod(rp[i: i + l - 1, j: j + l - 1])
        except Exception as e:
          continue
    p.append(sum)
  return np.array(p)


def det(rp, l_min):
  # Determinism measure from recurrence plot quantifying the fraction of
  # recurrence points which form diagonal lines
  p = pl(rp)
  l = np.arange(1, len(p), dtype = np.float64)
  det = np.sum(np.arange(l_min, len(p)) * p[l_min - 1: -1]) / np.sum(l * p[:-1])
  return det
  

def run(id):
  # Get data and subsequences
  data, t0 = getData(id)
#  print(data[0].max() - data[0].min(), len(data[0]), np.percentile(np.diff(data[0]), [5, 50, 95]))
  seq = subseq(data, dt = 50.) # avgnum = 6.) # 
  # Determine cost of adding/deleting a point
  lambda_0 = len(data[0]) / (data[0].max() - data[0].min())
  lambda_k = 1. / np.mean(np.abs(data[1, 1:] - data[1, :-1]))
  lambda_s = getLambdaS(seq, lambda_0, lambda_k)
  # Construct recurrence plot
  w = len(seq)
  d = np.zeros([w, w])
  rp = np.zeros([w, w])
  for i in range(w - 1):
    for j in range(i + 1, w):
      d[i, j] = vspdk(seq[i, 1], seq[j, 1], lambda_0, lambda_k, lambda_s)
      d[j, i] = d[i, j]
  # Determine epsilon
  sigma_d = np.std(d[np.triu_indices(w, 1)])
  eps = sigma_d
  rp[(d <= eps) & (d >= 0.)] = 1.
  # Recurrence plot statistics
  print(f"Recurrence rate: {rr(rp)}")
  print(f"Determinism: {det(rp, 2)}")
  
#  while rr(rp) < 0.5 and eps < 2:
#    rp[d <= eps] = 1.
#    print(eps, rr(rp))
#    eps += 0.001
#  print(rr(rp))
  # Plot recurrence plot
  fig, ax = plt.subplots()
  pts = np.nonzero(rp)
  ax.scatter(pts[0], pts[1], marker = 's', color = 'k', s = 4)
  ax.set_xlim(0, w)
  ax.set_ylim(0, w)
  plt.show()

  
if __name__ == "__main__":
  run(sys.argv[1])
