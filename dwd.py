import numpy as np
import pandas as pd
from .DWD1SM import DWD1SM


def dwd(Xp, Xn):

  # match genes between platforms
  p1_genes = Xp.index.values
  p2_genes = Xn.index.values
  intersecting_genes = list(set(p1_genes) & set(p2_genes))
  Xp = Xp.reindex(intersecting_genes)
  Xn = Xn.reindex(intersecting_genes)

  Xp = np.log2(Xp.sort_index())
  Xn = np.log2(Xn.sort_index())

  Xp_vals = Xp.values
  Xn_vals = Xn.values

  dirvec = DWD1SM(Xp_vals, Xn_vals)

  vprojp = np.matmul(Xp_vals.T, dirvec)
  vprojn = np.matmul(Xn_vals.T, dirvec)

  meanprojp = np.mean(vprojp)
  meanprojn = np.mean(vprojn)

  platform1_adjustment = -1 * meanprojp * dirvec
  platform2_adjustment = -1 * meanprojn * dirvec

  x = Xp.add(platform1_adjustment, axis = 0)
  y = Xn.add(platform2_adjustment, axis = 0)

  output = {'x': x,
            'y': y,
            'adjustment1': platform1_adjustment, 
            'adjustment2': platform2_adjustment}

  return output