import numpy as np

def get_topk_effectiveness(mac, map, k):
   assert(k <= mac)
   topvalues = []
   for i in range(mac):
      topvalues.append(sum(map[i])-map[i][i])
   topvalues = sorted(topvalues, reverse=True)
   return topvalues[:k]

def max_mean_dist_split(topk):
  if len(topk) == 1 or len(topk) == 2:
     return 999
  i = 1
  boundry = 1
  maxmeandiff = 0
  while i < len(topk):
      t1 = topk[:i]
      t2 = topk[i:]
      m1 = np.mean(t1)
      m2 = np.mean(t2)
      meandiff = abs(m1 - m2)
      if meandiff > maxmeandiff:
         maxmeandiff = meandiff
         boundry = i
      i += 1
  l1 = topk[:boundry]
  l2 = topk[boundry:]
  return abs(np.mean(l1)-np.mean(l2))