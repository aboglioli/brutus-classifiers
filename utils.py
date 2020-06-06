# Utils
def merge_two(arr1, arr2):
    res = []
    for v1 in arr1:
        for v2 in arr2:
            if type(v1) == list:
                res.append(v1 + [v2])
            else:
                res.append([v1, v2])
    return res

def merge(arrs):
    res = arrs[0]
    rest = arrs[1:]
    for arr in rest:
      res = merge_two(res, arr)
    return res

def split_obj(obj):
  res = []
  for k in obj:
    sub = []
    for i in obj[k]:
      sub.append({k: i})
    res.append(sub)
  return res

def compact_obj(arrs):
  res = []
  for arr in arrs:
    sub = {}
    for obj in arr:
      for k in obj:
        sub[k] = obj[k]
    res.append(sub)
  return res

def merge_args(args):
    return compact_obj(merge(split_obj(args)))
