import heapq
import pickle


# Save obj (could be Brain object, list of saved winners, etc) as file_name
def sim_save(file_name, obj):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)


def sim_load(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


# Compute item overlap between two lists viewed as sets.
def overlap(a, b):
    return len(set(a) & set(b))


# Compute overlap of each list of winners in winners_list
# with respect to a specific winners set, namely winners_list[base]
def get_overlaps(winners_list, base, percentage=False):
    overlaps = []
    base_winners = winners_list[base]
    k = len(base_winners)
    for i in range(len(winners_list)):
        o = overlap(winners_list[i], base_winners)
        if percentage:
            overlaps.append(float(o) / float(k))
        else:
            overlaps.append(o)
    return overlaps


# jaccard相似度，两个Assembly重叠比例，两个变量都需要是b.area[area].saved_winners中的成员，area应该相同，不然没有意义
def jaccard_similarity(assemblyA, assemblyB):
    setA = set(assemblyA)
    setB = set(assemblyB)
    intersection = setA.intersection(setB)
    if len(setA) == len(setB):
        return len(intersection) / len(setA)
    else:
        return len(intersection) / len(setA.union(setB))


# core中元素出现的比例,分母是core的大小,保证输出在0~1
def overlap_on_core(core, winners):
    if len(core) == 0:
        return 0
    setA = set(core)
    setB = set(winners)
    intersection = setA.intersection(setB)
    return len(intersection) / len(setA)


class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item, priority):
        heapq.heappush(self._queue, (-priority, self._index, item))
        self._index += 1

    def pop(self):
        if len(self._queue) == 0:
            print("PriorityQueue is empty")
            return
        return heapq.heappop(self._queue)[-1]
        # ,self._queue  #[-1]表示只输出name
    # heapq.heappop(heap) 弹出索引位置0中的值
