import numpy as np
import heapq
from collections import defaultdict
from brain_util import overlap_on_core, PriorityQueue
from scipy.stats import binom
from scipy.stats import truncnorm
import math
import random
from brain import Assembly, Stimulus
from brain_util import jaccard_similarity


# 因为重写了Area,如果继承意味着brain中大多数方法要重写
class Area:
    def __init__(self, name, n, k, beta=0.05, similarity_threshold=0.4):
        # 相似度阈值
        self.similarity_threshold = similarity_threshold
        # 字典 名字：assembly
        self.assembly_list = {}
        # 疲劳的神经元, 比如已经出现于assembly中的, 之后在winner中排除
        self.fatigue_neurons = []
        self.name = name
        self.n = n
        self.k = k
        # Default beta
        self.beta = beta
        # Betas from stimuli into this area.
        self.stimulus_beta = {}
        # Betas from areas into this area.
        self.area_beta = {}
        self.w = 0
        # List of winners currently (after previous action). Can be
        # read by caller.
        self.winners = []
        self.new_w = 0
        # new winners computed DURING a projection, do not use outside of internal project function
        self.new_winners = []
        # list of lists of all winners in each round
        self.saved_winners = []
        # list of size of support in each round
        self.saved_w = []
        self.num_first_winners = -1
        # Whether to fix (freeze) the assembly (winners) in this area
        self.fixed_assembly = False
        # Whether to fully simulate this area
        self.explicit = False

    def update_winners(self):
        self.winners = self.new_winners
        if not self.explicit:
            self.w = self.new_w

    def update_stimulus_beta(self, name, new_beta):
        self.stimulus_beta[name] = new_beta

    def update_area_beta(self, name, new_beta):
        self.area_beta[name] = new_beta

    def fix_assembly(self):
        if not self.winners:
            raise ValueError('Area %s does not have assembly; cannot fix.' % self.name)
            return
        self.fixed_assembly = True

    def unfix_assembly(self):
        self.fixed_assembly = False

    # 所有assembly操作都应该在Area内完成 因为涉及self
    # core_set support_set should be set object
    def add_assembly(self, name, core_set, support_set, father=None) -> Assembly:
        # if existed, should be remodel instead
        if name in self.assembly_list:
            self.remodel_assembly(name, core_set, support_set, father)
        new_assembly = Assembly(self.k, self.name, name, father, core_set, support_set)
        self.assembly_list[name] = new_assembly
        return new_assembly

    def remodel_assembly(self, name: str, core_set, support_set, father=None) -> Assembly:
        cur: Assembly = self.assembly_list[name]
        cur.core = core_set
        cur.support = support_set
        if father is not None:
            cur.father = father
        return cur

    # 用于新建随机assembly
    def add_random_assembly(self, name) -> Assembly:
        if name in self.assembly_list:
            return self.assembly_list[name]
        core_set = random.sample(range(0, self.n), self.k)
        support_set = core_set
        return self.add_assembly(name, core_set, support_set)

    # Todo: debug, and try method without project_into
    # 可以通过一个刺激得到这个刺激激活的神经元集所代表的词语
    def search_assembly(self, word_stimulu):
        # 首先通过词语在感官刺激区的assembly，这里的Stimulu是类似于LEX的一个脑区
        # 可以由此得到：word_stimulu = b.activeWord(Stimulu, word_name)
        area = self.name
        # 将刺激投影到该脑区得到这个词语激活的神经元集
        # word_assembly_core, word_assembly_support = word_assembly[0], word_assembly[1]
        # word_assembly_core.sort()
        # counter = 0
        # for word_ in self.assembly_list:
        #     for i in word_assembly_core:
        #         for j in self.assembly_list[word_][0]:
        #             if i == j:
        #                 counter = counter + 1
        #                 break
        #     if counter == len(word_assembly_core):
        #         return word_
        return None

    def search_core_first(self, winners):
        heap = PriorityQueue()
        for key, cur_ass in self.assembly_list.items():
            # cur_ass = item[1]
            core_sim = overlap_on_core(cur_ass.core, winners)
            support_sim = overlap_on_core(winners, cur_ass.support)
            if core_sim > self.similarity_threshold:
                heap.push(key, support_sim)
        ret = heap.pop()
        # return ret if ret is not None else "no matched assembly"
        return ret

    # Todo: complete it
    # 根据最新的winners判断激活的assembly, 空时为None
    def readout(self):
        last_winners = self.winners
        return self.search_core_first(last_winners)

    def readout_str(self):
        ret = self.readout()
        return ret if ret is not None else "no matched assembly"


class Brain:
    # we need to save winners to record the core set and support set
    def __init__(self, p, save_size=True, save_winners=False, train_times=20):
        self.areas: dict[str:Area] = {}
        self.stimuli: dict[str:Stimulus] = {}
        self.stimuli_connectomes = {}
        self.connectomes = {}
        self.p = p
        self.train_times = train_times
        self.save_size = save_size
        self.save_winners = save_winners
        # For debugging purposes in applications (eg. language)
        self.no_plasticity = False
        # 单突触权重上限
        self.MAX_SYNAPSE_POWER = 10

    def add_stimulus(self, name, k):
        self.stimuli[name] = Stimulus(k, name)
        new_connectomes = {}
        for key in self.areas:
            if self.areas[key].explicit:
                new_connectomes[key] = np.random.binomial(k, self.p, size=(self.areas[key].n)) * 1.0
            else:
                new_connectomes[key] = np.empty(0)
            self.areas[key].stimulus_beta[name] = self.areas[key].beta
        self.stimuli_connectomes[name] = new_connectomes

    def add_area(self, name, n, k, beta):
        self.areas[name] = Area(name, n, k, beta)

        for stim_name, stim_connectomes in self.stimuli_connectomes.items():
            stim_connectomes[name] = np.empty(0)
            self.areas[name].stimulus_beta[stim_name] = beta

        new_connectomes = {}
        for key in self.areas:
            other_area_size = 0
            if self.areas[key].explicit:
                other_area_size = self.areas[key].n
            new_connectomes[key] = np.empty((0, other_area_size))
            if key != name:
                self.connectomes[key][name] = np.empty((other_area_size, 0))
            self.areas[key].area_beta[name] = self.areas[key].beta
            self.areas[name].area_beta[key] = beta
        self.connectomes[name] = new_connectomes

        return self.areas[name]

    def add_explicit_area(self, name, n, k, beta):
        self.areas[name] = Area(name, n, k, beta)
        self.areas[name].explicit = True

        for stim_name, stim_connectomes in self.stimuli_connectomes.items():
            stim_connectomes[name] = np.random.binomial(self.stimuli[stim_name].k, self.p, size=(n)) * 1.0
            self.areas[name].stimulus_beta[stim_name] = beta

        new_connectomes = {}
        for key in self.areas:
            if key == name:  # create explicitly
                new_connectomes[key] = np.random.binomial(1, self.p, size=(n, n)) * 1.0
            if key != name:
                if self.areas[key].explicit:
                    other_n = self.areas[key].n
                    new_connectomes[key] = np.random.binomial(1, self.p, size=(n, other_n)) * 1.0
                    self.connectomes[key][name] = np.random.binomial(1, self.p, size=(other_n, n)) * 1.0
                else:  # we will fill these in on the fly
                    new_connectomes[key] = np.empty((n, 0))
                    self.connectomes[key][name] = np.empty((0, n))
            self.areas[key].area_beta[name] = self.areas[key].beta
            self.areas[name].area_beta[key] = beta
        self.connectomes[name] = new_connectomes
        # Explicitly set w to n so that all computations involving this area are explicit.
        self.areas[name].w = n

    def update_plasticities(self, area_update_map=None, stim_update_map=None):
        # area_update_map consists of area1: list[ (area2, new_beta) ]
        # represents new plasticity FROM area2 INTO area1
        if stim_update_map is None:
            stim_update_map = {}
        if area_update_map is None:
            area_update_map = {}
        for to_area, update_rules in area_update_map.items():
            for (from_area, new_beta) in update_rules:
                self.areas[to_area].area_beta[from_area] = new_beta

        # stim_update_map consists of area: list[ (stim, new_beta) ]f
        # represents new plasticity FROM stim INTO area
        for area, update_rules in stim_update_map.items():
            for (stim, new_beta) in update_rules:
                self.areas[area].stimulus_beta[stim] = new_beta

    def project(self, stim_to_area, area_to_area, verbose=False):
        # Validate stim_area, area_area well defined
        # stim_to_area: {"stim1":["A"], "stim2":["C","A"]}
        # area_to_area: {"A":["A","B"],"C":["C","A"]}

        stim_in = defaultdict(lambda: [])
        area_in = defaultdict(lambda: [])

        for stim, areas in stim_to_area.items():
            if stim not in self.stimuli:
                raise IndexError(stim + " not in brain.stimuli")
                return
            for area in areas:
                if area not in self.areas:
                    raise IndexError(area + " not in brain.areas")
                    return
                stim_in[area].append(stim)
        for from_area, to_areas in area_to_area.items():
            if from_area not in self.areas:
                raise IndexError(from_area + " not in brain.areas")
                return
            for to_area in to_areas:
                if to_area not in self.areas:
                    raise IndexError(to_area + " not in brain.areas")
                    return
                area_in[to_area].append(from_area)

        to_update = set().union(stim_in.keys(), area_in.keys())

        for area in to_update:
            num_first_winners = self.project_into(self.areas[area], stim_in[area], area_in[area], verbose)
            self.areas[area].num_first_winners = num_first_winners
            if self.save_winners:
                self.areas[area].saved_winners.append(self.areas[area].new_winners)

        # once done everything, for each area in to_update: area.update_winners()
        for area in to_update:
            self.areas[area].update_winners()
            if self.save_size:
                self.areas[area].saved_w.append(self.areas[area].w)

    def project_into(self, area, from_stimuli, from_areas, verbose=False):
        # projecting everything in from stim_in[area] and area_in[area]
        # calculate: inputs to self.connectomes[area] (previous winners)
        # calculate: potential new winners, Binomial(sum of in sizes, k-top)
        # k top of previous winners and potential new winners
        # if new winners > 0, redo connectome and intra_connectomes
        # have to wait to replace new_winners
        print("Projecting " + ",".join(from_stimuli) + " and " + ",".join(from_areas) + " into " + area.name)

        # If projecting from area with no assembly, complain.
        for from_area in from_areas:
            if not self.areas[from_area].winners or (self.areas[from_area].w == 0):
                raise Exception("Projecting from area with no assembly: " + from_area)

        name = area.name
        prev_winner_inputs = [0.] * area.w
        for stim in from_stimuli:
            stim_inputs = self.stimuli_connectomes[stim][name]
            for i in range(area.w):
                prev_winner_inputs[i] += stim_inputs[i]
        for from_area in from_areas:
            connectome = self.connectomes[from_area][name]
            for w in self.areas[from_area].winners:
                for i in range(area.w):
                    prev_winner_inputs[i] += connectome[w][i]

        if verbose:
            print("prev_winner_inputs: ")
            print(prev_winner_inputs)

        # simulate area.k potential new winners if the area is not explicit
        if not area.explicit:
            total_k = 0
            input_sizes = []
            num_inputs = 0
            for stim in from_stimuli:
                total_k += self.stimuli[stim].k
                input_sizes.append(self.stimuli[stim].k)
                num_inputs += 1
            for from_area in from_areas:
                # if self.areas[from_area].w < self.areas[from_area].k:
                #	raise ValueError("Area " + from_area + "does not have enough support.")
                effective_k = len(self.areas[from_area].winners)
                total_k += effective_k
                input_sizes.append(effective_k)
                num_inputs += 1

            if verbose:
                print("total_k = " + str(total_k) + " and input_sizes = " + str(input_sizes))

            effective_n = area.n - area.w
            # Threshold for inputs that are above (n-k)/n percentile.
            # self.p can be changed to have a custom connectivity into this brain area.
            alpha = binom.ppf((float(effective_n - area.k) / effective_n), total_k, self.p)
            if verbose:
                print("Alpha = " + str(alpha))
            # use normal approximation, between alpha and total_k, round to integer
            # create k potential_new_winners
            std = math.sqrt(total_k * self.p * (1.0 - self.p))
            mu = total_k * self.p
            a = float(alpha - mu) / std
            b = float(total_k - mu) / std
            potential_new_winners = truncnorm.rvs(a, b, scale=std, size=area.k)
            for i in range(area.k):
                potential_new_winners[i] += mu
                potential_new_winners[i] = round(potential_new_winners[i])
            potential_new_winners = potential_new_winners.tolist()

            if verbose:
                print("potential_new_winners: ")
                print(potential_new_winners)

            # take max among prev_winner_inputs, potential_new_winners
            # get num_first_winners (think something small)
            # can generate area.new_winners, note the new indices
            all_potential_winners = prev_winner_inputs + potential_new_winners
        else:
            all_potential_winners = prev_winner_inputs

        new_winner_indices = heapq.nlargest(area.k, range(len(all_potential_winners)),
                                            all_potential_winners.__getitem__)
        num_first_winners = 0

        if not area.explicit:
            first_winner_inputs = []
            for i in range(area.k):
                if new_winner_indices[i] >= area.w:
                    first_winner_inputs.append(potential_new_winners[new_winner_indices[i] - area.w])
                    new_winner_indices[i] = area.w + num_first_winners
                    num_first_winners += 1
        area.new_winners = new_winner_indices
        area.new_w = area.w + num_first_winners

        # For experiments with a "fixed" assembly in some area.
        if area.fixed_assembly:
            area.new_winners = area.winners
            area.new_w = area.w
            first_winner_inputs = []
            num_first_winners = 0

        # print name + " num_first_winners = " + str(num_first_winners)

        if verbose:
            print("new_winners: ")
            print(area.new_winners)

        # for i in num_first_winners
        # generate where input came from
        # 1) can sample input from array of size total_k, use ranges
        # 2) can use stars/stripes method: if m total inputs, sample (m-1) out of total_k
        first_winner_to_inputs = {}
        for i in range(num_first_winners):
            input_indices = random.sample(range(0, total_k), int(first_winner_inputs[i]))
            inputs = np.zeros(num_inputs)
            total_so_far = 0
            for j in range(num_inputs):
                inputs[j] = sum([((total_so_far + input_sizes[j]) > w >= total_so_far) for w in input_indices])
                total_so_far += input_sizes[j]
            first_winner_to_inputs[i] = inputs
            if verbose:
                print("for first_winner # " + str(i) + " with input " + str(first_winner_inputs[i]) + " split as so: ")
                print(inputs)

        m = 0
        # connectome for each stim->area
        # add num_first_winners cells, sampled input * (1+beta)
        # for i in repeat_winners, stimulus_inputs[i] *= (1+beta)
        for stim in from_stimuli:
            if num_first_winners > 0:
                self.stimuli_connectomes[stim][name] = np.resize(self.stimuli_connectomes[stim][name],
                                                                 area.w + num_first_winners)
            for i in range(num_first_winners):
                self.stimuli_connectomes[stim][name][area.w + i] = first_winner_to_inputs[i][m]
            stim_to_area_beta = area.stimulus_beta[stim]
            if self.no_plasticity:
                stim_to_area_beta = 0.0
            for i in area.new_winners:
                self.stimuli_connectomes[stim][name][i] *= (1 + stim_to_area_beta)
            if verbose:
                print(stim + " now looks like: ")
                print(self.stimuli_connectomes[stim][name])
            m += 1

        # !!!!!!!!!!!!!!!!
        # BIG TO DO: Need to update connectomes for stim that are NOT in from_stimuli
        # For example, if last round fired areas A->B, and stim has never been fired into B.

        # connectome for each in_area->area
        # add num_first_winners columns
        # for each i in num_first_winners, fill in (1+beta) for chosen neurons
        # for each i in repeat_winners, for j in in_area.winners, connectome[j][i] *= (1+beta)
        for from_area in from_areas:
            from_area_w = self.areas[from_area].w
            from_area_winners = self.areas[from_area].winners
            self.connectomes[from_area][name] = np.pad(self.connectomes[from_area][name],
                                                       ((0, 0), (0, num_first_winners)), 'constant', constant_values=0)
            for i in range(num_first_winners):
                total_in = first_winner_to_inputs[i][m]
                sample_indices = random.sample(from_area_winners, int(total_in))
                for j in range(from_area_w):
                    if j in sample_indices:
                        self.connectomes[from_area][name][j][area.w + i] = 1.0
                    if j not in from_area_winners:
                        self.connectomes[from_area][name][j][area.w + i] = np.random.binomial(1, self.p)
            area_to_area_beta = area.area_beta[from_area]
            if self.no_plasticity:
                area_to_area_beta = 0.0
            for i in area.new_winners:
                for j in from_area_winners:
                    # 大于阈值（单突触权重上限）直接跳过
                    if self.connectomes[from_area][name][j][i] >= self.MAX_SYNAPSE_POWER:
                        continue
                    else:
                        self.connectomes[from_area][name][j][i] = min(
                            self.connectomes[from_area][name][j][i] * (1.0 + area_to_area_beta), self.MAX_SYNAPSE_POWER)
                # self.connectomes[from_area][name][j][i] *= (1.0 + area_to_area_beta)
            if verbose:
                print("Connectome of " + from_area + " to " + name + " is now:")
                print(self.connectomes[from_area][name])
            m += 1

        # expand connectomes from other areas that did not fire into area
        # also expand connectome for area->other_area
        for other_area in self.areas:
            if other_area not in from_areas:
                self.connectomes[other_area][name] = np.pad(self.connectomes[other_area][name],
                                                            ((0, 0), (0, num_first_winners)), 'constant',
                                                            constant_values=0)
                for j in range(self.areas[other_area].w):
                    for i in range(area.w, area.new_w):
                        self.connectomes[other_area][name][j][i] = np.random.binomial(1, self.p)
            # add num_first_winners rows, all bernoulli with probability p
            self.connectomes[name][other_area] = np.pad(self.connectomes[name][other_area],
                                                        ((0, num_first_winners), (0, 0)), 'constant', constant_values=0)
            columns = (self.connectomes[name][other_area]).shape[1]
            for i in range(area.w, area.new_w):
                for j in range(columns):
                    self.connectomes[name][other_area][i][j] = np.random.binomial(1, self.p)
            if verbose:
                print("Connectome of " + name + " to " + other_area + " is now:")
                print(self.connectomes[name][other_area])

        return num_first_winners

    # Todo: atomic functions, like "projection", "reciprocal_projection", "merge", associate, coProjection,
    #  multi_stimuli_projection

    # other than project_into, it project existed assembly x in area A to area B to generate new assembly y
    def projection(self, from_area: str, to_area: str, new_name: str, father_assembly=None, father_winners=None) \
            -> Assembly:
        father_name: str = father_assembly
        src: Area = self.areas[from_area]
        des: Area = self.areas[to_area]
        # only one of father_assembly and father_winner will be considered at once
        if father_assembly is not None:
            # in this case, we should generate winners like a stimulus, with all core elements selected
            father: Assembly = src.assembly_list[father_assembly]
            prepare_area(src, father.core, father.support)
        else:
            # in this case, former winners is known
            father_name = src.readout()
            src.winners = father_winners
            src.fix_assembly()

        self.project({}, {src.name: [des.name]})
        new_core = set(des.new_winners)
        new_support = new_core
        for step in range(self.train_times):
            # recurrent projection is still needed
            self.project({}, {src.name: [des.name], des.name: [des.name]})
            new_core = new_core.intersection(set(des.new_winners))
            new_support = new_support.union(set(des.new_winners))
            # weight test
            self.weights_count_test(step, to_area, des.new_winners)
        #     clear, for later projections
        if self.save_winners:
            des.saved_winners = []

        # generate new assembly
        return des.add_assembly(new_name, new_core, new_support, father=[father_name])

    def weights_count_test(self, step, area_name, winners: list):
        weights = self.connectomes[area_name][area_name]
        # 计算神经元的平均权重和
        nw_sum = 0
        size = len(winners)
        for i in winners:
            line = weights[i]
            for j in line:
                nw_sum += j
        average_neuron_sum_of_weights = nw_sum / size

        # 计算突触的平均权重
        size = 0
        sw_sum = 0
        for i in weights:
            for j in i:
                if j > 1.01:
                    size += 1
                sw_sum += j

        average_synaptic_weight = sw_sum / size
        with open("weight_test.csv", "a") as f:
            f.write(str(step) + ', ' + str(average_neuron_sum_of_weights) + ', ' + str(average_synaptic_weight) + '\n')
            # f.write("average neurons' sum of weights: " + str(average_neuron_sum_of_weights) + '\n')
            # f.write("average synaptic weight: " + str(average_synaptic_weight) + '\n')

    # reciprocal means the father assembly will also be remodelled
    # 双向和merge一样需要递归
    def reciprocal_projection(self, from_area: str, to_area: str, new_name: str, father_assembly=None,
                              father_winners=None) -> Assembly:
        father_name: str = father_assembly
        src: Area = self.areas[from_area]
        des: Area = self.areas[to_area]
        # only one of father_assembly and father_winner will be considered at once
        if father_assembly is not None:
            # in this case, we should generate winners like a stimulus, with all core elements selected
            father: Assembly = src.assembly_list[father_assembly]
            prepare_area(src, father.core, father.support)
        else:
            # in this case, former winners is known
            father_name = src.readout()
            src.winners = father_winners
            src.fix_assembly()

        self.project({}, {src.name: [des.name]})
        new_core = set(des.new_winners)
        new_support = new_core

        for _ in range(self.train_times):
            self.project({}, {src.name: [des.name, src.name], des.name: [des.name, src.name]})
            new_core = new_core.intersection(set(des.new_winners))
            new_support = new_support.union(set(des.new_winners))

        father: Assembly = save_last_assembly(self, from_area, father_name)

        #     clear, for later projections
        if self.save_winners:
            des.saved_winners = []

        # generate new assembly
        return des.add_assembly(new_name, new_core, new_support, father=[father_name])

    # z = merge(x,y)
    # 其实问题很大 父亲也有父亲，而重塑父亲的过程中会导致祖父无法激活父亲，这样的关系可以递归下去，所以我们需要递归发现投影树
    def merge_by_assembly(self, parent_area_1, parent_area_2, to_area, new_name, parent_assembly_1, parent_assembly_2) \
            -> Assembly:
        src1: Area = self.areas[parent_area_1]
        src2: Area = self.areas[parent_area_2]
        des: Area = self.areas[to_area]

        parent1: Assembly = src1.assembly_list[parent_assembly_1]
        parent2: Assembly = src2.assembly_list[parent_assembly_2]

        prepare_area(src1, parent1.core, parent1.support)
        prepare_area(src2, parent2.core, parent2.support)

        self.project({}, {parent_area_1: [parent_area_1, to_area], parent_area_2: [parent_area_2, to_area]})
        new_core = set(des.new_winners)
        new_support = new_core

        for _ in range(self.train_times):
            self.project({}, {parent_area_1: [parent_area_1, to_area], parent_area_2: [parent_area_2, to_area],
                              to_area: [parent_area_1, parent_area_2]})
            new_core = new_core.intersection(set(des.new_winners))
            new_support = new_support.union(set(des.new_winners))

        parent1 = save_last_assembly(self, parent_area_1, parent_assembly_1)
        parent2 = save_last_assembly(self, parent_area_2, parent_assembly_2)

        if self.save_winners:
            des.saved_winners = []

        print(new_name)
        return des.add_assembly(new_name, new_core, new_support, father=[parent_assembly_1, parent_assembly_2])

    # much easier, because we don't need to find out the grandparents, grand-grandparents......
    def merge_by_winners(self, parent_area_1, parent_area_2, to_area, new_name, winners1, winners2) -> Assembly:
        src1 = self.areas[parent_area_1]
        src2 = self.areas[parent_area_2]
        des = self.areas[to_area]

        father_name1 = src1.readout()
        src1.winners = winners1
        src1.fix_assembly()

        father_name2 = src2.readout()
        src2.winners = winners2
        src2.fix_assembly()

        self.project({}, {parent_area_1: [parent_area_1, to_area], parent_area_2: [parent_area_2, to_area]})
        new_core = set(des.new_winners)
        new_support = new_core

        for _ in range(self.train_times):
            self.project({}, {parent_area_1: [parent_area_1, to_area], parent_area_2: [parent_area_2, to_area],
                              to_area: [parent_area_1, parent_area_2]})
            new_core = new_core.intersection(set(des.new_winners))
            new_support = new_support.union(set(des.new_winners))

        parent1 = save_last_assembly(self, parent_area_1, father_name1)
        parent2 = save_last_assembly(self, parent_area_2, father_name2)

        if self.save_winners:
            des.saved_winners = []

        return des.add_assembly(new_name, new_core, new_support, father=[father_name1, father_name2])

    # association will influence the assembly reacting to single stimulus, so we should renew it
    # according to AC's original design, association describe a process of
    def association(self, parent_area_1, parent_area_2, to_area, name1, name2, parent_assembly_1, parent_assembly_2):
        src1 = self.areas[parent_area_1]
        src2 = self.areas[parent_area_2]
        des = self.areas[to_area]

        # ensure there are two assemblies on to_area, if existed they will be remodeled
        new_assembly1 = self.projection(parent_area_1, to_area, name1, father_assembly=parent_assembly_1)
        new_assembly2 = self.projection(parent_area_2, to_area, name2, father_assembly=parent_assembly_2)

        # test computing, can be commented out
        overlap_core_1 = jaccard_similarity(new_assembly1.core, new_assembly2.core)
        overlap_support_1 = jaccard_similarity(new_assembly1.support, new_assembly2.support)

        parent1 = src1.assembly_list[parent_assembly_1]
        parent2 = src2.assembly_list[parent_assembly_2]

        prepare_area(src1, parent1.core, parent1.support)
        prepare_area(src2, parent2.core, parent2.support)

        self.project({}, {parent_area_1: [parent_area_1, to_area], parent_area_2: [parent_area_2, to_area]})

        for _ in range(self.train_times):
            self.project({}, {parent_area_1: [parent_area_1, to_area], parent_area_2: [parent_area_2, to_area],
                              to_area: [to_area]})

        # parents assemblies wouldn't change, but we can only read children assemblies one by one
        self.no_plasticity = True
        prepare_area(src1, parent1.core, parent1.support)
        for _ in range(5):
            self.project({}, {parent_area_1: [parent_area_1, to_area]})
        new_assembly1 = save_last_assembly(self, to_area, name1)
        prepare_area(src2, parent2.core, parent2.support)
        for _ in range(5):
            self.project({}, {parent_area_2: [parent_area_2, to_area]})
        new_assembly2 = save_last_assembly(self, to_area, name2)

        # test computing, can be commented out
        overlap_core_2 = jaccard_similarity(new_assembly1.core, new_assembly2.core)
        overlap_support_2 = jaccard_similarity(new_assembly1.support, new_assembly2.support)

        print("the overlap before is " + str(overlap_core_1) + " on core and " + str(overlap_support_1) + " on support")
        print("the overlap after is " + str(overlap_core_2) + " on core and " + str(overlap_support_2) + " on support")

        if self.save_winners:
            des.saved_winners = []
            src1.saved_winners = []
            src2.saved_winners = []

        self.no_plasticity = False

        return new_assembly1, new_assembly2

    # stimA: A case
    def stimulus_projection(self, to_area, stimulus_name, new_assembly_name, n=100000, k=317, beta=0.05) -> Assembly:
        if to_area not in self.areas:
            self.add_area(to_area, n=n, k=k, beta=beta)

        area: Area = self.areas[to_area]

        if stimulus_name not in self.stimuli:
            self.add_stimulus(stimulus_name, k=k)
        stimulus: Stimulus = self.stimuli[stimulus_name]

        self.project({stimulus_name: [to_area]}, {})
        new_core = set(area.new_winners)
        new_support = new_core
        for _ in range(self.train_times):
            self.project({stimulus_name: [to_area]}, {})
        new_core = new_core.intersection(set(area.new_winners))
        new_support = new_support.union(set(area.new_winners))

        if self.save_winners:
            area.saved_winners = []

        return area.add_assembly(new_assembly_name, new_core, new_support, father=[stimulus_name])

    # TODO: undone
    def co_projection(self, from_area, to_area, new_name, father_assembly=None, father_winners=None):
        father_name = father_assembly
        src = self.areas[from_area]
        des = self.areas[to_area]
        k = des.k
        # only one of father_assembly and father_winner will be considered at once
        if father_assembly is not None:
            # in this case, we should generate winners like a stimulus, with all core elements selected
            father = src.assembly_list[father_assembly]
            prepare_area(src, father.core, father.support)
        else:
            # in this case, former winners is known
            father_name = src.readout()
            src.winners = father_winners
            src.fix_assembly()

        self.project({}, {src.name: [des.name]})
        new_core = set(des.new_winners)
        new_support = new_core

        for _ in range(self.train_times):
            self.project({}, {src.name: [des.name, src.name], des.name: [des.name, src.name]})
            new_core = new_core.intersection(set(des.new_winners))
            new_support = new_support.union(set(des.new_winners))

        save_last_assembly(self, from_area, father_name)

        #     clear, for later projections
        if self.save_winners:
            des.saved_winners = []

        # generate new assembly
        return des.add_assembly(new_name, new_core, new_support, father=[father_name])


# save last winners as assembly
def save_last_assembly(brain, area_name, assembly_name) -> Assembly:
    all_winners = brain.areas[area_name].saved_winners
    # 最后一次的winners没保存，直接加载
    core = set(brain.areas[area_name].winners)
    support = set([])
    for winner in all_winners:
        core = core.intersection(set(winner))
        support = support.union(set(winner))
    new_assembly: Assembly = brain.areas[area_name].add_assembly(name=assembly_name, core_set=core, support_set=support)
    if brain.save_winners:
        brain.areas[area_name].saved_winners = []
    return new_assembly


def prepare_area(area, _core, _support):
    core = set(_core)
    support = set(_support) - core
    support = list(support)
    winners = _core
    if len(winners) < area.k:
        random.shuffle(support)
        # randomly select elements from support to fulfill the k winners
        winners = winners.union(set(support[0:(area.k - len(winners))]))
    area.winners = winners
    if area.w == 0:
        area.w = len(winners)
    area.fix_assembly()

def parseHelper(sentence):
    word_list  = sentence.split(" ")
