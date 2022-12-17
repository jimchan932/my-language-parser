from collections import defaultdict
from random import shuffle

from modified_brain import Brain, Area

# 我的命名逻辑：非终止符有谓词（pred）和指称词（deno）两种, 谓词根据变元数不同, 分为0, 1, 2, 3元等……
# 一个n元谓词可以填入一个指称词变为n-1元谓词
# 你可以有不同的命名逻辑, 如VP（替代谓词）, NP（替代指称词）等
# Area for stage 2 (components)
START = "START"
DET = "DET"
SUBJ = "SUBJ"
OBJ = "OBJ"
DENO = "DENO"
ADJNOUN = "ADJNOUN"
ADJ = "ADJ"
ADV = "ADV"
PREP = "PREP"
TRANSITIVEVERB = "TRANSITIVEVERB"
INTRANSITIVEVERB = "INTRANSITIVEVERB"
VERBPHRASE = "VERBPHRASE"

# 全局最深的pred层数
max_pred_level = 3

# stage2
stage_2_Areas = {
    START: {}, 
    DET: {},    
    SUBJ: {},
    OBJ: {},
    DENO: {},
    ADJNOUN: {},
    ADJ: {},
    ADV: {},
    PREP: {},
    TRANSITIVEVERB: {},
    INTRANSITIVEVERB: {},
    VERBPHRASE: {},
    PREP: {},
}


# 取幂集, 生成正则表达式测试用例用的
def power_set(set1):
    ans = [[]]
    for i in set1:
        l = len(ans)
        for j in range(l):
            t = []
            t.extend(ans[j])
            t.append(i)
            ans.append(t)
    return ans

class ParserBrain(Brain):
    def __init__(self, p, lexeme_dict=None, all_areas=None, recurrent_areas=None, initial_areas=None,
                 readout_rules=None):
        Brain.__init__(self, p)

        if readout_rules is None:
            readout_rules = {}
        if initial_areas is None:
            initial_areas = []
        if recurrent_areas is None:
            recurrent_areas = []
        if all_areas is None:
            all_areas = []
            
        self.component_dict = {}
        self.all_areas = all_areas
        self.recurrent_areas = recurrent_areas
        self.initial_areas = initial_areas

        self.fiber_states = defaultdict()
        self.area_states = defaultdict(set)
        self.activated_fibers = defaultdict(set)
        self.readout_rules = readout_rules
        self.parse_tree = []
        # self.initialize_states()

        self.lexeme_dict = lexeme_dict
        if lexeme_dict is None:
            self.generate_components()

    def parse(self, sentence, n=1000, k=30, beta=0.1):    
        word_list = sentence.split(' ')
        parse_tree_word_list = []
        num_words = len(word_list)
        bottom_level_word_list = []
        for word in word_list:            
            bottom_level_word_list.append((word, self.component_dict[word].area_name))
        # parse_tree_word_list[0] = X11 X22 ... Xnn
        parse_tree_word_list.append(bottom_level_word_list)
        print(bottom_level_word_list)
        # X0 to num_words-1
        # left part: Xi,k  right part: X(k+1),j

        for phrase_len in range(1, num_words):
            last_start_index = num_words - phrase_len
            same_level_word_list = []
            for start_index in range(0, last_start_index):
                parent_variable = None
                last_index = start_index + phrase_len
                for index_gap in range(0, phrase_len): # looping the middle index 
                    remainder_index_difference = phrase_len - index_gap - 1
                    if (parse_tree_word_list[index_gap][start_index] == None or
                        parse_tree_word_list[remainder_index_difference][last_index-remainder_index_difference] == None):
                        continue
                    left_child_assembly_name, left_child_area_name = parse_tree_word_list[index_gap][start_index]
                    right_child_assembly_name, right_child_area_name = parse_tree_word_list[phrase_len - index_gap-1][last_index-remainder_index_difference]
                    left_child_word_type = left_child_area_name.split('_')[0]
                    right_child_word_type = right_child_area_name.split('_')[0]
                    result_assembly_name = left_child_assembly_name + ' ' + right_child_assembly_name                
                    if(left_child_word_type == ADJ and right_child_word_type == ADJ):  # ADJ*
                        if(left_child_assembly_name < right_child_assembly_name):                        
                            result_assembly_name = left_child_assembly_name + ' ' + right_child_assembly_name
                        else:
                            result_assembly_name = right_child_assembly_name + ' ' + left_child_assembly_name
                        parent_variable = (result_assembly_name, self.component_dict[result_assembly_name].area_name)
                        break
                    if(left_child_word_type == ADJ and right_child_word_type == DENO): # ADJNOUN
                        to_area_name = ADJNOUN + "_{}".format(len(stage_2_Areas[ADJNOUN]))
                        stage_2_Areas[ADJNOUN].setdefault(len(stage_2_Areas[ADJNOUN]), to_area_name)
                        self.add_explicit_area(to_area_name, n, k, beta)
                        self.all_areas.append(to_area_name)         
                        result_assembly = self.merge_by_assembly(left_child_area_name, right_child_area_name,
                                                            to_area_name, result_assembly_name,
                                                            left_child_assembly_name, right_child_assembly_name)
                        parent_variable = (result_assembly_name, to_area_name)
                        break
                    if(left_child_word_type == DET and right_child_word_type == ADJNOUN): # SUBJ
                        print("Testing")
                        to_area_name = SUBJ + "_{}".format(len(stage_2_Areas[SUBJ]))
                        stage_2_Areas[SUBJ].setdefault(len(stage_2_Areas[SUBJ]), to_area_name)
                        self.add_explicit_area(to_area_name, n, k, beta)
                        self.all_areas.append(to_area_name)         
                        result_assembly = self.merge_by_assembly(left_child_area_name, right_child_area_name,
                                                            to_area_name, result_assembly_name,
                                                            left_child_assembly_name, right_child_assembly_name)
                        parent_variable = (result_assembly_name, to_area_name)
                        break
                    if(left_child_word_type == TRANSITIVEVERB and right_child_word_type == SUBJ
                           or left_child_word_type == INTRANSITIVEVERB and right_child_word_type == ADV): # VERBPHRASE
                        to_area_name = VERBPHRASE + "_{}".format(len(stage_2_Areas[VERBPHRASE]))
                        stage_2_Areas[VERBPHRASE].setdefault(len(stage_2_Areas[VERBPHRASE]), to_area_name)
                        self.add_explicit_area(to_area_name, n, k, beta)
                        self.all_areas.append(to_area_name)         
                        result_assembly = self.merge_by_assembly(left_child_area_name, right_child_area_name,
                                                            to_area_name, result_assembly_name,
                                                            left_child_assembly_name, right_child_assembly_name)
                        parent_variable = (result_assembly_name, to_area_name)
                        break
                    if(left_child_word_type == TRANSITIVEVERB and right_child_word_type == PREP): # TRANSITIVEVERB 
                        to_area_name = TRANSITIVEVERB + "_{}".format(len(stage_2_Areas[VERBPHRASE]))
                        stage_2_Areas[TRANSITIVEVERB].setdefault(len(stage_2_Areas[TRANSITIVEVERB]), to_area_name)
                        self.add_explicit_area(to_area_name, n, k, beta)
                        self.all_areas.append(to_area_name)         
                        result_assembly = self.merge_by_assembly(left_child_area_name, right_child_area_name,
                                                            to_area_name, result_assembly_name,
                                                            left_child_assembly_name, right_child_assembly_name)
                        parent_variable = (result_assembly_name, to_area_name)
                        break
                    if(left_child_word_type == SUBJ and right_child_word_type == VERBPHRASE): # START
                        to_area_name = START + "_{}".format(len(stage_2_Areas[VERBPHRASE]))                        
                        stage_2_Areas[START].setdefault(len(stage_2_Areas[START]), to_area_name)
                        self.add_explicit_area(to_area_name, n, k, beta)
                        self.all_areas.append(to_area_name)         
                        result_assembly = self.merge_by_assembly(left_child_area_name, right_child_area_name,
                                                            to_area_name, result_assembly_name,
                                                            left_child_assembly_name, right_child_assembly_name)
                        parent_variable = (result_assembly_name, to_area_name)
                        break
                same_level_word_list.append(parent_variable)               
            parse_tree_word_list.append(same_level_word_list)
        return parse_tree_word_list
    def generic_component(self, assembly_index, comp_name, n=1000, k=30, beta=0.1):
        area_name = comp_name + "_{}".format(len(stage_2_Areas[comp_name]))
        stage_2_Areas[comp_name].setdefault(len(stage_2_Areas[comp_name]), area_name)
        # 此时没有刺激, 只能从外部输入
        self.add_explicit_area(area_name, n, k, beta)
        area: Area = self.areas[area_name]
        # 返回的是Assembly对象
        return area.add_random_assembly(assembly_index)

    # 生成词库的方法
    # 可以参考original parser里的lexeme_dict
    def generate_components(self):
        # key: assembly_name
        COMP_DICT = {
            "the": self.generic_component("the", DET),
            "a": self.generic_component("a", DET),
            "my": self.generic_component("my", DET),
            "dog": self.generic_component("dog", DENO),
            "cat": self.generic_component("cat", DENO),
            "mice": self.generic_component("mice", DENO),
            "city": self.generic_component("city", DENO),
            "chase": self.generic_component("chase", TRANSITIVEVERB),
            "love": self.generic_component("love", TRANSITIVEVERB),
            "bite": self.generic_component("bite", TRANSITIVEVERB),
            "smart": self.generic_component("smart", ADJ),
            "beautiful": self.generic_component("beautiful", ADJ),
            "small": self.generic_component("small", ADJ),
            "big": self.generic_component("big", ADJ),
            "bad": self.generic_component("bad", ADJ),
            "run": self.generic_component("run", INTRANSITIVEVERB),
            "fly": self.generic_component("fly", INTRANSITIVEVERB),
            "quickly": self.generic_component("quickly", ADV),
            "rapidly": self.generic_component("rapidly", ADV),
            "usually": self.generic_component("usually", ADV),
            "I": self.generic_component("I", SUBJ),            
            "you": self.generic_component("you", SUBJ),
            "saw": self.generic_component("saw", TRANSITIVEVERB),
            "put": self.generic_component("put", TRANSITIVEVERB),
            "bring": self.generic_component("bring", TRANSITIVEVERB),
            "banana": self.generic_component("banana", DENO),
            "on": self.generic_component("on", PREP),
            "clothes": self.generic_component("clothes", DENO)
        }

        adjective_list = []
        for key, value in COMP_DICT.items():
            if value.area_name.split('_')[0] == "ADJ":
                adjective_list.append(key)

        adjective_permutation_list = power_set(adjective_list)        
        for adjective_permutation in adjective_permutation_list:
            adjective_permutation.sort()
            adjective_star =  ' '.join(adjective_permutation)           
            COMP_DICT.update({adjective_star : self.generic_component(adjective_star, ADJ)})
        self.component_dict = COMP_DICT

        for key, value in COMP_DICT.items():
            self.all_areas.append(value.area_name)
            print(value.area_name)
    
        

if __name__ == '__main__':
    pb = ParserBrain(0.01)
    sentence_1 = "the big cat chase a small mice"
    sentence_2 = "I put on my beautiful clothes"
    sentence_3 = "the smart dog run quickly"
    parse_tree_1 = pb.parse(sentence_1)
    parse_tree_2 = pb.parse(sentence_2)
    parse_tree_3 = pb.parse(sentence_3)
    
    print(sentence_1)
    for i in range(len(parse_tree_1)-1, -1, -1):        
        print(parse_tree_1[i])
    
    print(sentence_2)    
    for i in range(len(parse_tree_2)-1, -1, -1):        
        print(parse_tree_2[i])
    
    print(sentence_3)
    for i in range(len(parse_tree_3)-1, -1, -1):        
        print(parse_tree_3[i])        
    

    
        
