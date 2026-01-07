import easyocr
import json
from deepseek_api_calling import DPSKCalling
import re
import os
import cv2
from datetime import datetime
# from mtcnn import MTCNN
import dlib


class FilteringRuleConstruction:
    def __init__(self):
        # 初始化OCR引擎，支持多种语言
        self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)
        self.dpsk_calling = DPSKCalling()
        # Time of the tree/model storage
        current_time = datetime.now()
        self.time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

    def evaluate_specification(self, input_text, category_text):
        """
        本方法实现对某个元素的具体性检测，即如果某个元素描述的目标不够具体，如“历史事件”这种词，进行过滤。
        """
        system_cmd = "你是一个元素具体性检测器。"
        prompt_specification = \
                (f"请根据下列的输入的有害元素内容（以逗号分隔），分开每一个考虑，并进行筛选。\n"
                 f"最终，仅保留具有具体目标对象的元素，而非空洞的内容。同时，语句存在该目标元素，则可以独立产生对应的风险，。\n"
                 f"- 输入的有害元素如下：{input_text}：\n"
                 f"- 对应的风险信息如下：{category_text}"
                 f"输出结果包裹在标签<ret_elem></ret_elem>，输出的多个元素之间同样用逗号间隔。")
        filter_words_ret = self.dpsk_calling.create_response(system_cmd, prompt_specification)
        return filter_words_ret

    def generate_single_elem_rules(self, system_cmd, single_elem_rules, item_merge_feature, edge_list, category_list):
        """
        规则1生成器，规则1的字典：
            {"RuleType": "SingleElemRule", "Content": [{"RuleElem": "xxx", "RiskType": "xxx"}, ...]}
        """
        # print(item_merge_feature)
        if single_elem_rules is None:
            single_elem_rules = {}
        merged_text, single_feature_list = (item_merge_feature["trigger_feature"],
                                            item_merge_feature["original_features"])
        category_match_list = [item_edge["source"] for item_edge in edge_list
                               if item_edge['target'] == item_merge_feature["id"]]
        if len(category_match_list) == 0:
            return
        category_content = [item["content"] for item in category_list if item["id"] == category_match_list[0]]
        # print(category_content[0])
        prompt_rule_generator = \
            (f"请根据下列的模因检测的结果，构建一个规则元素列表。当前规则的描述如下：\n"
             f"规则类型1：单元素检测，一旦出现相关的关键词、人物画像、说明性文字，即返回存在风险。\n"
             f"对于这个规则，需要注意我们仅关注图像描述中存在的某一个元素，一旦存在该元素，即必然存在对应的风险。\n"
             f"在构建规则过程中，请考虑如下的信息：\n"
             f"- 合并的风险触发的总结：{merged_text}\n"
             f"- 单独的风险触发情况的总结（可能是一个列表）：{single_feature_list}\n"
             f"- 对应可能产生的风险类型：{category_content[0]}\n"
             f"在输出内容中，可能会有多个关键词信息，每个关键词都可能形成风险。"
             f"注意输出的关键词必须是简单且具体的，表示一个物件、一个事件、一个动作、一个元素等等，不应该是一个长短语。"
             f"输出的关键词数量也有限，小于等于3个，并且危害较大的元素排在前面，相对不那么直接的元素排在后面。\n"
             f"请在<single_elem></single_elem>输出结果，多个元素请用逗号间隔。")
        # print(prompt_rule_generator)
        gen_rule_elements = self.dpsk_calling.create_response(system_cmd, prompt_rule_generator)
        print(gen_rule_elements)
        res_detection_rule = re.search(r'<single_elem>(.*?)</single_elem>', gen_rule_elements, re.DOTALL)
        res_rule_text = res_detection_rule.group(1).strip() if res_detection_rule else "None_Feature"
        res_rule_filter = self.evaluate_specification(res_rule_text, category_content[0])
        print(res_rule_filter)
        res_filtered_rule = re.search(r'<ret_elem>(.*?)</ret_elem>', res_rule_filter, re.DOTALL)
        res_filtered_rule_text = res_filtered_rule.group(1).strip() if res_filtered_rule else "None_Feature"
        for res_filtered_item in res_filtered_rule_text.split(","):
            res_filtered_item = res_filtered_item.strip()
            if res_filtered_item in single_elem_rules.keys():
                # if category_content[0] not in single_elem_rules[res_filtered_item]:
                single_elem_rules[res_filtered_item].append(category_content[0])
            else:
                single_elem_rules[res_filtered_item] = [category_content[0]]
        print(single_elem_rules)
        rule_update_path = os.path.join(
            "/CVPR_Competition/Auto_Risk_Triggering_Extraction/TZGA_Deployment/",
            f"rule_path_{self.time_str}")
        if not os.path.exists(rule_update_path):
            os.makedirs(rule_update_path)
        with open(rule_update_path + "/single_elem_rules.jsonl", 'w', encoding='utf-8') as f:
            json.dump(single_elem_rules, f, indent=2, ensure_ascii=False)
        return single_elem_rules

    def read_graph_and_rule_construction(self, graph_path):
        """
        本方法负责通过模式，构建快筛规则。检测规则构建后，存放在本地的规则库中。
        - 规则类型1：单元素检测，一旦出现相关的关键词、人物画像、说明性文字，即返回存在风险。
        - 规则类型2：多元素检测，涉及多个元素之间同时满足条件，才能对应出现风险。
        - 规则类型3：多元素关系检测，存在多个元素，并且这些元素间具备一定关系，才能确定存在风险。
        - 规则类型4：语义条件检测，存在一定的语义关系，才能确定存在风险（待定，暂未实现）
        自动化生成规则的方式通过以下流程实现：
        1. 从图中读取merged_trigger_feature，得到合并的触发模式。
        2. 分上述规则类型阶段，构建列表字典，用来说明规则的出现，存放在规则目录下：
        （2.1）规则1的字典：
            {"RuleType": "SingleElemRule", "Content": [{"RuleElem": "xxx", "RiskType": "xxx"}, ...]}
        （2.2）规则2的字典：
            {"RuleType": "MultiElemRule", "Content": [{"RuleElem": ["xxx", "xxx", ...], "RiskType": "xxx"}, ...]}
        （2.3）规则3的字典：
            {"RuleType": "RelElemRule", "Content": [{"SourceElem": "xxx", "TargetElem": "xxx",
            "Relation": "xxx", "RiskType": "xxx"}, ...]}
        3. 将上述三个规则字典拼接成一个字典，即通过{"SingleElemRule": [xxx], "DoubleElemRule": [xxx], ...}
        """
        with open(graph_path, 'r', encoding='utf-8') as fin:
            data_dict = json.load(fin)
        merged_feature_list = [item for item in data_dict["nodes"] if item["type"] == "merged_trigger_feature"]
        edge_list = data_dict["edges"]
        category_list = [item for item in data_dict["nodes"] if item["type"] == "risk_type"]

        # print(f"{len(merged_feature_list)}\n{merged_feature_list}")
        system_cmd = "你是一个检测互联网模因图风险的规则生成器。"

        single_elem_rules, multi_elem_rule, rel_elem_rule = dict(), {}, {}
        for item_merge_feature in merged_feature_list:
            # 构建规则1：这里通过调用DeepSeek分析元素中存在的规则性知识，并进行增补。
            single_elem_rules = self.generate_single_elem_rules(system_cmd, single_elem_rules, item_merge_feature, edge_list, category_list)


if __name__ == "__main__":
    filter_rule_construct = FilteringRuleConstruction()
    filter_rule_construct.read_graph_and_rule_construction("/CVPR_Competition/Auto_Risk_Triggering_Extraction/TZGA_Deployment/Meme_Graph_TZGA/tzga_meme_risk_triggered_module_tree_llm_2025-10-10 17:59:29.json")
    # filter_rule_construct.ocr_fig_text_info_detection("/Harmful_Dataset/huzhouga_harmful_dataset/Dataset/test.jsonl")
