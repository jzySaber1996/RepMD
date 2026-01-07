import json
from collections import deque
from deepseek_api_calling import DPSKCalling
import re


class TreeNode:
    def __init__(self, node_id, content, trigger_feature, parent=None):
        self.id = node_id
        self.content = content
        self.trigger_feature = trigger_feature
        self.parent = parent
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)
        child_node.parent = self

    def get_id(self):
        return self.id

    def get_parent_ids(self):
        return self.parent

    def get_node_content(self):
        return self.content

    def get_trigger_feature(self):
        return self.trigger_feature

    def __repr__(self):
        return f"TreeNode(id={self.id}, content={self.content})"


class Tree:
    def __init__(self, root_id=None, root_content=None, trigger_feature=None):
        self.root = None
        self.nodes = {}
        if root_id is not None:
            self.root = TreeNode(root_id, root_content, trigger_feature)
            self.nodes[root_id] = self.root

    def add_node(self, parent_id, node_id, content, trigger_feature):
        if node_id in self.nodes:
            raise ValueError(f"Node ID {node_id} already exists")
        if parent_id not in self.nodes:
            raise ValueError(f"Parent ID {parent_id} not found")

        parent = self.nodes[parent_id]
        new_node = TreeNode(node_id, content, trigger_feature, parent)
        parent.add_child(new_node)
        self.nodes[node_id] = new_node
        return new_node

    def delete_node(self, node_id):
        if node_id not in self.nodes:
            raise ValueError(f"Node ID {node_id} not found")
        if node_id == self.root.id:
            self.root = None
            self.nodes = {}
            return

        node = self.nodes[node_id]
        if node.parent:
            node.parent.children.remove(node)

        # 递归删除所有子节点
        nodes_to_delete = self._get_subtree_nodes(node)
        for n in nodes_to_delete:
            del self.nodes[n.id]

    def update_node_content(self, node_id, new_content):
        if node_id not in self.nodes:
            raise ValueError(f"Node ID {node_id} not found")
        self.nodes[node_id].content = new_content

    def find_node(self, node_id):
        return self.nodes.get(node_id)

    def get_children(self, node_id):
        node = self.find_node(node_id)
        return node.children if node else []

    def get_parent(self, node_id):
        node = self.find_node(node_id)
        return node.parent if node else None

    def traverse_dfs(self, start_id=None, action=None):
        """深度优先遍历（递归实现）"""
        start = self.root if start_id is None else self.find_node(start_id)
        if not start:
            return

        if action:
            action(start)

        for child in start.children:
            self.traverse_dfs(child.id, action)

    def traverse_bfs(self, start_id=None, action=None):
        """广度优先遍历（队列实现）"""
        start = self.root if start_id is None else self.find_node(start_id)
        if not start:
            return

        queue = deque([start])
        while queue:
            node = queue.popleft()
            if action:
                action(node)
            queue.extend(node.children)
        return queue

    def to_dict(self):
        """将树转换为可序列化的字典结构"""
        if not self.root:
            return None

        def build_dict(node):
            print("Node_Tigger_Feature: ", node.trigger_feature)
            return {
                "id": node.id,
                "content": node.content,
                "trigger_feature": node.trigger_feature,
                "children": [build_dict(child) for child in node.children]
            }

        print(self.root)

        return build_dict(self.root)

    def save_to_json(self, filename, encoding="utf-8"):
        """保存树到JSON文件"""
        tree_dict = self.to_dict()
        if not tree_dict:
            raise ValueError("Cannot save an empty tree")

        print(tree_dict)
        print(f"saved_encoding: {encoding}")
        with open(filename, 'w', encoding=encoding) as f:
            json.dump(tree_dict, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_json(cls, filename):
        """从JSON文件加载树"""
        with open(filename, 'r', encoding='utf-8') as f:
            tree_dict = json.load(f)

        def build_tree(node_dict, parent=None):
            node = TreeNode(node_dict["id"], node_dict["content"], node_dict["trigger_feature"], parent)
            tree.nodes[node.id] = node

            for child_dict in node_dict.get("children", []):
                child_node = build_tree(child_dict, node)
                node.add_child(child_node)

            return node

        tree = cls()
        tree.root = build_tree(tree_dict)
        return tree

    def _get_subtree_nodes(self, node):
        """获取子树所有节点（包括当前节点）"""
        nodes = [node]
        for child in node.children:
            nodes.extend(self._get_subtree_nodes(child))
        return nodes

    # def check_update_node(self, node_content, feature_content):
    #     self.traverse_dfs()
    #     return

    def get_depth(self):
        """计算树的整体层数（深度）"""
        if not self.root:
            return 0

        max_depth = 0

        def update_depth(node, level):
            nonlocal max_depth
            if level > max_depth:
                max_depth = level
            for child in node.children:
                update_depth(child, level + 1)

        update_depth(self.root, 1)
        return max_depth


    def get_nodes_at_level(self, level):
        """获取指定层的所有节点"""
        if level < 1:
            raise ValueError("Level must be at least 1")

        if not self.root:
            return []

        result = []

        # 使用BFS遍历并记录每个节点的层级
        queue = deque([(self.root, 1)])
        while queue:
            node, current_level = queue.popleft()

            if current_level == level:
                result.append(node)
            elif current_level < level:  # 只有当前层级小于目标层级时才继续
                for child in node.children:
                    queue.append((child, current_level + 1))
        return result

    def __repr__(self):
        return f"Tree(root={self.root})"


class MemesTree:
    def __init__(self):
        self.type_id = 1
        self.sub_type_id = 1
        self.trig_feat_id = 1
        self.memes_tree = Tree("Root", "Memes Risk Detection", "")
        self.memes_tree.add_node("Root", "Type_1", "TestType", "")
        self.memes_tree.add_node("Type_1", "Sub_Type_1", "TestSubType", "")
        self.memes_tree.add_node("Sub_Type_1", "Trig_Feat_1", "", "TestTriggerFeature")

    def check_tree(self):
        tree_depth = self.memes_tree.get_depth()
        tree_nodes_at_level = self.memes_tree.get_nodes_at_level(2)
        print(f"Tree_nodes_at_level: {[tree_nodes_item.get_node_content() for tree_nodes_item in tree_nodes_at_level]}")
        print(f"tree_depth: {tree_depth}, tree_nodes_at_level: {tree_nodes_at_level[0].get_trigger_feature()}")
        # print(tree_traverse_result)

    def get_meme_tree(self):
        return self.memes_tree

    def show_meme_tree(self):
        return self.memes_tree.traverse_bfs(action=lambda node: print(f"Loaded: {node.id}"))

    def insert_node_with_info(self, parent_id, node_id, content, trigger_feature):
        self.memes_tree.add_node(parent_id, node_id, content, trigger_feature)

    def update_tree(self, content_candidate):
        dpsk_calling = DPSKCalling()
        system_cmd = "You are a analyzer for the memes' risks."

        """
        Check and update the 2nd Layer: Types.
        The update algorithm is as follows:
            1. Check if the new node could match the current nodes in the meme types.
            2. If the node does not match the current type, add a new type.
            3. If the node match the current type, record the matched type.
        """

        tree_nodes_type_layer = self.memes_tree.get_nodes_at_level(2)
        content_node_type_layer = [tree_nodes_item.get_node_content() for tree_nodes_item in tree_nodes_type_layer]
        prompt_type_summary = (f"In the following task, you need to decide what the type of the meme's harmful risk it belongs to."
                               f"The following is the list of current risk types: <type>{content_node_type_layer}</type>."
                               f"You need to decide what the current risk type it belongs to, "
                               f"based on the incorrect prediction cases, and the reason for that the model cannot identify the risk's triggering features <content>{content_candidate}</content>."
                               f"Then, output the result with the data wrapped by <output></output>."
                               f"Note that, the content will be two part. The first part is 'yes/no', "
                               f"'yes' means the content belongs to the previous types, "
                               f"'no' means the content does not belong to the current types."
                               f"The second part is the type. If the first is 'yes', output the type's name."
                               f"If 'no', summarize a new type's name."
                               f"The first&second part in the output results will be seperated by ','."
                               f"For example, <output>Yes, type_name</output>, <output>No, new_type_name</output>."
                               f"You should only output the results with no other information.")

        content_ret = dpsk_calling.create_response(system_cmd, prompt_type_summary)
        print(content_ret)
        type_ret_result = re.search(r'<output>(.*?)</output>', content_ret, re.DOTALL)
        type_ret_text = type_ret_result.group(1).strip() if type_ret_result else "no text found"
        print(type_ret_text)
        type_split_res = type_ret_text.split(",")
        sub_type_parent_id = ""
        if 'No' in type_split_res[0].strip():
            self.type_id += 1
            self.memes_tree.add_node("Root", f"Type_{self.type_id}", f"{type_split_res[1].strip()}", "")
            sub_type_parent_id = f"Type_{self.type_id}"
        elif 'Yes' in type_split_res[0].strip():
            if f"{type_split_res[1].strip()}" in content_node_type_layer:
                # 下面这行代码，直接匹配能match到的
                ids_match = [tree_nodes_item.get_id() for tree_nodes_item in tree_nodes_type_layer if tree_nodes_item.get_node_content() == f"{type_split_res[1].strip()}"]
                sub_type_parent_id = ids_match[0]
            else:
                return ""

        """
        Update the 3->n Layers: SubTypes
            1. Check if the current node in the subtype matches the XXX
            2. Check if the 
        """

        if sub_type_parent_id == "":
            return ""
        tree_nodes_sub_type_layer = self.memes_tree.get_nodes_at_level(3)
        nodes_matches_type_ids = [node_item for node_item in tree_nodes_sub_type_layer if node_item.get_parent_ids().get_id() == sub_type_parent_id]
        print(nodes_matches_type_ids)
        content_node_sub_type_layer = [tree_nodes_item.get_node_content() for tree_nodes_item in nodes_matches_type_ids]
        print(content_node_sub_type_layer)
        prompt_subtype_summary = (f"In the following task, you need to decide what the sub_type of the risk it belongs to."
                               f"The type of this meme is <type>{sub_type_parent_id}</type>."
                               f"The following is the list of current subtypes belongs to the type: <subtype>{content_node_sub_type_layer}</subtype>."
                               f"You need to decide whether the current subtype it belongs to,"
                               f"based on the incorrect prediction cases, and the reason for that the model cannot identify the risk's triggering features <content>{content_candidate}</content>."
                               f"Then, output the result with the data wrapped by <output></output>."
                               f"Note that, the content will be three parts. "
                               f"The first part is 'yes/no', 'yes' means the content belongs to the previous types, "
                               f"'no' means the content does not belong to the current types."
                               f"The second part is the subtype. If the first is 'yes', output the subtype's name."
                               f"If 'no', summarize a new subtype's name."
                               f"The third part is how the risk is triggered, and please summarize the steps to describe how the risks are triggered, and try to describe it in detail."
                               f"The three part in the output results will be seperated by ','."
                               f"For example, <output>Yes, subtype_name, triggering</output>, <output>No, new_subtype_name, triggering</output>."
                               f"You should only output the results with no other information.")

        content_ret_subtype = dpsk_calling.create_response(system_cmd, prompt_subtype_summary)
        print(content_ret_subtype)
        subtype_ret_result = re.search(r'<output>(.*?)</output>', content_ret_subtype, re.DOTALL)
        subtype_ret_text = subtype_ret_result.group(1).strip() if subtype_ret_result else "no text found"
        print(subtype_ret_text)
        subtype_split_res = subtype_ret_text.split(",")
        print(subtype_split_res)
        leaf_parent_id = ""
        if 'No' in subtype_split_res[0].strip():
            self.sub_type_id += 1
            self.memes_tree.add_node(sub_type_parent_id, f"Sub_Type_{self.sub_type_id}", f"{subtype_split_res[1].strip()}", "")
            leaf_parent_id = f"Sub_Type_{self.sub_type_id}"
        elif 'Yes' in subtype_split_res[0].strip():
            if f"{subtype_split_res[1].strip()}" in content_node_sub_type_layer:
                # 更新之后的
                ids_match = [tree_nodes_item.get_id() for tree_nodes_item in nodes_matches_type_ids if
                             tree_nodes_item.get_node_content() == f"{subtype_split_res[1].strip()}"]
                leaf_parent_id = ids_match[0]
                # leaf_parent_id = f"Sub_Type_{self.type_id}"
            else:
                return ""
        if leaf_parent_id == "":
            return ""
        self.trig_feat_id += 1
        # 这里需要拼接一步上下文信息，因为trigger_feature可能会比较长，所以subtype_split_res[2:]都引入
        trigger_feature_contents = subtype_split_res[2:]
        print("The trigger feature is: ", trigger_feature_contents)
        trigger_feature_content = ', '.join(trigger_feature_contents)
        self.memes_tree.add_node(leaf_parent_id, f"Trig_Feat_{self.trig_feat_id}", "", f"{trigger_feature_content.strip()}")

        # if sub_type_parent_id == "":
        #     return ""
        # layer_total = self.memes_tree.get_depth()
        # print(layer_total)
        #
        # parent_node_ids_list = [sub_type_parent_id]
        # for i in range(layer_total - 1):
        #     node_list = self.memes_tree.get_nodes_at_level(i + 2)
        #     for node_item in node_list:
        #         if node_item.parent_id == sub_type_parent_id:
        #     parent_ids_at_node_list = [node_item.get_parent_ids().get_id() for node_item in node_list]
        #     print(f"Layer {i + 1}: {parent_ids_at_node_list}")

        # text1 = """
        # Artificial intelligence is transforming various industries.
        # Machine learning algorithms can now recognize patterns in large datasets
        # and make predictions with remarkable accuracy.
        # """
        #
        # text2 = """
        # Deep learning models have revolutionized computer vision tasks.
        # These neural networks can identify objects in images almost as well as humans,
        # enabling advancements in autonomous vehicles and medical imaging.
        # """
        #
        # content_system = "You are a analyzer for the similarity between two texts."
        # content_user = (f"Please check whether the following two sentences' topics have similarities."
        #                 f"You should only return the similarity scores between them.\n"
        #                 f"Sentence 1: {text1}\n"
        #                 f"Sentence 2: {text2}\n")
        #
        # content_ret = dpsk_calling.create_response(content_system, content_user)
        return content_ret

    def save_to_json(self, save_addr, encoding="utf-8"):
        self.memes_tree.save_to_json(save_addr, encoding)
        # self.memes_tree.save_to_json(
        #     "/CVPR_Competition/Auto_Risk_Triggering_Extraction/MemeTriggerTree_FHM/meme_risk_triggered_module_tree_llm_v1.json")


# 示例使用
if __name__ == "__main__":
    # 创建树
    memes_tree = MemesTree()
    memes_tree.check_tree()

    with open('/CVPR_Competition/Auto_Risk_Triggering_Extraction/test_tree_updated.txt', 'r', encoding='utf8') as fin:
        update_content_reads = fin.readlines()
        fin.close()
    print(update_content_reads)
    for update_content_each in update_content_reads:
        memes_tree.update_tree(update_content_each)

    family_tree = Tree("Root", "Memes Risk Detection", "")
    family_tree.add_node("Root", "dad", "Dad", "111")
    family_tree.add_node("Root", "uncle", "Uncle", "111")
    family_tree.add_node("dad", "me", "Me", "111")
    family_tree.add_node("dad", "sister", "Sister", "111")

    # 修改节点内容
    family_tree.update_node_content("me", "Myself")

    # 遍历操作
    print("DFS遍历:")
    family_tree.traverse_dfs(action=lambda node: print(f"Visiting: {node.id}"))

    print("\nBFS遍历:")
    family_tree.traverse_bfs(action=lambda node: print(f"Visiting: {node.id}"))

    # 保存到文件
    family_tree.save_to_json("/CVPR_Competition/Auto_Risk_Triggering_Extraction/memes_risk_triggered_tree_v2.json")

    # 从文件加载
    loaded_tree = Tree.load_from_json("/CVPR_Competition/Auto_Risk_Triggering_Extraction/memes_risk_triggered_tree_v2.json")
    print("\n加载后的树结构:")
    loaded_tree.traverse_bfs(action=lambda node: print(f"Loaded: {node.id}"))

    # 删除节点
    loaded_tree.delete_node("sister")
    print("\n删除节点后的BFS遍历:")
    loaded_tree.traverse_bfs(action=lambda node: print(f"After delete: {node.id}"))