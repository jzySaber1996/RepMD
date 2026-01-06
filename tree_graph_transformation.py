import json
import networkx as nx
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from deepseek_api_calling import DPSKCalling
from tqdm import tqdm, trange


class EnhancedMemeRiskGraphBuilder:
    def __init__(self, tree_data):
        """
        初始化增强的Meme风险异构图构建器

        Args:
            tree_data: 原始的模因风险触发树数据
        """
        self.tree_data = tree_data
        self.risk_nodes = []  # 风险类型节点
        self.trigger_nodes = []  # 原始触发特征节点
        self.merged_triggers = []  # 合并后的触发特征节点
        self.edges = []  # 节点关系
        self.edge_descriptions = {}  # 存储边描述 {(source, target): description}
        self.trigger_mapping = {}  # 原始触发特征ID到合并后ID的映射
        self.graph = None  # 构建的异构图
        self.dpsk_calling = DPSKCalling()

    def parse_tree(self):
        """
        解析原始树结构，提取风险节点和触发特征节点
        """
        self.risk_nodes = []
        self.trigger_nodes = []
        self.edges = []
        self.edge_descriptions = {}

        def traverse(node, parent_risk_id=None):
            node_id = node["id"]

            # 判断节点类型（对于公开的FHM数据，选择下面这个判断标准）
            # if node_id.startswith("Type_") or node_id.startswith("Sub_Type_"):
            # 判断节点类型（对于公开的FHM数据，选择下面这个判断标准）
            if node_id.startswith("C") or node_id.startswith("I"):
                # 风险类型节点
                self.risk_nodes.append({
                    "id": node_id,
                    "content": node["content"],
                    "type": "risk_type",
                    "trigger_feature": node.get("trigger_feature", "")
                })

                # 如果有父风险节点，添加关系
                if parent_risk_id:
                    self.edges.append((parent_risk_id, node_id, {"relation": "subtype"}))

                # 更新当前父节点ID
                current_parent = node_id
            elif node_id.startswith("Trig_Feat_"):
                # 触发特征节点
                if node.get("trigger_feature"):  # 只处理非空触发特征
                    self.trigger_nodes.append({
                        "id": node_id,
                        "trigger_feature": node["trigger_feature"],
                        "type": "trigger_feature"
                    })
                    # 添加与父风险节点的关系
                    if parent_risk_id:
                        self.edges.append((parent_risk_id, node_id, {"relation": "has_trigger"}))

                        # 生成边描述
                        risk_content = next((r["content"] for r in self.risk_nodes if r["id"] == parent_risk_id), "")
                        trigger_content = node["trigger_feature"]
                        description = self._generate_edge_description(risk_content, trigger_content)
                        self.edge_descriptions[(parent_risk_id, node_id)] = description

                current_parent = parent_risk_id
            else:
                # 根节点或其他节点，不作为风险或触发节点处理
                current_parent = parent_risk_id

            # 递归处理子节点
            for child in node.get("children", []):
                traverse(child, current_parent)

        traverse(self.tree_data)
        return self

    def _generate_edge_description(self, risk_content, trigger_content):
        """
        生成边描述，说明风险类型和触发特征之间的关系

        Args:
            risk_content: 风险类型内容
            trigger_content: 触发特征内容

        Returns:
            边描述文本
        """
        # 这里直接返回描述值就行了

        if risk_content and trigger_content:
            return f"Risk triggered by '{trigger_content}'"
        elif risk_content:
            return f"Risk type is '{risk_content}'"
        else:
            return f"Risk triggered by '{trigger_content}'"

    def _summarize_trigger_feature(self, text):
        """
        使用DeepSeek API对触发特征进行摘要

        Args:
            text: 触发特征文本

        Returns:
            摘要后的文本
        """
        # 这里模拟调用DeepSeek API进行文本摘要
        # 实际应用中应该调用真实的API
        system_cmd = "You are an analyzer for the memes' risk triggered features."
        prompt_summary_trigger = (f"## Task of Summary"
                                  f"Please summarize the text with one single sentence with less than 10 words."
                                  f"Focus on what is the main type of this risk, and the main feature of how the risk is triggered."
                                  f"The feature are elements in the meme, relations between elements, some implicit description, or other information the input has emphasized, etc."
                                  f"## Text of Input"
                                  f"{text}"
                                  f"## Output"
                                  f"Directly output the summarized result without any other additional information")
        summary_text = self.dpsk_calling.create_response(system_cmd, prompt_summary_trigger)

        # 简单的摘要逻辑
        # if len(text) > 50:
        #     return text[:50] + "..."
        return summary_text

    def merge_similar_triggers(self, similarity_threshold=0.6):
        """
        合并相似的触发特征

        Args:
            similarity_threshold: 相似度阈值，高于此值的触发特征将被合并
        """
        if not self.trigger_nodes:
            return self

        # 先对触发特征进行摘要
        summarized_triggers = []

        loop_trigger_summarize = tqdm(range(len(self.trigger_nodes)), desc="Summarize Trigger Feature's Text")

        for node_item_index in loop_trigger_summarize:
            node = self.trigger_nodes[node_item_index]
            summarized_text = self._summarize_trigger_feature(node["trigger_feature"])
            summarized_triggers.append({
                "id": node["id"],
                "trigger_feature": node["trigger_feature"],
                "summarized_feature": summarized_text,
                "type": "trigger_feature"
            })

        # 提取摘要后的触发特征文本
        trigger_texts = [node["summarized_feature"] for node in summarized_triggers]

        # 调用API进行相似度分析和聚类
        clusters = self._call_deepseek_api(trigger_texts, similarity_threshold)

        # 按聚类结果分组
        grouped_triggers = defaultdict(list)
        for i, cluster_id in enumerate(clusters):
            grouped_triggers[cluster_id].append(summarized_triggers[i])

        # 创建合并后的触发特征节点
        self.merged_triggers = []
        self.trigger_mapping = {}

        for cluster_id, triggers in grouped_triggers.items():
            if not triggers:
                continue

            # 创建合并节点
            merged_id = f"Merged_Trigger_{cluster_id}"

            # 使用摘要后的文本作为代表
            representative_text = triggers[0]["summarized_feature"]

            # 如果有多个，可以生成更综合的描述
            if len(triggers) > 1:
                # 使用摘要后的文本
                representative_text = f"Feature {', '.join([t['summarized_feature'] for t in triggers[:3]])}"
                if len(triggers) > 3:
                    representative_text += f" etc"

            self.merged_triggers.append({
                "id": merged_id,
                "trigger_feature": representative_text,
                "original_features": [t["trigger_feature"] for t in triggers],
                "type": "merged_trigger_feature",
                "original_count": len(triggers),
                "original_ids": [t["id"] for t in triggers]
            })

            # 建立映射
            for trigger in triggers:
                self.trigger_mapping[trigger["id"]] = merged_id

        return self

    def _call_deepseek_api(self, texts, threshold):
        """
        模拟DeepSeek API调用进行文本相似度计算和聚类

        Args:
            texts: 文本列表
            threshold: 相似度阈值

        Returns:
            聚类结果，每个文本对应的类别标签
        """
        if len(texts) <= 1:
            return [0] * len(texts)

        # 使用TF-IDF向量化文本
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)

        # 计算余弦相似度
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # 简单的聚类：相似度高于阈值则视为同一类
        clusters = []
        cluster_id = 0
        assigned = [-1] * len(texts)

        for i in range(len(texts)):
            if assigned[i] == -1:
                assigned[i] = cluster_id
                for j in range(i + 1, len(texts)):
                    if similarity_matrix[i][j] > threshold and assigned[j] == -1:
                        assigned[j] = cluster_id
                cluster_id += 1

        return assigned

    def build_graph(self):
        """
        构建异构图
        """
        G = nx.Graph()

        # 添加风险类型节点
        for node in self.risk_nodes:
            G.add_node(node["id"], **node)

        # 添加合并后的触发特征节点
        for node in self.merged_triggers:
            G.add_node(node["id"], **node)

        # 添加边（处理触发特征节点的映射）
        for source, target, attr in self.edges:
            original_target = target

            # 如果目标节点是触发特征且已被合并，则映射到合并后的节点
            if target.startswith("Trig_Feat_") and target in self.trigger_mapping:
                target = self.trigger_mapping[target]

                # 添加边描述
                if (source, original_target) in self.edge_descriptions:
                    attr["description"] = self.edge_descriptions[(source, original_target)]

            # 添加边
            if G.has_node(source) and G.has_node(target):
                G.add_edge(source, target, **attr)

        self.graph = G
        return self

    def to_json(self, filename=None):
        """
        将异构图转换为JSON格式

        Args:
            filename: 如果提供，则将JSON保存到文件

        Returns:
            JSON格式的图数据
        """
        if self.graph is None:
            raise ValueError("请先调用build_graph()方法构建图")

        # 构建节点列表
        nodes = []
        for node_id, data in self.graph.nodes(data=True):
            node_data = {"id": node_id}
            node_data.update(data)
            nodes.append(node_data)

        # 构建边列表
        edges = []
        for source, target, data in self.graph.edges(data=True):
            edge_data = {"source": source, "target": target}
            edge_data.update(data)
            edges.append(edge_data)

        # 构建完整的图数据
        graph_data = {
            "metadata": {
                "risk_node_count": len(self.risk_nodes),
                "trigger_node_count": len(self.trigger_nodes),
                "merged_trigger_count": len(self.merged_triggers),
                "edge_count": len(edges),
                "edge_description_count": len(self.edge_descriptions)
            },
            "nodes": nodes,
            "edges": edges
        }

        # 如果提供了文件名，则保存到文件
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)

        return graph_data

    @classmethod
    def load(cls, filename):
        """
        从JSON文件加载异构图

        Args:
            filename: JSON文件名

        Returns:
            EnhancedMemeRiskGraphBuilder实例
        """
        with open(filename, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)

        # 创建构建器实例
        builder = cls()

        # 重建图
        G = nx.Graph()

        # 添加节点
        for node in graph_data["nodes"]:
            node_id = node.pop("id")
            G.add_node(node_id, **node)

        # 添加边
        for edge in graph_data["edges"]:
            source = edge.pop("source")
            target = edge.pop("target")
            G.add_edge(source, target, **edge)

        builder.graph = G

        # 恢复其他状态（如果存在）
        if "metadata" in graph_data:
            metadata = graph_data["metadata"]
            builder.risk_nodes = [n for n in graph_data["nodes"] if n.get("type") == "risk_type"]
            builder.merged_triggers = [n for n in graph_data["nodes"] if n.get("type") == "merged_trigger_feature"]

        if "trigger_mapping" in graph_data:
            builder.trigger_mapping = graph_data["trigger_mapping"]

        if "edge_descriptions" in graph_data:
            # 将字符串键转换回元组
            builder.edge_descriptions = {}
            for k, v in graph_data["edge_descriptions"].items():
                parts = k.split("-")
                if len(parts) == 2:
                    builder.edge_descriptions[(parts[0], parts[1])] = v

        return builder

    def get_stats(self):
        """
        获取图的统计信息
        """
        if self.graph is None:
            return {"status": "图尚未构建"}

        return {
            "risk_nodes": len(self.risk_nodes),
            "trigger_nodes": len(self.trigger_nodes),
            "merged_triggers": len(self.merged_triggers),
            "edges": len(self.edges),
            "edge_descriptions": len(self.edge_descriptions),
            "graph_nodes": self.graph.number_of_nodes(),
            "graph_edges": self.graph.number_of_edges()
        }


# 主函数
def main():
    # 原始树数据（开源FHM数据集）
    # tree_file_name = "meme_risk_triggered_module_tree_llm_2025-08-21 15:31:56.json"
    # with open(f"/newdisk/public/JZY/CVPR_Competition/Auto_Risk_Triggering_Extraction/MemeTriggerTree_FHM/{tree_file_name}", "r", encoding="utf-8") as f:
    #     tree_data = json.load(f)

    # 台州公安数据
    tree_file_name = "tzga_meme_risk_triggered_module_tree_llm_2025-10-10 17:59:29.json"
    with open(
            f"/newdisk/public/JZY/CVPR_Competition/Auto_Risk_Triggering_Extraction/TZGA_Deployment/{tree_file_name}",
            "r", encoding="utf-8") as f:
        tree_data = json.load(f)

    # 创建构建器实例
    builder = EnhancedMemeRiskGraphBuilder(tree_data)

    # 构建异构图
    builder.parse_tree() \
        .merge_similar_triggers(similarity_threshold=0.6) \
        .build_graph()

    # 获取统计信息
    stats = builder.get_stats()
    print("图统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 保存为JSON文件（公开FHM数据集）
    # json_data = builder.to_json(f"/newdisk/public/JZY/CVPR_Competition/Auto_Risk_Triggering_Extraction/Meme_HGraph_FHM/{tree_file_name}")

    # 保存为JSON文件（台州公安数据集）
    json_data = builder.to_json(f"/newdisk/public/JZY/CVPR_Competition/Auto_Risk_Triggering_Extraction/TZGA_Deployment/Meme_Graph_TZGA/{tree_file_name}")

    print(f"\n图已保存到 enhanced_meme_risk_graph.json")

    # 打印部分节点和边信息
    print("\n风险类型节点示例:")
    for node in builder.risk_nodes[:3]:
        print(f"  {node['id']}: {node['content']}")

    print("\n合并后的触发特征节点示例:")
    for node in builder.merged_triggers[:3]:
        print(f"  {node['id']}: {node['trigger_feature']}")

    print("\n边描述示例:")
    for i, (edge, description) in enumerate(list(builder.edge_descriptions.items())[:3]):
        print(f"  {edge[0]} -> {edge[1]}: {description}")

    print("\n节点关系示例:")
    for i, edge in enumerate(builder.edges[:3]):
        print(f"  {edge[0]} -- {edge[1]} ({edge[2]['relation']})")


if __name__ == "__main__":
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(["Risk triggered by sensitive historical imagery and text manipulation to evoke emotional responses.",
                                             "Risk triggered by text and imagery subtly alluding to historical atrocities through insensitive humor"])

    # 计算余弦相似度
    similarity_matrix = cosine_similarity(tfidf_matrix)
    print(similarity_matrix)

    main()