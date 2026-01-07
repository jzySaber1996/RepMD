import os
import json
import base64
from PIL import Image
from datasets import Dataset, DatasetDict
import torch
from transformers import AutoTokenizer, AutoProcessor
from transformers import Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import argparse
from volcenginesdkarkruntime import Ark
from tqdm import tqdm, trange
import pandas as pd
from datetime import datetime
import re
from vllm import LLM, SamplingParams
from pathlib import Path
from typing import List, Dict, Any
from tree_triggering import MemesTree
from deepseek_api_calling import DPSKCalling


class AutoPromptTuner:
    def __init__(self, initial_prompt_dir='prompt_0', prompt_filename='prompt_risk_detection.txt',
                 prompt_suggestion_enhance_filename="suggestion_for_RAG.txt",
                 update_prompt_file='prompt_update.txt', args=""):
        self.model_info = "ep-20250427174648-pz2m7"
        self.prompt_dir = initial_prompt_dir
        self.prompt_file = prompt_filename
        self.prompt_enhance_file = prompt_suggestion_enhance_filename
        self.update_prompt_file = update_prompt_file
        # self.prompt_original = self.prompt
        # Two prompts, one is original, the second is the enhanced RAG
        self.prompt_original, self.prompt_original_enhance = self._load_prompt()
        self.content_ret_reason = ""
        print(self.prompt_original_enhance)
        self.prompt = self.prompt_original_enhance.replace("{SUGGESTIONS_FOR_IDENTIFY}", self.content_ret_reason)
        self.incorrect_reason = ""
        self.memes_tree = MemesTree()
        self.memes_tree.check_tree()

        # Time of the tree/model storage
        current_time = datetime.now()
        self.time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

        self.iteration = []
        self.accuracy = []
        self.precision = []
        self.recall = []
        self.f1 = []
        if args.gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus  # must be set *before* vLLM loads CUDA context
            gpu_count = len(args.gpus.split(","))
        else:
            # Count GPUs from nvidia‑smi visibility (fallback 1)
            try:
                import torch
                gpu_count = torch.cuda.device_count()
            except Exception:
                gpu_count = 1

        # 请确保您已将 API Key 存储在环境变量 ARK_API_KEY 中
        # 初始化Ark客户端，从环境变量中读取您的API Key
        self.client = Ark(
            # 此为默认路径，您可根据业务所在地域进行配置
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            # 从环境变量中获取您的 API Key。此为默认方式，您可根据需要进行修改
            api_key="b99a9cc7-b445-4c48-aad1-6296b6bc8cb1", # Doubao-1.5-vision-pro模型
            # api_key="1a20912b-bccf-47ff-bb89-1dc15bb643a3", # Doubao-1.5-vision-lite模型
        )

    def _load_prompt(self):
        prompt_path = os.path.join(self.prompt_dir, self.prompt_file)
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
            f.close()

        prompt_path_enhance = os.path.join(self.prompt_dir, self.prompt_enhance_file)
        with open(prompt_path_enhance, "r", encoding="utf-8") as f:
            prompt_enhance = f.read().strip()
            f.close()

        return prompt, prompt_enhance


    def model_predict(self, text_input_item, img):
        """
        实际调用模型预测的代码待填充，当前是占位符。
        """
        # 模型预测代码
        response = ""
        messages = ""
        # chatml_prompt = self.make_analysis_prompt(text_input_item)
        # prompt_ids = self.tokenizer.encode(chatml_prompt)
        if img:
            response = self.client.chat.completions.create(
                # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
                # model="ep-20250811084536-g4tcg",
                model=self.model_info,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img}"}},
                            {"type": "text", "text": text_input_item},
                        ],
                    }
                ],
                # 免费开启推理会话应用层加密，访问 https://www.volcengine.com/docs/82379/1389905 了解更多
                extra_headers={'x-is-encrypted': 'true'},
                temperature=0.0
            )
        else:
            response = self.client.chat.completions.create(
                # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
                # model="ep-20250811084536-g4tcg",
                model=self.model_info,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text_input_item},
                        ],
                    }
                ],
                # 免费开启推理会话应用层加密，访问 https://www.volcengine.com/docs/82379/1389905 了解更多
                extra_headers={'x-is-encrypted': 'true'},
                temperature=0.0
            )
        response_risk_detection = response.choices[0].message.content
        return response_risk_detection  # just a placeholder

    def encode_image(self, img_path):
        with open(img_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def run_prediction(self, input_data, train_eval="train"):
        """
        模型预测逻辑可以在此处实现，目前留空。
        输入数据（input_data）是图片描述文本列表。
        返回列表形式的预测结果 ['Yes', 'No',...]
        """
        predictions = []

        count_match = 0
        count_total = 0


        failed_img_example_list, failed_img_name_list = [], []
        success_img_example_list, success_img_name_list = [], []
        for date_index_item in input_data:
            # date_index_item = list_res[index_finding]
            # if date_index_item['label'] == 1:
            count_total += 1
            text_desc_index_item = date_index_item['text']
            # text_desc_index_item = ""
            text_input_item = self.prompt.replace('{TEXT_DESC_OF_PIC}', text_desc_index_item)
            # data_index_replaced = date_index_item.replace('月','-').replace('日','')
            img_name_item = date_index_item['img']
            img_name_item_replaced = img_name_item.split('\/')[-1]
            image_path = f'/Dataset-FHM/{img_name_item_replaced}'
            # image_path = f'/Harmful_Dataset/huzhouga_harmful_dataset/Dataset/{img_name_item_replaced}'
            base64_image = self.encode_image(image_path)
            # image = Image.open(image_path).convert('RGB')
            response_risk_detection = self.model_predict(text_input_item, base64_image)
            # 存储预测结果数据
            pred_item_dict = dict()
            pred_item_dict['id'] = date_index_item['id']
            pred_item_dict['text'] = date_index_item['text']
            # pred_item_dict['text'] = ""
            pred_item_dict['img'] = date_index_item['img']
            pred_item_dict['pred_desc'] = response_risk_detection
            # print(f"Response is: \n{response_risk_detection}")
            if 'yes' in response_risk_detection.split('\n')[-1].lower():
                pred_item_dict['pred'] = 1
            else:
                pred_item_dict['pred'] = 0
            pred_item_dict['label'] = date_index_item['label']
            predictions.append(pred_item_dict)
            # 分析预测结果差异
            if train_eval == 'eval':
                if pred_item_dict['pred'] == pred_item_dict['label']:
                    count_match += 1
                else:
                    failed_img_example_list.append(date_index_item['id'])
                    failed_img_name_list.append(img_name_item_replaced)
                    output_failed_dict = {'Failed_Item': failed_img_example_list,
                                          'Failed_Item_Img': failed_img_name_list}
                    df_output_failed = pd.DataFrame(output_failed_dict)
                    df_output_failed.to_csv(
                        f'/Dataset-FHM/Doubao_Bad_Cases/failed_test_seen_examples_memes_{self.time_str}.csv',
                        index=False)
            # det_pre = (1.0 * count_match) / count_total
            # loop_find_calling.set_postfix(count_match=count_match, count_total=count_total, precision=det_pre)
            # print(det_pre)
        return predictions

    def analyze_issues(self, predictions):
        """
        分析prompt可能存在的问题。此处用简单的对比分析示例表示。
        返回分析出的待解决问题（可以是简单的字符串描述）。
        """
        update_path = self.update_prompt_file
        if not os.path.exists(update_path):
            print("Prompt update file missing.")
            return

        with open(update_path, "r", encoding="utf-8") as f:
            update_strategy = f.read().strip()

        '''
        This prediction aims to find the prediction results:
        * If the label==1 but predict==0, that means the incorrect prediction will be added to the tree.
        * Another thing is, if label==0 but predict==1, we need to identify it, maintain an opposite tree.
        '''
        # false_predictions = [prediction_item['pred_desc'] for prediction_item in predictions if
        #                      prediction_item['pred'] != prediction_item['label']]

        false_predictions = [prediction_item['pred_desc'] for prediction_item in predictions if
                             prediction_item['pred'] == 0 and prediction_item['label'] == 1]

        issues_and_update, false_predictions_input = "", ""
        if len(false_predictions) > 0:
            false_predictions_input = '\n'.join(false_predictions)
            # print(false_predictions)

            update_strategy = (update_strategy.replace('{TEXT_DESC_OF_PREDICTION}', false_predictions_input))
            # update_strategy = (update_strategy.replace('{TEXT_DESC_OF_PREDICTION}', false_predictions_input).
            #                    replace('{PREVIOUS_PROMPT}', self.prompt))
            # print(f"---------------------The update strategy is ------------------\n"
            #       f"{update_strategy}\n"
            #       f"----------------------End of Strategy-------------------------")
            issues_and_update = self.model_predict(update_strategy, None)
            # print(issues_and_update)
        return issues_and_update, false_predictions_input

    '''
    Determine the cases for conducting RAG.
    1. LLM determines whether they are uncertain for the detection results.
    2. LLM calculate the similarity between the test case's feature with the triggering features in the database.
    '''
    def rag_determination(self):
        return

    '''
    Perform RAG.
    + Retrieve&Rank the most similar tree's triggering feature.
    + Use the feature, 
    '''
    def graph_rag_execution(self):
        return

    '''
    Implement the detection algorithm for the test dataset, with three steps:
    + Step-1: Rank the bottom-layer's triggering feature with the image itself, inserting into the queue.
    + Step-2: Bottom-Top traversing the tree, pruning the sub-trees with the triggering features.
    + Step-3: Detect the potential type in the tree.
    '''
    def eval_algorithm(self, date_index_item):
        text_desc_index_item = date_index_item['text']
        text_input_wo_enhance = self.prompt_original.format(TEXT_DESC_OF_PIC=text_desc_index_item)
        text_input_item = self.prompt.format(TEXT_DESC_OF_PIC=text_desc_index_item)
        
        # Reformat input text with the following step rules:

        '''
        These following appended rules are added into the original prompts.
        '''
        # appended_rules = ("Please use the tree to help you, which contains three parts: "
        #                   "* type: hateful type of the meme, "
        #                   "* subtype: child of the type, a more precise category,"
        #                   "* trigger feature: how the risk may trigger in the meme."
        #                   "You can analyze the previous trees with the following rules to detect the hateful risks:"
        #                   "+ Rule 1: Rank the triggering feature with the scores of meme's image, where the higher score means the image can match the feature."
        #                   "+ Rule 2: Bottom-to-Top traversing the tree based on the match scores pruning the subtypes and types with the triggering features."
        #                   "+ Rule 3: Detect the risk again based on the subtypes and your analysis of the image."
        #                   "Note that, all the previous analysis processes are silent, should not be shown in the final output!")
        #
        # text_input_item += appended_rules
        
        # data_index_replaced = date_index_item.replace('月','-').replace('日','')
        img_name_item = date_index_item['img']
        img_name_item_replaced = img_name_item.split('\/')[-1]
        image_path = f'/Dataset-FHM/{img_name_item_replaced}'
        # image_path = f'/Harmful_Dataset/huzhouga_harmful_dataset/Dataset/{img_name_item_replaced}'

        # image = Image.open(image_path).convert('RGB')
        base64_image = self.encode_image(image_path)
        # response_risk_detection = self.model_predict(text_input_item, image)

        '''
        Original Identification, judging the confidence and risks with the original prompt
        '''
        response_original = self.client.chat.completions.create(
            # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
            # model="ep-20250811084536-g4tcg",
            model=self.model_info,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                        {"type": "text", "text": text_input_wo_enhance},
                    ],
                }
            ],

            # 免费开启推理会话应用层加密，访问 https://www.volcengine.com/docs/82379/1389905 了解更多
            extra_headers={'x-is-encrypted': 'true'},
            temperature=0.0
        )

        response_risk_detection_original = response_original.choices[0].message.content
        response_risk_list_original = response_risk_detection_original.split('\n')
        response_risk_list_original = [item_risk.replace("- ", "") for item_risk in response_risk_list_original if "- " in item_risk]
        determine_result = response_risk_list_original[-1].lower()
        risk_confidence_score_original, risk_confidence_score = -1, -1
        try:
            risk_confidence_score_original = float(response_risk_list_original[-2])
            if risk_confidence_score_original >= 0.6:
                # response_risk_detection = result_vllm[0].outputs[0].text
                response = self.client.chat.completions.create(
                    # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
                    # model="ep-20250811084536-g4tcg",
                    model=self.model_info,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                                {"type": "text", "text": text_input_item},
                            ],
                        }
                    ],

                    # 免费开启推理会话应用层加密，访问 https://www.volcengine.com/docs/82379/1389905 了解更多
                    extra_headers={'x-is-encrypted': 'true'},
                    temperature=0.0
                )

                response_risk_detection = response.choices[0].message.content
                response_risk_list = response_risk_detection.split('\n')
                response_risk_list = [item_risk.replace("- ", "") for item_risk in response_risk_list
                                               if "- " in item_risk]
                risk_confidence_score = float(response_risk_list[-2])
                if risk_confidence_score_original - risk_confidence_score >= 0.2:
                    determine_result = response_risk_list[-1].lower()
                # print(f"Eval_Result: {determine_result}")
        except Exception:
            print(response_risk_list_original)
            return determine_result, img_name_item_replaced, risk_confidence_score_original, risk_confidence_score

        return determine_result, img_name_item_replaced, risk_confidence_score_original, risk_confidence_score


    def evaluation(self, iter_num):
        list_res = []
        with open('/Dataset-FHM/dev.jsonl', 'r',
                  encoding='utf-8') as fin:
            for line in fin:
                data_dict = json.loads(line)
                list_res.append(data_dict)
            fin.close()

        # prompt_text_desc = '''
        #     Please detect whether this image contains the potential risk, and you need to do the following three things:
        #     1. Output what the elements in this image, and what the meaningful information it describes.
        #     2. Decide whether it contains the potential risk, such as the politics (including but not limited to derogatory or unsubstantiated comparisons of political figures, one - sided and accusatory political statements that may polarize, inappropriate associations of identity with political stances), racists (including text that makes inappropriate racial statements, equates race with non - relevant things in a harmful way, and also including text with racial undertones that make inappropriate connections), religious discrimination (such as derogatory and offensive statements about a particular religion or its followers), trivialization of violent and serious acts (such as bomb - related violent acts which are associated with terrorism and cause great harm, presented in a matter - of - fact or potentially humorous way), trivialization of the presence of weapons in public/community spaces (even in a humorous context), and also consider the historical significance and gravity of using figures like Hitler (who was responsible for the Holocaust - a systematic genocide of millions including Jews, Romani people, disabled people, homosexuals etc.) in a context. If used in a trivial or inappropriate (like for commercial products in a non - serious way) manner, it may be a risk. Also, consider the trivialization of significant events (even if the figure involved is controversial) in a crude or dehumanizing way (such as using disrespectful language to refer to a person's death that had geopolitical and human consequences). Additionally, check for derogatory and offensive language towards marginalized groups (such as transgender individuals, people with dwarfism etc.) and also check for vulgar language which can be offensive in some settings. Also, check for vulgar and sexual content (such as objectification of the body in a sexual context which may be considered inappropriate).
        #     3. Output "Yes/No", where "Yes" means it has potential risk, and "No" means it is safe.
        #     Note that, the output 1~3 will separate in three lines splitted by symbol '\\n'.
        #     The detail of this image is: {TEXT_DESC_OF_PIC}
        #     '''

        loop_find_calling = tqdm(range(len(list_res)), desc='Identifying risks in dev.jsonl')

        count_match, count_total = 0, 0
        count_precision, count_pred_positive = 0, 0
        count_recall, count_label_positive = 0, 0

        failed_img_example_list, failed_img_name_list = [], []
        success_img_example_list, success_img_name_list = [], []
        det_acc, det_pre, det_rec, det_f1 = 0.0, 0.0, 0.0, 0.0
        print("------------Start of Prompt at this iteration----------\n")
        print(self.prompt)
        print("------------End of Prompt at this iteration----------\n")
        for index_finding in loop_find_calling:
            date_index_item = list_res[index_finding]
            # if date_index_item['label'] == 1:
            count_total += 1
            determine_result, img_name_item_replaced, risk_confidence_score_original, risk_confidence_score = self.eval_algorithm(date_index_item)
            # print(determine_result)
            # pred_res, label_res = 0, 0
            if 'yes' in determine_result:
                pred_res = 1
            else:
                pred_res = 0
            label_res = date_index_item['label']

            # response_risk_detection = response.choices[0].message.content
            # determine_result = response_risk_detection.split('\n')[-1].lower()

            # Calculate Accuracy
            if pred_res == label_res:
                count_match += 1
            det_acc = (1.0 * count_match) / count_total

            # Calculate Precision
            if pred_res == 1:
                count_pred_positive += 1
                if label_res == 1:
                    count_precision += 1
            if count_pred_positive > 0:
                det_pre = (1.0 * count_precision) / count_pred_positive
            else:
                det_pre = 0.0

            # Calculate Recall
            if label_res == 1:
                count_label_positive += 1
                if pred_res == 1:
                    count_recall += 1
            if count_label_positive > 0:
                det_rec = (1.0 * count_recall) / count_label_positive
            else:
                det_rec = 0.0

            # Calculate F1
            if (det_pre + det_rec) > 0:
                det_f1 = (2.0 * det_pre * det_rec) / (det_pre + det_rec)
            else:
                det_f1 = 0.0

            if pred_res != label_res:
                failed_img_example_list.append(date_index_item['id'])
                failed_img_name_list.append(img_name_item_replaced)
                output_failed_dict = {'Failed_Item': failed_img_example_list,
                                      'Failed_Item_Img': failed_img_name_list}
                df_output_failed = pd.DataFrame(output_failed_dict)
                df_output_failed.to_csv(
                    f'/Dataset-FHM/Doubao_Bad_Cases/failed_dev_examples_memes_{self.time_str}.csv',
                    index=False)

            # loop_find_calling.set_postfix(pred_res=pred_res, label_res=label_res, count_match=count_match,
            #                               count_total=count_total, acc=det_acc)
            loop_find_calling.set_postfix(pred_res=pred_res, label_res=label_res, confidence_score_original=risk_confidence_score_original,
                                          confidence_score=risk_confidence_score, acc=det_acc, pre=det_pre, rec=det_rec, f1=det_f1)
                # print(det_pre)
        print(f"#Positive-Label {count_label_positive}, #Negative-Label {count_total - count_label_positive}")
        self.iteration.append(iter_num)
        self.accuracy.append(det_acc)
        self.precision.append(det_pre)
        self.recall.append(det_rec)
        self.f1.append(det_f1)
        with open(f"/CVPR_Competition/Auto_Risk_Triggering_Extraction/Results_FHM/results_logs_meme_risk_triggered_module_tree_llm_{self.time_str}.txt", "w", encoding='utf8') as f_res_out:
            for iter_num_item, det_acc_item, det_pre_item, det_rec_item, det_f1_item in zip(self.iteration, self.accuracy, self.precision, self.recall, self.f1):
                f_res_out.write(f"Iteration: {iter_num_item}, Accuracy: {det_acc_item}, Precision: {det_pre_item}, Recall: {det_rec_item}, F1: {det_f1_item}\n")
            f_res_out.close()

    def auto_prompt_tune(self, rounds=10, batch_size=8, eval_steps=300):
        """
        自动调优提示，经过指定的轮次预测和真实标签对比，根据表现决定是否更新prompt。
        """
        # 基本训练数据的加载
        prompt_update_dir_path = os.path.join("/CVPR_Competition/prompts/",
                                              f"update_prompt_{self.time_str}")
        if not os.path.exists(prompt_update_dir_path):
            os.makedirs(prompt_update_dir_path)
        list_res = []
        # with open("/Harmful_Dataset/huzhouga_harmful_dataset/Dataset/train.jsonl", 'r',
        #           encoding='utf-8') as fin:

        with open('/Dataset-FHM/train.jsonl', 'r',
                  encoding='utf-8') as fin:
            for line in fin:
                data_dict = json.loads(line)
                list_res.append(data_dict)
            fin.close()
        count_steps = 0
        for round_idx in range(rounds):
            # has_updated_in_round = False

            # Batch-wise processing with tqdm progress bar
            total_batches = (len(list_res) + batch_size - 1) // batch_size
            print(f"Round {round_idx + 1}/{rounds}, Total Batch {total_batches}:")

            for batch_start in tqdm(range(0, len(list_res), batch_size), desc=f"Round {round_idx + 1}",
                                    total=total_batches):
                count_steps += 1
                batch = list_res[batch_start: batch_start + batch_size]

                # 提取当前batch的label，在predictions中可不提取
                # true_labels_batch = [item['label'] for item in batch]

                # 运行当前batch预测
                predictions = self.run_prediction(batch, train_eval="train")
                # print(predictions)

                # 分析问题，并对应更新prompt
                reason, false_predictions_input = self.analyze_issues(predictions)
                if reason:
                    # self.prompt += f"\n* {reason}"
                    reason_list = reason.split('\n')
                    for reason_item in reason_list:
                        self.memes_tree.update_tree(reason_item)
                    self.memes_tree.save_to_json(
                        f"/CVPR_Competition/Auto_Risk_Triggering_Extraction/MemeTriggerTree_FHM/meme_risk_triggered_module_tree_llm_{self.time_str}.json")
                    system_cmd = "You are an analyzer for the memes' risks."
                    dpsk_calling = DPSKCalling()
                    # print("-----Print the MemeTree's Data-----\n")
                    tree_json_data = ""
                    addr_json_tree = f"/CVPR_Competition/Auto_Risk_Triggering_Extraction/MemeTriggerTree_FHM/meme_risk_triggered_module_tree_llm_{self.time_str}.json"

                    with open(addr_json_tree, "r", encoding='utf8') as json_tree_in:
                        tree_json_data = json.load(json_tree_in)
                        json_tree_in.close()
                    # print(tree_json_data)
                    # prompt_update_original = (f"Please analyze the main information in the tree for memes' risks that the current model cannot accurately identified,"
                    #                           f"including the risk types, subtypes, and the features of how the risks are triggered:"
                    #                           f"Note that, please delete the subtree stared with 'Type_1', since it is a case for testing."
                    #                           f"Then, output the suggestions with some details to improve the model's ability."
                    #                           f"<meme_tree> {tree_json_data} </meme_tree>."
                    #                           f"Note that, each suggestion should be in detail, and occupies one line, with the following format:"
                    #                           f"Suggestion 1: The model need to xxx (details of the suggestion 1)."
                    #                           f"Suggestion 2: The model need to xxx (details of the suggestion 1)."
                    #                           f"..."
                    #                           f"Suggestion N： xxx.")

                    with open("/CVPR_Competition/prompts/suggestions.txt", "r", encoding="utf-8") as f_suggestion:
                        prompt_update_original = f_suggestion.read().strip()
                    # print("-----End of  the MemeTree's Data-----\n")
                    prompt_update_original = prompt_update_original.format(TREE_JSON_DATA=tree_json_data)
                    # print("------Prompt for generating suggestions---------\n")
                    # print(prompt_update_original)
                    # print("\n---------End of prompt for generating suggestions---------\n")
                    self.content_ret_reason = dpsk_calling.create_response(system_cmd, prompt_update_original)
                    self.prompt = self.prompt_original_enhance.replace("{SUGGESTIONS_FOR_IDENTIFY}", self.content_ret_reason)

                # 提取<reason>中的文本
                # reason = re.search(r'<reason>(.*?)</reason>', issues, re.DOTALL)
                # reason_text = reason.group(1).strip() if reason else "no reason found"

                # 提取<new_prompt>中的文本
                # new_prompt = re.search(r'<new_prompt>(.*?)</new_prompt>', issues, re.DOTALL)
                # new_prompt_text = new_prompt.group(1).strip() if new_prompt else self.prompt

                # 输出查看
                # print("Reason:\n", reason_text, "\n")
                # print("New Prompt:\n", new_prompt_text)

                # if "{TEXT_DESC_OF_PIC}" not in new_prompt_text:
                #     new_prompt_text += "{TEXT_DESC_OF_PIC}"


                # 该替换仅服务于Qwen2VL，因为缺乏对指令的遵循性，输出结果仅作为增加的内容添加到原始prompt后
                # if "{TEXT_DESC_OF_PIC}" in new_prompt_text:
                #     new_prompt_text.replace("{TEXT_DESC_OF_PIC}", "")
                # new_prompt_text = f"{self.prompt}\n* {reason_text}"
                update_file_path = os.path.join(prompt_update_dir_path, f"update_{count_steps}")
                if not os.path.exists(update_file_path):
                    os.makedirs(update_file_path)
                with open(update_file_path + "/reason.txt", "w", encoding="utf-8") as f:
                    f.write(reason)
                    f.close()
                with open(update_file_path + "/prompt.txt", "w", encoding="utf-8") as f:
                    f.write(self.prompt)
                    f.close()
                with open(update_file_path + "/false_prediction.txt", "w", encoding="utf-8") as f:
                    f.write(false_predictions_input)
                    f.close()

                if count_steps % eval_steps == 0:
                # if count_steps % eval_steps == 0:
                    # or count_steps == 1
                    # list_res_eval = []
                    # with open('/Dataset-FHM/dev.jsonl', 'r',
                    #           encoding='utf-8') as fin:
                    #     for line in fin:
                    #         data_dict_eval = json.loads(line)
                    #         list_res_eval.append(data_dict_eval)
                    #     fin.close()
                    self.evaluation(count_steps)
                # break

    def update_prompt(self, issues):
        """
        从 prompt_update 文件读取提示更新策略，并基于发现的问题issues对当前prompt做修改。
        当前采用简单替换方式，实际可以更复杂。
        """
        update_path = self.update_prompt_file
        if not os.path.exists(update_path):
            print("Prompt update file missing.")
            return

        with open(update_path, "r", encoding="utf-8") as f:
            update_strategy = f.read().strip()

        # 简单的示例更新策略：“追加”更新策略到当前prompt的末尾，即说明当前的问题
        updated_prompt = self.prompt + "\n\n" + "# Updates based on issues:\n" + update_strategy + "\n# Identified Issues:\n" + "\n".join(
            issues)

        # 更新内部prompt变量
        self.prompt = updated_prompt

        # 也可以选择保存新的提示到磁盘
        updated_prompt_path = os.path.join(self.prompt_dir, f'prompt_risk_detection_updated.txt')
        with open(updated_prompt_path, "w", encoding="utf-8") as f:
            f.write(updated_prompt)

    def __del__(self):
        '''
        进程释放
        '''
        print("The process is being released")


def parse_args():
    parser = argparse.ArgumentParser(description="Batch inference with Qwen2‑VL‑2B on vLLM – JSON I/O version")
    parser.add_argument("--dataset", type=Path,
                        default=Path("/newdisk/public/wws/01-AIGC-GPRO/LLaMA-Factory/data_use/test_seen.jsonl"),
                        help="Path to evaluation JSON file")
    parser.add_argument("--img_dir", type=str, default="/newdisk/public/wws/00-Dataset-AIGC/FHM_new/img",
                        help="base directory for images")
    parser.add_argument("--save", type=Path, default=Path("predictions_auc_v2.json"), help="Path to output JSON file")
    parser.add_argument("--model", type=str,
                        default="/newdisk/public/wws/01-AIGC-GPRO/LLaMA-Factory/output/qwen2vl_7b/lora_450v2_sft_4e_lr1e-4",
                        help="HF hub id or local dir")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--adapter", type=str, default=None, help="Path to LoRA adapter")
    parser.add_argument("--max_lora_rank", type=int, default=8,
                        help="Maximum LoRA rank (should match the rank used in SFT)")

    # GPU & performance
    parser.add_argument("--gpus", type=str, default="1", help="Visible GPU ids, e.g. '0,1'. Empty = all visible")
    parser.add_argument("--tp", type=int, default=1, help="Tensor‑parallel world size (default = #GPUs)")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=42)
    # args = parser.parse_args()
    return parser.parse_args()


def main():
    args = parse_args()

    # 构建数据集
    # dataset = build_hcd_dataset(
    #     args.jsonl_path,
    #     args.image_prefix,
    #     args.output_path,
    #     args.model_path,
    #     args.device
    # )

    image_descriptions = [
        "A child standing near the edge of the rooftop.",
        "A flower garden under clear sunny sky."
    ]
    true_labels = ['Yes', 'No']  # 真实标签

    # 实例化调优器并执行自动调优
    tuner = AutoPromptTuner(initial_prompt_dir="/CVPR_Competition/prompts/prompt_0/",
                            prompt_filename="prompt_risk_detection_tree_optimization_v0.txt",
                            prompt_suggestion_enhance_filename="suggestion_for_RAG.txt",
                            update_prompt_file="/CVPR_Competition/prompts/summarize_issue_prompt.txt",
                            args=args)
    # tuner.auto_prompt_tune(image_descriptions, true_labels, rounds=10)
    tuner.auto_prompt_tune()
    # print(prompt)

    print(f"Dataset saved to {args.output_path}")


if __name__ == "__main__":
    main()
