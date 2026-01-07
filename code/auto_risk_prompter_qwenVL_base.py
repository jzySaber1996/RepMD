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
                 update_prompt_file='prompt_update.txt', args=""):
        self.prompt_dir = initial_prompt_dir
        self.prompt_file = prompt_filename
        self.update_prompt_file = update_prompt_file
        self.prompt = self._load_prompt()
        self.prompt_original = self.prompt
        self.incorrect_reason = ""
        self.memes_tree = MemesTree()
        self.memes_tree.check_tree()
        current_time = datetime.now()
        self.time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

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

        tp_size = args.tp or max(1, gpu_count)  # default TP = number of visible GPUs
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(args.model)
        # 下面代码采用VLLM加速
        self.llm = self.build_llm(args.model,
                        args.adapter,
                        dtype=args.dtype,
                        tp_size=tp_size,
                        max_lora_rank=args.max_lora_rank,
                        gpu_memory_utilization=args.gpu_memory_utilization
                        )

        self.sampler = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            seed=args.seed,
            stop_token_ids=[self.tokenizer.eos_token_id],
        )

        # 下面代码不采用VLLM加速
        # self.model = Qwen2VLForConditionalGeneration.from_pretrained(
        #     model_path,
        #     torch_dtype=torch.bfloat16,
        #     device_map=device,
        #     trust_remote_code=True
        # )
        # self.processor = AutoProcessor.from_pretrained(
        #     model_path,
        #     trust_remote_code=True,
        #     # use_fast=True
        # )
        # 请确保您已将 API Key 存储在环境变量 ARK_API_KEY 中
        # 初始化Ark客户端，从环境变量中读取您的API Key
        # self.client = Ark(
        #     # 此为默认路径，您可根据业务所在地域进行配置
        #     base_url="https://ark.cn-beijing.volces.com/api/v3",
        #     # 从环境变量中获取您的 API Key。此为默认方式，您可根据需要进行修改
        #     api_key="b99a9cc7-b445-4c48-aad1-6296b6bc8cb1",
        # )

    def _load_prompt(self):
        prompt_path = os.path.join(self.prompt_dir, self.prompt_file)
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        return prompt

    # def get_first_device(self):
    #     return next(self.model.parameters()).device

    def build_llm(self, model_name: str,
                  adapter_path: str = None,
                  dtype: str = "bfloat16",
                  tp_size: int = 1,
                  max_model_len: int = 4096,
                  gpu_memory_utilization: float = 0.9,
                  max_lora_rank: int = 32,
                  ) -> LLM:
        """Create a vLLM engine for Qwen2-VL-2B."""
        import torch
        torch.cuda.empty_cache()

        engine_args = {
            "model": model_name,
            "trust_remote_code": True,
            "dtype": dtype,
            "tensor_parallel_size": tp_size,
            "max_model_len": max_model_len,
            "limit_mm_per_prompt": {"image": 1, "video": 0},  # Qwen2‑VL supports 1 img/prompt by default
            "gpu_memory_utilization": gpu_memory_utilization,
            "enable_lora": adapter_path is not None,  # 启用 LoRA 支持
            "max_lora_rank": max_lora_rank,  # 设置最大 LoRA rank
        }
        return LLM(**engine_args)

    def make_analysis_prompt(self, prompt_original) -> str:
        """Wrap raw prompt text with ChatML tags expected by Qwen‑family models."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},  # 只写 type 即可，真正图片走 mm_data
                    {"type": "text", "text": prompt_original},
                ],
            }
        ]
        chatml_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return chatml_prompt

    def model_predict(self, text_input_item, img):
        """
        实际调用模型预测的代码待填充，当前是占位符。
        """
        # 模型预测代码
        response = ""
        messages = ""
        chatml_prompt = self.make_analysis_prompt(text_input_item)
        prompt_ids = self.tokenizer.encode(chatml_prompt)
        if img:
            """加载模型生成response过程"""

            # QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final judgment (harmful / harmless) in <answer> </answer> tags."
            # prompt = QUESTION_TEMPLATE.format(Question=problem)

            # 构建多模态输入
            messages = [
                {
                    "prompt_token_ids": prompt_ids,
                    "multi_modal_data": {"image": [img]},
                }
            ]
        else:
            messages = [
                {
                    "prompt_token_ids": prompt_ids,
                    "multi_modal_data": {},  # ★ 无图片
                }
            ]

        # 组装 Qwen 的输入
        # chat_text = self.processor.apply_chat_template(
        #     messages, tokenize=False, add_generation_prompt=True
        # )
        #
        # image_inputs, video_inputs = process_vision_info(messages)
        # inputs = self.processor(
        #     text=[chat_text],
        #     images=image_inputs,
        #     videos=video_inputs,
        #     padding=True,
        #     return_tensors="pt",
        # )
        # device0 = self.get_first_device()
        # for k, v in inputs.items():
        #     if torch.is_tensor(v):
        #         inputs[k] = v.to(device0, non_blocking=True)
        #
        # with torch.no_grad():
        #     generated_ids = self.model.generate(
        #         **inputs,
        #         max_new_tokens=512,
        #         do_sample=True,
        #         temperature=0.7,
        #         top_p=0.9,
        #     )
        #
        # # 与原代码同样的解码逻辑（只截取新生成部分）
        # generated_ids_trimmed = [
        #     out_ids[len(in_ids):]
        #     for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        # ]
        #
        # # outputs = llm.generate([chat_text], sampling_params)
        # # response = outputs[0].outputs[0].text
        #
        # response = self.processor.batch_decode(
        #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        # )[0]

        result_vllm = self.llm.generate(messages, self.sampler)
        response = result_vllm[0].outputs[0].text
        return response  # just a placeholder

    def run_prediction(self, input_data, train_eval="train"):
        """
        模型预测逻辑可以在此处实现，目前留空。
        输入数据（input_data）是图片描述文本列表。
        返回列表形式的预测结果 ['Yes', 'No',...]
        """
        predictions = []
        # list_res = []
        # with open('/CVPR_Competition/fudan_vlux_V1/Dataset-FHM/train.jsonl', 'r',
        #           encoding='utf-8') as fin:
        #     for line in fin:
        #         data_dict = json.loads(line)
        #         list_res.append(data_dict)
        #     fin.close()

        # loop_find_calling = tqdm(range(len(list_res)), desc='Detecting Img Risks')

        count_match = 0
        count_total = 0

        # current_time = datetime.now()
        # time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

        failed_img_example_list, failed_img_name_list = [], []
        success_img_example_list, success_img_name_list = [], []
        for date_index_item in input_data:
            # date_index_item = list_res[index_finding]
            # if date_index_item['label'] == 1:
            count_total += 1
            text_desc_index_item = date_index_item['text']
            text_input_item = self.prompt.replace('{TEXT_DESC_OF_PIC}', text_desc_index_item)
            # data_index_replaced = date_index_item.replace('月','-').replace('日','')
            img_name_item = date_index_item['img']
            img_name_item_replaced = img_name_item.split('\/')[-1]
            image_path = f'/CVPR_Competition/fudan_vlux_V1/Dataset-FHM/{img_name_item_replaced}'
            image = Image.open(image_path).convert('RGB')
            response_risk_detection = self.model_predict(text_input_item, image)
            # 存储预测结果数据
            pred_item_dict = dict()
            pred_item_dict['id'] = date_index_item['id']
            pred_item_dict['text'] = date_index_item['text']
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
                        f'/CVPR_Competition/prompts_qwen2vl/Qwen_Bad_Cases/failed_eval_seen_examples_memes_{self.time_str}.csv',
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

        false_predictions = [prediction_item['pred_desc'] for prediction_item in predictions if
                             prediction_item['pred'] != prediction_item['label']]
        issues_and_update, false_predictions_input = "", ""
        if len(false_predictions) > 0:
            false_predictions_input = '\n------------\n'.join(false_predictions)
            # print(false_predictions_input)

            update_strategy = (update_strategy.replace('{TEXT_DESC_OF_PREDICTION}', false_predictions_input))
            # update_strategy = (update_strategy.replace('{TEXT_DESC_OF_PREDICTION}', false_predictions_input).
            #                    replace('{PREVIOUS_PROMPT}', self.prompt))
            print(f"---------------------The update strategy is ------------------\n"
                  f"{update_strategy}\n"
                  f"----------------------End of Strategy-------------------------")
            issues_and_update = self.model_predict(update_strategy, None)
            print(issues_and_update)
        return issues_and_update, false_predictions_input

    def evaluation(self):
        list_res = []
        with open('/CVPR_Competition/fudan_vlux_V1/Dataset-FHM/dev.jsonl', 'r',
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

        loop_find_calling = tqdm(range(len(list_res)), desc='Identifying Risks in dev.jsonl')

        count_match = 0
        count_total = 0

        # current_time = datetime.now()
        # time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

        failed_img_example_list, failed_img_name_list = [], []
        success_img_example_list, success_img_name_list = [], []
        for index_finding in loop_find_calling:
            date_index_item = list_res[index_finding]
            if date_index_item['label'] == 1:
                count_total += 1
                text_desc_index_item = date_index_item['text']
                text_input_item = self.prompt.format(TEXT_DESC_OF_PIC=text_desc_index_item)
                # data_index_replaced = date_index_item.replace('月','-').replace('日','')
                img_name_item = date_index_item['img']
                img_name_item_replaced = img_name_item.split('\/')[-1]
                image_path = f'/CVPR_Competition/fudan_vlux_V1/Dataset-FHM/{img_name_item_replaced}'
                image = Image.open(image_path).convert('RGB')
                response_risk_detection = self.model_predict(text_input_item, image)

                determine_result = response_risk_detection.split('\n')[-1].lower()
                # print(determine_result)
                # pred_res, label_res = 0, 0
                if 'yes' in determine_result:
                    pred_res = 1
                else:
                    pred_res = 0
                label_res = date_index_item['label']

                # response_risk_detection = response.choices[0].message.content
                # determine_result = response_risk_detection.split('\n')[-1].lower()
                if pred_res == label_res:
                    count_match += 1
                else:
                    failed_img_example_list.append(date_index_item['id'])
                    failed_img_name_list.append(img_name_item_replaced)
                    output_failed_dict = {'Failed_Item': failed_img_example_list,
                                          'Failed_Item_Img': failed_img_name_list}
                    df_output_failed = pd.DataFrame(output_failed_dict)
                    df_output_failed.to_csv(
                        f'/CVPR_Competition/prompts_qwen2vl/Qwen_Bad_Cases/failed_dev_examples_memes_{self.time_str}.csv',
                        index=False)
                det_pre = (1.0 * count_match) / count_total
                loop_find_calling.set_postfix(pred_res=pred_res, label_res=label_res, count_match=count_match,
                                              count_total=count_total, acc=det_pre)
                # print(det_pre)

    def auto_prompt_tune(self, rounds=10, batch_size=8, eval_steps=10):
        """
        自动调优提示，经过指定的轮次预测和真实标签对比，根据表现决定是否更新prompt。
        """
        # 基本训练数据的加载
        current_time = datetime.now()
        time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
        prompt_update_dir_path = os.path.join("/CVPR_Competition/prompts_qwen2vl/",
                                              f"update_prompt_{self.time_str}")
        if not os.path.exists(prompt_update_dir_path):
            os.makedirs(prompt_update_dir_path)
        list_res = []
        with open('/CVPR_Competition/fudan_vlux_V1/Dataset-FHM/train.jsonl', 'r',
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
                        f"/CVPR_Competition/Auto_Risk_Triggering_Extraction/MemeTriggerTree_Qwen_EXP/meme_risk_triggered_module_tree_qwen_{self.time_str}.json")
                    system_cmd = "You are an analyzer for the memes' risks."
                    dpsk_calling = DPSKCalling()
                    print("-----Print the MemeTree's Data-----\n")
                    tree_json_data = ""
                    addr_json_tree = f"/CVPR_Competition/Auto_Risk_Triggering_Extraction/MemeTriggerTree_Qwen_EXP/meme_risk_triggered_module_tree_qwen_{self.time_str}.json"

                    with open(addr_json_tree, "r", encoding='utf8') as json_tree_in:
                        tree_json_data = json.load(json_tree_in)
                        json_tree_in.close()
                    print(tree_json_data)
                    prompt_update_original = (f"Please describe the main information in the tree for memes' risks"
                                              f"The top of this tree is the meme's risk detection."
                                              f"The following is the risk types, subtypes, "
                                              f"and how the data is triggered:"
                                              f"<meme_tree> {tree_json_data} </meme_tree>."
                                              f"Note that, the output value will be the summary,"
                                              f"without other information.")
                    print("-----End of  the MemeTree's Data-----\n")
                    content_ret_reason = dpsk_calling.create_response(system_cmd, prompt_update_original)
                    self.prompt = self.prompt_original + "\n" + content_ret_reason

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
                    # or count_steps == 1
                    # list_res_eval = []
                    # with open('/CVPR_Competition/fudan_vlux_V1/Dataset-FHM/dev.jsonl', 'r',
                    #           encoding='utf-8') as fin:
                    #     for line in fin:
                    #         data_dict_eval = json.loads(line)
                    #         list_res_eval.append(data_dict_eval)
                    #     fin.close()
                    self.evaluation()
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
    tuner = AutoPromptTuner(initial_prompt_dir="/CVPR_Competition/prompts_qwen2vl/prompt_0/",
                            prompt_filename="prompt_risk_detection.txt",
                            update_prompt_file="/CVPR_Competition/prompts_qwen2vl/summarize_issue_prompt.txt",
                            args=args)
    # tuner.auto_prompt_tune(image_descriptions, true_labels, rounds=10)
    tuner.auto_prompt_tune()
    # print(prompt)

    print(f"Dataset saved to {args.output_path}")


if __name__ == "__main__":
    main()
