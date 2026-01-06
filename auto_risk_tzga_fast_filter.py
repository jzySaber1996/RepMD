import easyocr
import json
from deepseek_api_calling import DPSKCalling
from transformers import AutoTokenizer, AutoProcessor
from transformers import Qwen2VLForConditionalGeneration
import re
import os
import cv2
from datetime import datetime
# from mtcnn import MTCNN
import dlib
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from PIL import Image
import argparse
from pathlib import Path
from typing import List, Dict, Any


class FilteringRuleApplication:
    def __init__(self, args):
        # 初始化OCR引擎，支持多种语言
        self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)
        self.dpsk_calling = DPSKCalling()
        # Time of the tree/model storage
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

        result_vllm = self.llm.generate(messages, self.sampler)
        response = result_vllm[0].outputs[0].text
        return response  # just a placeholder

    def ocr_fig_text_info_detection(self, file_path):
        with open("/newdisk/public/JZY/CVPR_Competition/Auto_Risk_Triggering_Extraction/TZGA_Deployment/rule_path_2025-10-11 18:09:31/single_elem_rules.jsonl",
                  'r', encoding='utf-8') as fin:
            data_dict = json.load(fin)
        single_rule_keys = data_dict.keys()
        # 读取图像
        # fig_path = "/newdisk/public/JZY/Harmful_Dataset/huzhouga_harmful_dataset/Dataset/train/no_126.jpg"
        # image = cv2.imread('/newdisk/public/JZY/Harmful_Dataset/huzhouga_harmful_dataset/Dataset/train/no_126.jpg')
        # 使用OCR检测文本
        file_test_data = []
        with open(file_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                file_test_data.append(json.loads(line))
        for test_data_item in file_test_data:
            img_name = test_data_item["img"]
            fig_path = f"/newdisk/public/JZY/Harmful_Dataset/huzhouga_harmful_dataset/Dataset/{img_name}"
            image = Image.open(image_path).convert('RGB')
            text_input_item = "请分析下列图片中，是否有包含风险数据信息。"
            response_risk_detection = self.model_predict(text_input_item, image)
            print(f"风险信息判断结果：{response_risk_detection}")
            result = self.reader.readtext(fig_path)
            text_res_ret = ""
            for res in result:
                text_res_ret += f"{res[1]} "
            fast_filter_label = False
            label_harm = test_data_item["label"]
            for key in single_rule_keys:
                if key in text_res_ret:
                    fast_filter_label = True
                    break
            print(f"Text Info: {text_res_ret}, Harmful Label: {label_harm} Filter Prediction: {fast_filter_label}")
        # # 加载图像
        # image = cv2.imread(fig_path)
        # # 创建一个人脸检测器
        # detector = dlib.get_frontal_face_detector()
        # # 转换为灰度图像
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # # 检测人脸
        # faces = detector(gray)
        # # 在检测到的人脸周围画矩形框
        # for face in faces:
        #     x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        #     print(f"人脸坐标信息：{x1}, {y1}, {x2}, {y2}")
        #     # cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # 显示图像
        # cv2.imshow('Face detection', image)

def parse_args():
    parser = argparse.ArgumentParser(description="Batch inference with Qwen2‑VL‑2B on vLLM – JSON I/O version")
    parser.add_argument("--dataset", type=Path,
                        default=Path("/newdisk/public/wws/01-AIGC-GPRO/LLaMA-Factory/data_use/test_seen.jsonl"),
                        help="Path to evaluation JSON file")
    parser.add_argument("--img_dir", type=str, default="/newdisk/public/wws/00-Dataset-AIGC/FHM_new/img",
                        help="base directory for images")
    parser.add_argument("--save", type=Path, default=Path("predictions_auc_v2.json"), help="Path to output JSON file")
    parser.add_argument("--model", type=str,
                        default="/newdisk/public/wws/01-AIGC-GPRO/LLaMA-Factory/output/qwen2vl_2b/lora_sft_cls_450_4e_lr1e-4",
                        help="HF hub id or local dir")
    parser.add_argument("--gpu_memory_utilization", type=float, default=1, help="GPU memory utilization")
    parser.add_argument("--adapter", type=str, default=None, help="Path to LoRA adapter")
    parser.add_argument("--max_lora_rank", type=int, default=8,
                        help="Maximum LoRA rank (should match the rank used in SFT)")

    # GPU & performance
    parser.add_argument("--gpus", type=str, default="", help="Visible GPU ids, e.g. '0,1'. Empty = all visible")
    parser.add_argument("--tp", type=int, default=1, help="Tensor‑parallel world size (default = #GPUs)")

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=42)
    # args = parser.parse_args()
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    filter_rule_application = FilteringRuleApplication(args)
    filter_rule_application.ocr_fig_text_info_detection("/newdisk/public/JZY/Harmful_Dataset/huzhouga_harmful_dataset/Dataset/test.jsonl")
