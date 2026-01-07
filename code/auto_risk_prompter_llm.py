import os
import base64
from volcenginesdkarkruntime import Ark
from tqdm import tqdm, trange
import pandas as pd
from datetime import datetime
import json
import re


class AutoPromptTuner:
    def __init__(self, initial_prompt_dir='prompt_0', prompt_filename='prompt_risk_detection.txt',
                 update_prompt_file='prompt_update.txt'):
        self.prompt_dir = initial_prompt_dir
        self.prompt_file = prompt_filename
        self.update_prompt_file = update_prompt_file
        self.prompt = self._load_prompt()
        # 请确保您已将 API Key 存储在环境变量 ARK_API_KEY 中
        # 初始化Ark客户端，从环境变量中读取您的API Key
        self.client = Ark(
            # 此为默认路径，您可根据业务所在地域进行配置
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            # 从环境变量中获取您的 API Key。此为默认方式，您可根据需要进行修改
            api_key="b99a9cc7-b445-4c48-aad1-6296b6bc8cb1",
        )

    def _load_prompt(self):
        prompt_path = os.path.join(self.prompt_dir, self.prompt_file)
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        return prompt

    def model_predict(self, text_input_item, img):
        """
        实际调用模型预测的代码待填充，当前是占位符。
        """
        # 模型预测代码待完善
        response = ""
        if img:
            response = self.client.chat.completions.create(
                # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
                model="ep-20250427174648-pz2m7",
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
            )
        else:
            response = self.client.chat.completions.create(
                # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
                model="ep-20250427174648-pz2m7",
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
            )
        response_risk_detection = response.choices[0].message.content
        return response_risk_detection  # just a placeholder

    def run_prediction(self, input_data, train_eval="train"):
        """
        模型预测逻辑可以在此处实现，目前留空。
        输入数据（input_data）是图片描述文本列表。
        返回列表形式的预测结果 ['Yes', 'No',...]
        """
        predictions = []
        # list_res = []
        # with open('/newdisk/public/JZY/CVPR_Competition/fudan_vlux_V1/Dataset-FHM/train.jsonl', 'r',
        #           encoding='utf-8') as fin:
        #     for line in fin:
        #         data_dict = json.loads(line)
        #         list_res.append(data_dict)
        #     fin.close()

        # loop_find_calling = tqdm(range(len(list_res)), desc='Detecting Img Risks')

        count_match = 0
        count_total = 0

        # 定义方法将指定路径图片转为Base64编码
        def encode_image(img_path):
            with open(img_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

        current_time = datetime.now()
        time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

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
            image_path = f'/newdisk/public/JZY/CVPR_Competition/fudan_vlux_V1/Dataset-FHM/{img_name_item_replaced}'
            base64_image = encode_image(image_path)
            response_risk_detection = self.model_predict(text_input_item, base64_image)
            # 存储预测结果数据
            pred_item_dict = dict()
            pred_item_dict['id'] = date_index_item['id']
            pred_item_dict['text'] = date_index_item['text']
            pred_item_dict['img'] = date_index_item['img']
            pred_item_dict['pred_desc'] = response_risk_detection
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
                        f'/newdisk/public/JZY/CVPR_Competition/fudan_vlux_V1/Dataset-FHM/Doubao_Bad_Cases/failed_test_seen_examples_memes_{time_str}.csv',
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

            update_strategy = (update_strategy.replace('{TEXT_DESC_OF_PREDICTION}', false_predictions_input).
                               replace('{PREVIOUS_PROMPT}', self.prompt))
            issues_and_update = self.model_predict(update_strategy, None)
        return issues_and_update, false_predictions_input

    def evaluation(self):
        list_res = []
        with open('/newdisk/public/JZY/CVPR_Competition/fudan_vlux_V1/Dataset-FHM/dev.jsonl', 'r',
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

        # 定义方法将指定路径图片转为Base64编码
        def encode_image(img_path):
            with open(img_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

        current_time = datetime.now()
        time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

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
                image_path = f'/newdisk/public/JZY/CVPR_Competition/fudan_vlux_V1/Dataset-FHM/{img_name_item_replaced}'
                base64_image = encode_image(image_path)

                response = self.client.chat.completions.create(
                    # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
                    model="ep-20250427174648-pz2m7",
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
                )

                response_risk_detection = response.choices[0].message.content
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
                        f'/newdisk/public/JZY/CVPR_Competition/fudan_vlux_V1/Dataset-FHM/Doubao_Bad_Cases/failed_dev_examples_memes_{time_str}.csv',
                        index=False)
                det_pre = (1.0 * count_match) / count_total
                loop_find_calling.set_postfix(pred_res=pred_res, label_res=label_res, count_match=count_match, count_total=count_total, acc=det_pre)
                # print(det_pre)

    def auto_prompt_tune(self, rounds=10, batch_size=8, eval_steps=10):
        """
        自动调优提示，经过指定的轮次预测和真实标签对比，根据表现决定是否更新prompt。
        """
        # 基本训练数据的加载
        current_time = datetime.now()
        time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
        prompt_update_dir_path = os.path.join("/newdisk/public/JZY/CVPR_Competition/prompts/", f"update_prompt_{time_str}")
        if not os.path.exists(prompt_update_dir_path):
            os.makedirs(prompt_update_dir_path)
        list_res = []
        with open('/newdisk/public/JZY/CVPR_Competition/fudan_vlux_V1/Dataset-FHM/train.jsonl', 'r',
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
                issues, false_predictions_text = self.analyze_issues(predictions)

                # 提取<reason>中的文本
                reason = re.search(r'<reason>(.*?)</reason>', issues, re.DOTALL)
                reason_text = reason.group(1).strip() if reason else "no reason found"

                # 提取<new_prompt>中的文本
                new_prompt = re.search(r'<new_prompt>(.*?)</new_prompt>', issues, re.DOTALL)
                new_prompt_text = new_prompt.group(1).strip() if new_prompt else self.prompt

                # 输出查看
                # print("Reason:\n", reason_text, "\n")
                # print("New Prompt:\n", new_prompt_text)

                if "{TEXT_DESC_OF_PIC}" not in new_prompt_text:
                    new_prompt_text += "{TEXT_DESC_OF_PIC}"
                update_file_path = os.path.join(prompt_update_dir_path, f"update_{count_steps}")
                if not os.path.exists(update_file_path):
                    os.makedirs(update_file_path)
                with open(update_file_path + "/reason.txt", "w", encoding="utf-8") as f:
                    f.write(reason_text)
                    f.close()
                with open(update_file_path + "/prompt.txt", "w", encoding="utf-8") as f:
                    f.write(new_prompt_text)
                    f.close()
                with open(update_file_path + "/false_prediction.txt", "w", encoding="utf-8") as f:
                    f.write(false_predictions_text)
                    f.close()
                self.prompt = new_prompt_text
                if count_steps % eval_steps == 0:
                    # list_res_eval = []
                    # with open('/newdisk/public/JZY/CVPR_Competition/fudan_vlux_V1/Dataset-FHM/dev.jsonl', 'r',
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

        # 你也可以选择保存新的提示到磁盘
        updated_prompt_path = os.path.join(self.prompt_dir, f'prompt_risk_detection_updated.txt')
        with open(updated_prompt_path, "w", encoding="utf-8") as f:
            f.write(updated_prompt)


if __name__ == "__main__":
    # 示例输入数据与真实标签
    image_descriptions = [
        "A child standing near the edge of the rooftop.",
        "A flower garden under clear sunny sky."
    ]
    true_labels = ['Yes', 'No']  # 真实标签

    # 实例化调优器并执行自动调优
    tuner = AutoPromptTuner(initial_prompt_dir="/newdisk/public/JZY/CVPR_Competition/prompts/prompt_0/",
                            prompt_filename="prompt_risk_detection.txt",
                            update_prompt_file="/newdisk/public/JZY/CVPR_Competition/prompts/analyze_issues_prompt.txt")
    # tuner.auto_prompt_tune(image_descriptions, true_labels, rounds=10)
    tuner.auto_prompt_tune()
    # print(prompt)
