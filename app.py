import torch
from PIL import Image
from transformers import BertTokenizer, ViTFeatureExtractor
import gradio as gr
import json
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from config.Config_base import Config_base
from model.MHKE import MHKE, MHKE_CLIP

# --- Part 1: 模型和预处理器加载 (后端核心逻辑) ---
print("正在加载配置和深度融合模型，请稍候...")

model_to_load = "MHKE"
task_name = "task_2"
config = Config_base(model_to_load, task_name)

tokenizer = BertTokenizer.from_pretrained(config.roberta_path)
image_extractor = ViTFeatureExtractor.from_pretrained(config.vit_path)
if model_to_load == "clip":
    from transformers import CLIPProcessor

    processor = CLIPProcessor.from_pretrained(config.clip_path)
    tokenizer = processor
    image_extractor = processor

print(f"正在实例化改进后的深度融合模型: {model_to_load}")
if model_to_load == "MHKE":
    model = MHKE(config).to(config.device)
    checkpoint_path = '{}/ckp-MHKE_B-32_E-10_Lr-1e-05_w-0.5_task_2_add_Fusion-BEST.tar'.format(config.checkpoint_path)
elif model_to_load == "clip":
    model = MHKE_CLIP(config).to(config.device)
    checkpoint_path = '{}/ckp-clip_B-32_E-10_Lr-1e-05_w-0.5_task_2_add_Fusion-BEST.tar'.format(config.checkpoint_path)
else:
    raise ValueError("未知的模型名称")

try:
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"成功加载深度融合模型权重: {checkpoint_path}")
except FileNotFoundError:
    print(f"错误：找不到模型文件 {checkpoint_path}。请确认您的最佳模型文件名是否正确。")
    exit()

model.eval()

class_names = {0: "无害", 1: "刻板印象与偏见", 2: "色情与性暗示", 3: "侮辱与攻击", 4: "自嘲与消极"}
class_labels = list(class_names.values())

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# --- Part 2: 核心预测函数 ---
def predict(image, text, text_description, meme_description):
    if image is None or not text:
        return {label: 0 for label in class_labels}

    # 如果描述为空，则使用主文本
    if not text_description:
        text_description = text
    if not meme_description:
        meme_description = text

    if model_to_load == "clip":
        inputs = processor(text=text, images=image, return_tensors="pt", padding="max_length",
                           max_length=config.pad_size, truncation=True)
        td_inputs = processor(text=text_description, return_tensors="pt", padding="max_length",
                              max_length=config.pad_size, truncation=True)
        md_inputs = processor(text=meme_description, return_tensors="pt", padding="max_length",
                              max_length=config.pad_size, truncation=True)
        model_inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"],
                        "image_tensor": inputs["pixel_values"], "text_discription_input_ids": td_inputs["input_ids"],
                        "text_discription_attention_mask": td_inputs["attention_mask"],
                        "meme_discription_input_ids": md_inputs["input_ids"],
                        "meme_discription_attention_mask": md_inputs["attention_mask"], }
    else:
        image_inputs = image_extractor(image, return_tensors='pt')
        text_inputs = tokenizer(text, max_length=config.pad_size, padding="max_length", truncation=True,
                                return_tensors="pt")
        td_inputs = tokenizer(text_description, max_length=config.pad_size, padding="max_length", truncation=True,
                              return_tensors="pt")
        md_inputs = tokenizer(meme_description, max_length=config.pad_size, padding="max_length", truncation=True,
                              return_tensors="pt")
        model_inputs = {"input_ids": text_inputs["input_ids"], "attention_mask": text_inputs["attention_mask"],
                        "image_tensor": image_inputs["pixel_values"],
                        "text_discription_input_ids": td_inputs["input_ids"],
                        "text_discription_attention_mask": td_inputs["attention_mask"],
                        "meme_discription_input_ids": md_inputs["input_ids"],
                        "meme_discription_attention_mask": md_inputs["attention_mask"], }

    for key, value in model_inputs.items():
        model_inputs[key] = value.to(config.device)

    with torch.no_grad():
        logit = model(**model_inputs).cpu()

    probabilities = torch.softmax(logit, dim=1).squeeze().numpy()
    return {class_labels[i]: float(probabilities[i]) for i in range(len(class_labels))}


# --- Part 3: 批量验证与绘图函数 ---
def evaluate_on_demo_data():
    print("开始在demo数据集上进行批量验证...")
    try:
        with open('demo_data.json', 'r', encoding='utf-8') as f:
            demo_data = json.load(f)
    except FileNotFoundError:
        return pd.DataFrame(), "错误: demo_data.json 未找到。", None, None, None

    results_list, true_labels_str, pred_labels_str = [], [], []
    for i, item in enumerate(demo_data):
        image_path = os.path.join(config.meme_path, item['new_path'])
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"警告: 找不到图片 {image_path}, 跳过。")
            continue

        # --- MODIFIED: 修复 UnboundLocalError ---
        text = item['text']
        text_desc = item.get('text_discription', text)
        meme_desc = item.get('meme_discription', text)
        # --- 修改结束 ---

        prediction_scores = predict(image, text, text_desc, meme_desc)
        predicted_label = max(prediction_scores, key=prediction_scores.get)
        true_label = class_names.get(item['type'], "未知")
        true_labels_str.append(true_label)
        pred_labels_str.append(predicted_label)
        results_list.append(
            {"图片": item['new_path'], "文本": text, "真实类别": true_label, "预测类别": predicted_label,
             "是否正确": "✔️" if true_label == predicted_label else "❌"})
        yield pd.DataFrame(results_list), f"处理中... {i + 1}/{len(demo_data)}", None, None, None

    if not results_list:
        return pd.DataFrame(), "错误：没有成功处理任何数据。", None, None, None

    report = classification_report(true_labels_str, pred_labels_str, labels=class_labels, output_dict=True,
                                   zero_division=0)
    summary_text = f"✅ 批量验证完成！\n\n\n总览:\n\n- 总准确率 : {report['accuracy']:.2%}\n- 宏平均F1分数: {report['macro avg']['f1-score']:.4f}\n\n\n图表展示了更详细的性能分析。"

    cm = confusion_matrix(true_labels_str, pred_labels_str, labels=class_labels)
    fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=class_labels, yticklabels=class_labels, ax=ax_cm)
    ax_cm.set_xlabel('预测类别');
    ax_cm.set_ylabel('真实类别');
    ax_cm.set_title('混淆矩阵')
    plt.setp(ax_cm.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    plt.setp(ax_cm.get_yticklabels(), rotation=0)
    fig_cm.tight_layout()

    report_df = pd.DataFrame(report).transpose()
    metrics_df = report_df.loc[class_labels, ['precision', 'recall', 'f1-score']]
    fig_metrics, ax_metrics = plt.subplots(figsize=(12, 7))
    metrics_df.plot(kind='bar', ax=ax_metrics, colormap='viridis')
    ax_metrics.set_title('各类别性能指标');
    ax_metrics.set_ylabel('分数');
    ax_metrics.set_xlabel('类别')
    plt.setp(ax_metrics.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    ax_metrics.grid(axis='y', linestyle='--', alpha=0.7)
    fig_metrics.tight_layout()

    class_counts = pd.Series(true_labels_str).value_counts().reindex(class_labels, fill_value=0)
    fig_dist, ax_dist = plt.subplots(figsize=(8, 8))
    ax_dist.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=90,
                colors=sns.color_palette("pastel"))
    ax_dist.axis('equal');
    ax_dist.set_title('Demo数据集中各类别分布')
    fig_dist.tight_layout()

    print("批量验证完成！")
    yield pd.DataFrame(results_list), summary_text, fig_cm, fig_metrics, fig_dist


# --- Part 4: Gradio Web UI 界面构建 ---
with gr.Blocks(theme=gr.themes.Soft(), title="UGC多模态智能审核系统") as demo:
    gr.Markdown(
        """ # 🎓 **UGC多模态智能审核系统 (深度融合改进版)**\n本系统基于改进的 **MHKE** 模型。\n**核心创新**: 引入了基于 **双向交叉注意力** 的深度融合模块，实现了对文本与图像内容的严格联合表征学习，有效弥补了原始模型浅层融合的不足。 """)
    with gr.Tabs():
        with gr.TabItem("交互式预测"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. 输入数据")
                    input_image = gr.Image(type="pil", label="上传图片")
                    input_text = gr.Textbox(lines=2, label="图片中的文本")
                    input_text_desc = gr.Textbox(lines=2, label="文本描述", placeholder="若留空，将使用文本代替")
                    input_meme_desc = gr.Textbox(lines=2, label="图像描述", placeholder="若留空，将使用文本代替")
                    predict_btn = gr.Button("🚀 开始预测", variant="primary", scale=2)
                with gr.Column(scale=1):
                    gr.Markdown("### 2. 预测结果")
                    output_label = gr.Label(label="分类概率", num_top_classes=5)

            gr.Examples(
                examples=[
                    [os.path.join(config.meme_path, "10768.jpg"), "要说傻逼谁能比过你", "这是一句带有侮辱性质的话语...",
                     "这是一幅卡通化的图画..."],
                    [os.path.join(config.meme_path, "79.jpg"), "你今天忘记手冲了。闭嘴我戒了。",
                     "这句话可能是两个人对话...", "一张漫画..."],
                    [os.path.join(config.meme_path, "11541.jpg"), "想過離開！ 是因為那些姿態那些旁白那些傷害...",
                     "这段文本表达了作者内心的痛苦和无助...", "一位戴着耳机、心碎表情的卡通蜜蜂..."],
                ],
                inputs=[input_image, input_text, input_text_desc, input_meme_desc],
                label="示例数据 (点击自动填充)"
            )
        with gr.TabItem("批量数据验证"):
            gr.Markdown("点击下方按钮，系统将读取 `demo_data.json` 中的全部数据进行预测，并展示详细结果与性能评估图表。")
            eval_btn = gr.Button("🔍 运行全体验证集", variant="primary")
            with gr.Row():
                with gr.Column(scale=1):
                    eval_status = gr.Textbox(label="📊 性能摘要", lines=6)
                    class_distribution_plot = gr.Plot(label="类别分布")
                with gr.Column(scale=1):
                    confusion_matrix_plot = gr.Plot(label="混淆矩阵")
                    metrics_bar_chart_plot = gr.Plot(label="各类别性能指标")
            results_dataframe = gr.DataFrame(headers=["图片", "文本", "真实类别", "预测类别", "是否正确"],
                                             label="详细预测结果", wrap=True)

    predict_btn.click(fn=predict, inputs=[input_image, input_text, input_text_desc, input_meme_desc],
                      outputs=output_label)
    eval_btn.click(fn=evaluate_on_demo_data,
                   outputs=[results_dataframe, eval_status, confusion_matrix_plot, metrics_bar_chart_plot,
                            class_distribution_plot])

print("Gradio界面准备就绪，正在启动...")
demo.launch(share=True)