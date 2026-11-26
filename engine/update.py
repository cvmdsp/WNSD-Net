"""
@project:WaveDH-main
@Author:Wenjing Jiang,Hebei University
@file:update.py
@ide:PyCharm
@time:2024-10-03 19 :22 :39
"""
import json
import os.path
import os

def update_json(predictions_file_path,save_dir):
# 文件路径
    predictions_file_path = predictions_file_path  # 替换为你的预测文件路径
    labels_file_path = r'/home/cvmdsp3/opt/ship_datasets/HRSID/annotations/test.json'  # 替换为你的标签文件路径
    output_file_path = os.path.join(save_dir,"upredictions.json")  # 输出文件路径

    # 读取预测文件
    with open(predictions_file_path, 'r') as f:
        predictions = json.load(f)

    # 读取标签文件
    with open(labels_file_path, 'r') as f:
        labels = json.load(f)

    # 创建映射关系 image_id 到 id
    image_id_to_id = {image['file_name']: image['id'] for image in labels['images']}

    # 更新预测文件中的 image_id
    for prediction in predictions:
        image_id = f"{prediction['image_id']}"+'.jpg'
        # 找到对应的 id
        if image_id in image_id_to_id:
            prediction['image_id'] = image_id_to_id[image_id]
        else:
            print(f"Warning: image_id '{image_id}' not found in labels.")

    # 保存更新后的预测文件
    with open(output_file_path, 'w') as f:
        json.dump(predictions, f, indent=4)

    print(f"Updated predictions saved to {output_file_path}.")
