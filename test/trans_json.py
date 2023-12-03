import os
import json

def convert_json_to_txt(json_directory, output_directory):
    # 获取目录下所有JSON文件
    json_files = [file for file in os.listdir(json_directory) if file.endswith('.json')]

    for json_file in json_files:
        json_path = os.path.join(json_directory, json_file)

        with open(json_path, 'r') as f:
            json_data = json.load(f)

        try:
            # 提取2D姿势关键点和置信度
            keypoints_2d = json_data["people"][0]["pose_keypoints_2d"]
            confidence_values = keypoints_2d[2::3]  # 置信度值
            xy_coordinates = [(keypoints_2d[i], keypoints_2d[i + 1]) for i in range(0, len(keypoints_2d), 3)]

            # 获取置信度排名前18的坐标
            sorted_coordinates = [coord for _, coord in sorted(zip(confidence_values, xy_coordinates), reverse=True)[:18]]

            # 将结果写入txt文件，文件名保持不变
            output_txt_path = os.path.join(output_directory, f"{os.path.splitext(json_file)[0]}.txt")

            with open(output_txt_path, 'w') as txt_output:
                for coord in sorted_coordinates:
                    txt_output.write(f"{coord[0]:.15e} {coord[1]:.15e}\n")
        except:
            continue

# 替换为你的实际目录
json_directory = 'E:\Documents\PythonScripts\PIDM\dataset\deepfashion\pose'
output_directory = 'E:\Documents\PythonScripts\PIDM\dataset\deepfashion\pose'

convert_json_to_txt(json_directory, output_directory)
