import json
import os

def generate_chatml_dataset(data_dir, output_file):
    # 1. 路径设置
    mapping_file = os.path.join(data_dir, 'ffmpeg_results.json')
    print(f"正在处理映射文件: {mapping_file}")
    
    if not os.path.exists(mapping_file):
        print(f"错误：找不到文件 {mapping_file}")
        return

    # 2. 读取映射文件内容
    with open(mapping_file, 'r', encoding='utf-8-sig') as f:
        try:
            results_data = json.load(f)
        except json.JSONDecodeError:
            print("错误：ffmpeg_results.json 格式不正确，请确保它是有效的 JSON 数组。")
            return

    # 确保数据是列表格式
    if not isinstance(results_data, list):
        results_data = [results_data]

    dataset = []

    # 3. 遍历并整理数据
    for entry in results_data:
        log_sub_dir = 'ffmpeg_logs'
        log_filename = entry.get('log_file')
        log_path = os.path.join(data_dir, log_sub_dir, log_filename)

        # 检查对应的 log 文件是否存在
        if os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8-sig') as f_log:
                log_content = f_log.read()

            # 提取需要的 4 个字段
            assistant_output_data = {
                "successful": entry.get("successful"),
                "psnr_value": entry.get("psnr_value"),
                "error_message": entry.get("error_message"),
                "resolution_steps": entry.get("resolution_steps")
            }

            # 4. 构建 ChatML 结构
            # 系统提示词可以根据你的微调目的修改
            chatml_entry = {
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个 FFmpeg 日志分析专家。请根据输入的日志内容，分析任务是否成功，并输出包含 successful、psnr_value、error_message 和 resolution_steps 的 JSON 字符串。"
                    },
                    {
                        "role": "user",
                        "content": log_content
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(assistant_output_data, ensure_ascii=False)
                    }
                ]
            }
            dataset.append(chatml_entry)
        else:
            print(f"跳过：未找到日志文件 {log_filename}")

    # 5. 保存为 JSONL 格式（每一行一个 JSON 对象，适合微调）
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in dataset:
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"处理完成！共生成 {len(dataset)} 条微调数据。")
    print(f"保存路径: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    # 根据你之前提供的路径
    DATA_DIRECTORY = "./data"
    OUTPUT_FILENAME = "fine_tuning_dataset.jsonl"
    
    generate_chatml_dataset(DATA_DIRECTORY, OUTPUT_FILENAME)