import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 選擇要使用的模型
model_choice = "pegasus-large_FT100"  
# 可選: "bart-large_FT100", "bart-base_FT100", "pegasus-large_FT100", "t5-base", "t5-large", "longt5-base", "longt5-large"

# 根據選擇設置模型名稱和參數
if model_choice == "bart-large_FT100":
    model_name = "facebook/bart-large-cnn"
    max_input_length = 1024
    max_output_length = 128
    use_prefix = False
elif model_choice == "bart-large":
    model_name = "facebook/bart-large"
    max_input_length = 1024
    max_output_length = 128
    use_prefix = False
elif model_choice == "bart-base_FT100":
    model_name = "ainize/bart-base-cnn" 
    max_input_length = 1024
    max_output_length = 128
    use_prefix = False
elif model_choice == "bart-base":
    model_name = "facebook/bart-base"
    max_input_length = 1024
    max_output_length = 128
    use_prefix = False
elif model_choice == "pegasus-large_FT100":
    model_name = "google/pegasus-cnn_dailymail"
    max_input_length = 1024
    max_output_length = 128
    use_prefix = False
elif model_choice == "pegasus-large":
    model_name = "google/pegasus-large"
    max_input_length = 1024
    max_output_length = 128
    use_prefix = False
elif model_choice == "t5-base":
    model_name = "google-t5/t5-base"
    max_input_length = 512
    max_output_length = 128
    use_prefix = True
elif model_choice == "t5-base_FT100":
    model_name = "flax-community/t5-base-cnn-dm"
    max_input_length = 512
    max_output_length = 128
    use_prefix = True
elif model_choice == "t5-large":
    model_name = "google-t5/t5-large"
    max_input_length = 512
    max_output_length = 128
    use_prefix = True
elif model_choice == "t5-large_FT100":
    model_name = "kssteven/T5-large-cnndm"
    max_input_length = 512
    max_output_length = 128
    use_prefix = True
elif model_choice == "flan-t5-large_FT100":
    model_name = "spacemanidol/flan-t5-large-cnndm"
    max_input_length = 512
    max_output_length = 128
    use_prefix = True
elif model_choice == "longt5-base":
    model_name = "google/long-t5-tglobal-base"
    max_input_length = 4096
    max_output_length = 128
    use_prefix = True
elif model_choice == "longt5-large":
    model_name = "google/long-t5-tglobal-large"
    max_input_length = 8192
    max_output_length = 128
    use_prefix = True
else:
    raise ValueError("Invalid model choice")

# 加載模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 加載數據集
with open('./datasets/cnndm/test/cnndm_test_200.json') as fs:
    original_dataset = json.load(fs)

for data in original_dataset:
    text = data["src"]
    
    # 根據模型需求添加前綴
    if use_prefix:
        text = "summarize: " + text

    # 對輸入進行編碼
    input_tokens = tokenizer.encode(text, return_tensors="pt", max_length=max_input_length, truncation=True)

    # 生成摘要
    summary_ids = model.generate(input_tokens, max_length=max_output_length, num_beams=8, early_stopping=True)
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # 將結果保存到數據中
    data[model_choice] = summary_text

    print("No.", data["id"])
    print("摘要:")
    print(summary_text)

# 保存結果
output_file = f'./output/cnndm_{model_choice}.json'
with open(output_file, "w") as f:
    json.dump(original_dataset, f, indent=4)
    print(f"Save to {output_file}")