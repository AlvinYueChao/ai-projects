from unsloth import FastLanguageModel, FastModel
import torch
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from unsloth import to_sharegpt
from unsloth import standardize_sharegpt, get_chat_template
from unsloth import is_bfloat16_supported
from unsloth import apply_chat_template
from transformers import TrainingArguments
from transformers import TextStreamer

max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "../qwen25-3b",
    load_in_4bit = False,
    load_in_8bit = False,
    full_finetuning = False,
    use_gradient_checkpointing = True,
    local_files_only=True,
    # output_loading_info=True,
    # ignore_mismatched_sizes=True,
)

model = FastLanguageModel.get_peft_model(
   model,
   r = 16,
   target_modules = ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj",],
   lora_alpha = 32,
   lora_dropout = 0,
   bias = "none",
#    qat_scheme = "int4",
   use_gradient_checkpointing = "unsloth",
   random_state = 3407,
   use_rslora = False,
   loftq_config = None,
)

dataset = load_dataset('json', data_files='fine_tuning_dataset.jsonl', split='train')
print(dataset.column_names)

# chat_template = """Below are FFmpeg transcode log content that describe the result of the transcoding. Analyze the log content and provide the result of the transcoding in json string.

# ### Instruction:
# {INPUT}

# ### Response:
# {OUTPUT}"""

dataset = dataset.rename_column("messages", "conversations")
dataset = standardize_sharegpt(dataset)

tokenizer = get_chat_template(tokenizer, chat_template = "chatml")

dataset = apply_chat_template(
    dataset,
    tokenizer=tokenizer,
    # default_system_message = "You are a helpful assistant", << [OPTIONAL]
)


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 16,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        #max_steps = 60,
        num_train_epochs = 2, # For longer training runs!
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

FastLanguageModel.for_inference(model) # Enable native 2x faster inference

messages = [                    # Change below!
    {"role": "user", "content": "ffmpeg version N-random-g7578330541d shared. Build: gcc-latest.\nInput #0, mov,mp4, from 'input_video_38.mp4':\n  Duration: 00:00:09.00, start: 0.000000, bitrate: 785 kb/s\n    Stream #0:0: Video: h264 (Main), yuv420p, 1920x1080, 27 fps\n    Stream #0:1: Audio: aac, 48000 Hz, stereo\n[libx264 @ 0x48b6c433360b9432] PSNR Y:44.20 U:40.09 V:47.38 Avg:43.83 Global:35.31\nOutput #0, mp4, to 'output_video_51.mp4':\n  Stream #0:0: Video: h264 (H.264 Main)\n  Stream #0:1: Audio: aac\nframe=  243 fps= 27 q=28.0 size=     4631kB time=00:00:09.00 bitrate=4024kbits/s speed=1.5x\nvideo:4167kB audio:463kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.000%\n"},
]
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,
    return_tensors = "pt",
).to("cuda")

text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(input_ids, streamer = text_streamer, max_new_tokens = 128, pad_token_id = tokenizer.eos_token_id)

print("saving...")
model.save_pretrained("finetuned_lora")
tokenizer.save_pretrained("finetuned_lora")
print("saved")