import os
from dataclasses import dataclass, field, asdict
from typing import Optional
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datasets import load_dataset
import transformers
import trl

@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen2.5-32B-Instruct")
    block_size: int = field(default=32768)
    train_size: Optional[float] = field(default=1)
    dagger: bool = field(default=False)
    def __post_init__(self):
        os.environ["WANDB_DISABLED"] = "true"

def train():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")

    # loading model
    kwargs = {"use_cache": False}
    model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    
    dataset = load_dataset("simplescaling/s1K_tokenized")
    dataset = dataset['train']
    train_test_split = dataset.train_test_split(train_size=config.train_size, seed=42, shuffle=True)
    train_dataset = train_test_split['train']
    test_dataset =  train_test_split['test']
    print(f"train dataset: {len(train_dataset)} test_dataset: {len(test_dataset)}")
    
    instruction_template = "<|im_start|>user"
    response_template = "<|im_start|>assistant\n"
    tokenizer.pad_token = "<|fim_pad|>"

    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )
    args.dataset_text_field = 'text'
    args.max_seq_length = config.block_size
    trainer = trl.SFTTrainer(
        model,
        train_dataset = train_dataset,
        eval_dataset= test_dataset,
        args=args,
        data_collator=collator
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()
    
if __name__ == "__main__":
    train()