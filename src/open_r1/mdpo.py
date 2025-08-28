# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import json
import sys
from dataclasses import dataclass, field
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import datasets
import numpy as np
import torch
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from src.open_r1.configs import MDPOConfig
from src.open_r1.mdpo.mdpo_trainer import MDPOTrainer
from src.open_r1.rewards import (
    accuracy_reward,
    code_reward,
    format_reward,
    get_code_format_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    len_reward,
    reasoning_steps_reward,
    tag_count_reward,
    verify_math,
    verify_countdown,
    countdown_reward,
    sudoku_reward, verify_sudoku
)
from src.open_r1.utils import get_tokenizer
from src.open_r1.utils.callbacks import get_callbacks
from src.open_r1.utils.wandb_logging import init_wandb_training
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from llada.modeling_llada import LLaDAModelLM
import deepspeed
deepspeed.ops.op_builder.CPUAdamBuilder().load()
logger = logging.getLogger(__name__)


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', 'tag_count', 'code', 'code_format'.
        cosine_min_value_wrong (`float`):
            Minimum reward for cosine scaling for wrong answers.
        cosine_max_value_wrong (`float`):
            Maximum reward for cosine scaling for wrong answers.
        cosine_min_value_correct (`float`):
            Minimum reward for cosine scaling for correct answers.
        cosine_max_value_correct (`float`):
            Maximum reward for cosine scaling for correct answers.
        cosine_max_len (`int`):
            Maximum length for cosine scaling.
        code_language (`str`):
            Language for code format reward.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', tag_count', 'code', 'code_format'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "Maximum (negative) penalty for for repetition penalty reward"},
    )
    code_language: str = field(
        default="python",
        metadata={
            "help": "Language for code format reward. Based on E2B supported languages https://e2b.dev/docs/code-interpreting/supported-languages",
            "choices": ["python", "javascript", "r", "java", "bash"],
        },
    )
    lora: bool = field(default=False, metadata={"help": "Use LoRa or not"})
    peft_task_type: str = field(default="CAUSAL_LM")
    num_train_samples: int = field(default=3000, metadata={"help": "Number of training samples"})
    incremental_training: bool = field(default=False, metadata={"help": "Whether incrementally select ICFW data"})
    mixture_data: bool = field(default=False, metadata={"help": "Sample the data in a mixture of settings way"})
    ab_path: str = field(default=None, metadata={"help": "Path to preprocessed answer backslide data"})

def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)
    try:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config, cache_dir="cache")
        for split in dataset:
            if "messages" in dataset[split].column_names:
                dataset[split] = dataset[split].remove_columns("messages")
    except Exception as e:
        print(e)
    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)
    tokenizer.mask_token_id = 126336 #fixed
    # tokenizer.pad_token_id = tokenizer.mask_token_id
    # Get reward functions
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
        "code": code_reward,
        "code_format": get_code_format_reward(language=script_args.code_language),
        "tag_count": tag_count_reward,
        "countdown": countdown_reward,
        "sudoku": sudoku_reward
    }
    system_prompt = training_args.system_prompt
    if training_args.system_prompt_type == "format":
        system_prompt = "Let's first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer> and output the final answer within \\boxed{} inbetween the <answer> </answer> tags"
    elif training_args.system_prompt_type == "step_by_step":
        system_prompt = "Let's think step by step and output the final answer within \\boxed{}."
    elif training_args.system_prompt_type == "countdown":
        from eval.countdown import CTD_SYSTEM_PROMPT
        system_prompt = CTD_SYSTEM_PROMPT
    elif training_args.system_prompt_type == "sudoku":
        from eval.sudoku import SUDOKU_SYSTEM_PROMPT
        system_prompt = SUDOKU_SYSTEM_PROMPT
    elif training_args.system_prompt_type == "d1":
        system_prompt = """You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}. Respond in the following format: <reasoning> Your reasoning here </reasoning> <answer> \\boxed{...} </answer>" """
    training_args.system_prompt = system_prompt
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]
    is_countdown = "countdown" in script_args.dataset_name.lower()
    is_sudoku = "sudoku" in script_args.dataset_name.lower()
    # Format into conversation
    def make_conversation(example):
        prompt = []
        if training_args.system_prompt is not None:
            # prompt.append({"role": "system", "content": training_args.system_prompt})
            # LLaDA doesn't offer system prompt so we append the system prompt to the end of the user prompt
            example["problem"] += ("\n" + training_args.system_prompt)
        if "Base" in tokenizer.name_or_path:
            return {"prompt": example["problem"]}
        else:
            prompt.append({"role": "user", "content": example["problem"]})
            return {"prompt": prompt}

    def make_sudoku_conversation(example):
        prompt = []
        solution = example["Solution"]
        question = example["Puzzle"]
        problem = f"Solve the following Sudoku puzzle: {question}\n"
        if training_args.system_prompt is not None:
            # prompt.append({"role": "system", "content": training_args.system_prompt})
            # LLaDA doesn't offer system prompt so we append the system prompt to the end of the user prompt
            problem = training_args.system_prompt + "\n\n" + problem
        prompt.append({"role": "user", "content": problem})
        return {"prompt": prompt, "problem": problem, "solution": str(solution), "question": str(question)}
    def make_countdown_conversation(example):
        prompt = []
        problem = f"Numbers: {example['nums']}\nTarget: {example['target']}"
        if training_args.system_prompt is not None:
            # prompt.append({"role": "system", "content": training_args.system_prompt})
            # LLaDA doesn't offer system prompt so we append the system prompt to the end of the user prompt
            problem = training_args.system_prompt + "\n\n" + problem
        prompt.append({"role": "user", "content": problem})
        return {"prompt": prompt, "problem": problem, "solution": [{"nums": example['nums'], "target": example['target']}]}

    if script_args.dataset_name == "camel-ai/amc_aime_self_improving":
        dataset[script_args.dataset_train_split] = dataset[script_args.dataset_train_split].rename_column("groud_truth_solution", "solution")
        include_idx = list(range(len(dataset[script_args.dataset_train_split])))
    elif script_args.dataset_name in ["agentica-org/DeepScaleR-Preview-Dataset", "open-r1/OpenR1-Math-220k"]:
        dataset[script_args.dataset_train_split] = dataset[script_args.dataset_train_split].remove_columns(
            ["solution"])
        dataset[script_args.dataset_train_split] = dataset[script_args.dataset_train_split].rename_column(
            "answer", "solution")
        dataset[script_args.dataset_train_split] = dataset[script_args.dataset_train_split].remove_columns(
            [i for i in dataset[script_args.dataset_train_split].column_names if i not in ["problem", "solution", "prompt"]])
        include_idx = np.random.choice(np.arange(len(dataset[script_args.dataset_train_split])), size=script_args.num_train_samples,
                                       replace=False).tolist()
    elif is_sudoku:
        dataset = {}
        dataset["train"] = datasets.Dataset.from_csv("eval/dataset/4x4_sudoku_unique_puzzles.csv", keep_in_memory=True, split=script_args.dataset_train_split, cache_dir="./cache", features=datasets.Features({"Puzzle": datasets.Value("string"), "Solution": datasets.Value("string")}))

        include_idx = np.random.choice(np.arange(len(dataset[script_args.dataset_train_split])),
                                       size=script_args.num_train_samples, replace=False).tolist()
    else:
        include_idx = np.random.choice(np.arange(len(dataset[script_args.dataset_train_split])), size=script_args.num_train_samples,replace=False).tolist()
    dataset = dataset[script_args.dataset_train_split].select((
        i for i in range(len(dataset[script_args.dataset_train_split]))
        if i in set(include_idx)
    ))
    if is_countdown:
        dataset_mapping_fn = make_countdown_conversation
    elif is_sudoku:
        dataset_mapping_fn = make_sudoku_conversation
    else:
        dataset_mapping_fn = make_conversation
    dataset = dataset.map(dataset_mapping_fn, keep_in_memory=True)
    # Evaluation dataset processing
    if is_sudoku:
        eval_dataset = datasets.Dataset.from_csv("eval/dataset/4x4_test_sudoku.csv", keep_in_memory=True, split=script_args.dataset_train_split, cache_dir="./cache", features=datasets.Features({"Puzzle": datasets.Value("string"), "Solution": datasets.Value("string")}))
        eval_dataset = eval_dataset.map(make_sudoku_conversation, keep_in_memory=True)
    elif is_countdown:
        eval_samples = []
        with open(f"eval/dataset/countdown_cd3_test.jsonl", "r") as f:
            for line in f:
                eval_data = json.loads(line)
                nums = [int(num) for num in eval_data["input"].split(",")]
                target = int(eval_data["output"])
                eval_samples.append({"nums": nums, "target": target})
        eval_dataset = datasets.Dataset.from_list(eval_samples)
        eval_dataset = eval_dataset.map(make_countdown_conversation, keep_in_memory=True)
        columns_to_check = ("nums", "target")

        def make_key(example, columns):
            key = []
            for col in columns:
                val = example[col]
                # Convert lists to tuples to make them hashable
                if isinstance(val, list):
                    val = tuple(val)
                key.append(val)
            return tuple(key)

        # Build sets of hashable keys
        set1 = set(make_key(row, columns_to_check) for row in dataset)
        set2 = set(make_key(row, columns_to_check) for row in eval_dataset)

        # Find duplicates based on those columns
        duplicates = set1.intersection(set2)
        dataset_filtered = dataset.filter(lambda ex: make_key(ex, columns_to_check) not in duplicates)
        logger.info(f"*** After filter out the samples in test set, {len(dataset_filtered)} samples are left ***")
        dataset = dataset_filtered
    else:
        eval_dataset = load_dataset("HuggingFaceH4/MATH-500", cache_dir="./cache")["test"]
        # eval_dataset = eval_dataset.select(range(100, 150))
        eval_dataset = eval_dataset.map(make_conversation, keep_in_memory=True)
        # eval_dataset = eval_dataset.remove_columns(["solution"])
        # eval_dataset = eval_dataset.rename_column("answer", "solution")
    if is_countdown:
        dataset.verify_fn = verify_countdown
        eval_dataset.verify_fn = verify_countdown
    elif is_sudoku:
        dataset.verify_fn = verify_sudoku
        eval_dataset.verify_fn = verify_sudoku
    else:
        dataset.verify_fn = verify_math
        eval_dataset.verify_fn = verify_math


    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        cache_dir="./cache"
    )

    model = LLaDAModelLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    if script_args.lora:
        peft_config = LoraConfig(
            task_type=script_args.peft_task_type,
            inference_mode=False,
            target_modules=["q_proj", "k_proj", "v_proj"],
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            modules_to_save=["classifier"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    # model_config = LLaDAConfig.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    # model_config.n_layers = 4
    # model = LLaDAModelLM(model_config)
    #############################
    # Initialize the MDPO trainer
    #############################
    trainer = MDPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset, #dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
        incremental_training=script_args.incremental_training,
        mixture_data=script_args.mixture_data,
        ab_path=script_args.ab_path
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    # metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        # metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, MDPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
