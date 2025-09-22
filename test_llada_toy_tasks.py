import os.path

import torch
import itertools
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.llada.modeling_llada import LLaDAModelLM
from datasets import load_dataset
from src.dream import DreamModel
from math_verify import LatexExtractionConfig, parse, verify
import numpy as np
from src.llada.generate import get_num_transfer_tokens, add_gumbel_noise, get_num_transfer_tokens_maskgit
from src.mdlm_generation_utils import diffusion_generate
from src.open_r1.utils.trainer_utils import profiling_context, CustomDistributedSampler
import torch.distributed as dist
import torch.nn.functional as F
import random
import pandas as pd
from latex2sympy2_extended import NormalizationConfig
from tqdm import tqdm
from visualize_diffusion import DiffusionModelVisualizer
from torch.utils.data import DataLoader
from eval.sudoku import SudokuDataset
from eval.countdown import CTDDataset
from eval.gsm8k import GSM8KDataset
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed, broadcast

def setup_ddp():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()
    
DATASET_MAP = {
    "gsm8k": GSM8KDataset,
    "countdown": CTDDataset,
    "sudoku": SudokuDataset,
}

num_evals = {"gsm8k": -1, "math": -1, "countdown": -1, "sudoku": -1}

def visualize_intermediates(intermediates, intermediate_inputs, intermediate_correct_cnt, vis_file_name):
    # Create visualizer
    visualizer = DiffusionModelVisualizer(cmap_name='plasma')
    # Load data
    responses = []
    for response in intermediates:
        resp_tokens = tokenizer.convert_ids_to_tokens(response.cpu()[0, -args.gen_length:])
        new_resp_tokens = []
        for token in resp_tokens:
            if token == "Ċ":
                new_resp_tokens.append("Ċ")
            elif token == "Ġ":
                new_resp_tokens.append("Ġ")
            elif token.startswith("Ġ"):
                new_resp_tokens.append(token.lstrip("Ġ"))
            else:
                new_resp_tokens.append(token)
        responses.append(new_resp_tokens)
    inputs = []
    for input_tokens in intermediate_inputs:
        inp_tokens = tokenizer.convert_ids_to_tokens(input_tokens.cpu()[0, -args.gen_length:])
        new_inp_tokens = []
        for token in inp_tokens:
            if token == "Ċ":
                new_inp_tokens.append("Ċ")
            elif token == "Ġ":
                new_inp_tokens.append("Ġ")
            elif token.startswith("Ġ"):
                new_inp_tokens.append(token.lstrip("Ġ"))
            elif token == "<|mdm_mask|>":
                new_inp_tokens.append("[MASK]")
            else:
                new_inp_tokens.append(token)
        inputs.append(new_inp_tokens)
    confidence_scores = [
        torch.where(i[0, -args.gen_length:].cpu() == float("-inf"), 1, i[0, -args.gen_length:].cpu()).numpy().tolist()
        for i in confidences]
    visualizer.load_data(responses, confidence_scores,
                         ["Correct" if i in intermediate_correct_cnt else "Wrong" for i in range(len(inputs))], inputs=inputs)
    # Create web visualization
    visualizer.create_web_visualization(vis_file_name)

def parse_solution(solution):
    gold_parsed = parse(
        solution,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )
    if len(gold_parsed) == 0:
        gold_parsed = parse(
            "$" + solution + "$",
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
    return gold_parsed

if __name__ == '__main__':
    local_rank = setup_ddp()
    device = local_rank
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, choices=["gsm8k", "countdown", "sudoku", "game24"], default="sudoku")
    parser.add_argument("--split", default="test")
    parser.add_argument("--gen_length", type=int, default=512)
    parser.add_argument("--model_path", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--lora_path", default=None, type=str)
    parser.add_argument("--mode", default="linear", choices=["linear", "cosine", "pow2", "pow3", "pow0.5", "log", "exp"])
    parser.add_argument("--log_visualizations", default=False, action="store_true")
    parser.add_argument("--rcr", default=False, action="store_true")
    parser.add_argument("--conf_alg", default="origin", choices=["random", "llada", "topk_margin", "entropy"])
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0)
    args = parser.parse_args()

    # model_path = "data/LLaDA-8B-Instruct-GDPO-random-test-diff-reward/"
    # model_path = "data/LLaDA-8B-Instruct-GDPO-numina-adv-v3"
    # model_path = "GSAI-ML/LLaDA-8B-Instruct"
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, cache_dir="./cache")
    except:
        tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True, cache_dir="./cache")
    if "llada" in args.model_path.lower():
        MODEL_MODULE = LLaDAModelLM
    elif "dream" in args.model_path.lower():
        MODEL_MODULE = DreamModel
    else:
        raise NotImplementedError(f"Model {args.model_path} not supported yet")
    model = MODEL_MODULE.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                         cache_dir="./cache", device_map=device)
    if args.lora_path is not None:
        model.load_adapter(args.lora_path)
    
    ds = DATASET_MAP[args.dataset_name](
        tokenizer,
        subsample=num_evals[args.dataset_name],
        num_examples=0, # We don't do few_shot currently
        add_reasoning=False,  # prefill for all models
    )

    dataset_name = args.dataset_name
    all_results = []
    dataloader = DataLoader(
        ds,
        batch_size=1,
        sampler=CustomDistributedSampler(ds, shuffle=False),
        collate_fn=ds.collate_fn,
    )
    for p_index, d in tqdm(enumerate(dataloader), total=len(dataloader)):
        # problem_index = random.randint(0, len(ds) - 1)
        # problem_index = 5371
        # problem, answer, solution = ds[problem_index]["problem"], ds[problem_index]["answer"], ds[problem_index]['solution']

        problem, solution = d["questions"][0], d["answers"][0]
        unique_id = d.get("unique_id", [p_index])[0]
        unique_id = unique_id.replace("/", "_").rstrip(".json") if isinstance(unique_id, str) else unique_id
        input_ids = d["input_ids"].to(device)
        block_sizes = [32, 512]
        steps = [128, 64, 256]
        for block_size in block_sizes:
            for step in steps:
                # for block_size, step in sampling_settings:
                if step % (args.gen_length / block_size) != 0:
                    break
                out, intermediates, confidences, intermediate_inputs = diffusion_generate(model, input_ids,
                                                                                          mask_id=model.config.mask_token_id,
                                                                                          gen_length=args.gen_length,
                                                                                          block_length=block_size,
                                                                                          steps=step,
                                                                                          temperature=args.temperature,
                                                                                          conf_alg=args.conf_alg,
                                                                                          rcr=args.rcr,
                                                                                          top_p=args.top_p,
                                                                                          top_k=args.top_k)
                model_answer = tokenizer.batch_decode(out, skip_special_tokens=True)[0]

                intermediate_answers = tokenizer.batch_decode(
                    torch.cat(intermediates, dim=0),
                    skip_special_tokens=True)
                answer_correct = ds.validate(model_answer, solution, question=problem)
                if args.dataset_name == "sudoku":
                    answer_correct = answer_correct[-1] == 1.0
                # print(f"Question {problem_index} is {str(answer_correct)}")
                # intermediate_correct = False
                intermediate_correct_cnt = []
                for i, intermediate_answer in enumerate(intermediate_answers):
                    inter_answer_correct = ds.validate(intermediate_answer, solution, question=problem)
                    if args.dataset_name == "sudoku":
                        inter_answer_correct = inter_answer_correct[-1] == 1.0
                    if inter_answer_correct:
                        # intermediate_correct = True
                        intermediate_correct_cnt.append(i)
                    # if verify(gold_parsed, intermediate_parsed) and not answer_correct:
                    #     print(f"Correct prediction at timestep {i} for question {problem_index}")
                if (not answer_correct) and len(intermediate_correct_cnt) > 0 and args.log_visualizations:
                    vis_file_name = f"logs/visualizations/htmls/{args.model_path.rstrip('/').split('/')[-1] if args.lora_path is None else args.lora_path.rstrip('/').split('/')[-1]}_{args.mode}_{step}_{block_size}_{unique_id}_remask_{args.conf_alg}_RCR_{str(args.rcr)}.html"
                    visualize_intermediates(intermediates, intermediate_inputs, intermediate_correct_cnt, vis_file_name)
                all_results.append({"id": unique_id,"problem": problem, "solution": solution, "model_answer": model_answer,
                                    "block_size": block_size, "step": step,
                                    "answer_correct": answer_correct, "intermediate_correct": intermediate_correct_cnt})
    dist.barrier()
    file_name = f"./local_rank_{dist.get_rank()}_{dataset_name.split('/')[-1]}_{args.model_path.rstrip('/').split('/')[-1] if args.lora_path is None else args.lora_path.rstrip('/').split('/')[-1]}_{args.mode}_{args.gen_length}_remask_{args.conf_alg}_RCR_{str(args.rcr)}.csv"
    pd.DataFrame(all_results).to_csv(os.path.join("./logs", file_name), index=False)
    if dist.get_rank() == 0:
        dfs = []
        all_file_name = file_name = f"./{dataset_name.split('/')[-1]}_{args.model_path.rstrip('/').split('/')[-1] if args.lora_path is None else args.lora_path.rstrip('/').split('/')[-1]}_{args.mode}_{args.gen_length}_remask_{args.conf_alg}_RCR_{str(args.rcr)}.csv"
        for rank in range(dist.get_world_size()):
            file_name = f"./local_rank_{rank}_{dataset_name.split('/')[-1]}_{args.model_path.rstrip('/').split('/')[-1] if args.lora_path is None else args.lora_path.rstrip('/').split('/')[-1]}_{args.mode}_{args.gen_length}_remask_{args.conf_alg}_RCR_{str(args.rcr)}.csv"
            dfs.append(pd.read_csv(os.path.join("./logs", file_name)))
            os.remove(os.path.join("./logs", file_name))
        pd.concat(dfs).to_csv(os.path.join("./logs", all_file_name), index=False)
    cleanup_ddp()