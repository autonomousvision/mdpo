import os.path

import torch
import itertools
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from src.llada.modeling_llada import LLaDAModelLM
from src.dream import DreamModel
from datasets import load_dataset, Features, Value
from math_verify import LatexExtractionConfig, parse, verify
from src.open_r1.utils.trainer_utils import profiling_context, CustomDistributedSampler
import torch.distributed as dist
from src.mdlm_generation_utils import diffusion_generate
import pandas as pd
from latex2sympy2_extended import NormalizationConfig
from tqdm import tqdm
from visualize_diffusion import DiffusionModelVisualizer
from torch.utils.data import DataLoader
from src.open_r1.rewards import code_reward
def setup_ddp():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()

def visualize_intermediates(intermediates, intermediate_inputs, intermediate_correct_cnt, vis_file_name):
    # Create visualizer
    visualizer = DiffusionModelVisualizer(cmap_name='plasma')
    # Load data
    responses = []
    for response in intermediates:
        resp_tokens = tokenizer.convert_ids_to_tokens(response.cpu()[0])
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
        inp_tokens = tokenizer.convert_ids_to_tokens(input_tokens.cpu()[0])
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
        torch.where(i[0].cpu() == float("-inf"), 1, i[0].cpu()).numpy().tolist() for i in confidences]
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

def collate_fn(batch):
    return {
        "problem_id": [item["problem_id"] for item in batch],
        "problem_statement": [item["problem_statement"] for item in batch],
        "verification_info": [item["verification_info"] for item in batch],
        "gold_standard_solution": [item["gold_standard_solution"] for item in batch]
    }

def test_reward_fn(ds):
    # run router locally: python scripts/e2b_router.py
    NUM_SAMPLES = 128
    samples = ds.select(range(NUM_SAMPLES))
    test_completions = [sample["gold_standard_solution"] for sample in samples]
    reward_kwargs = {"verification_info": [sample["verification_info"] for sample in samples]}
    rewards = code_reward(test_completions, provider_type="local", **reward_kwargs)
    assert rewards == [1.0] * NUM_SAMPLES

if __name__ == '__main__':
    local_rank = setup_ddp()
    device = local_rank
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="open-r1/verifiable-coding-problems-python", choices=["open-r1/verifiable-coding-problems-python", "open-r1/ioi", "open-r1/codeforces"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--system_prompt_type", default="open-r1")
    parser.add_argument("--gen_length", type=int, default=512)
    parser.add_argument("--model_path", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--lora_path", default=None, type=str)
    parser.add_argument("--mode", default="linear", choices=["linear", "cosine", "pow2", "pow3", "pow0.5", "log", "exp"])
    parser.add_argument("--log_visualizations", default=False, action="store_true")
    parser.add_argument("--rcr", default=False, action="store_true")
    parser.add_argument("--conf_alg", default="llada", choices=["random", "llada", "topk_margin", "entropy"])
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0)
    args = parser.parse_args()
    # model_path = "GSAI-ML/LLaDA-8B-Instruct"
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, cache_dir="./cache")
    except Exception as e:
        print(e)
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
    if args.system_prompt_type == "open-r1":
        system_prompt = "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer."
    else:
        system_prompt = "You are a helpful assistant."
    dataset_name = args.dataset_name
    if dataset_name == "DigitalLearningGmbH/MATH-lighteval":
        ds = load_dataset(dataset_name, cache_dir="./cache")["test"]
        import pandas as pd

        df = pd.read_csv("MATH-lighteval.csv")
        include_idx = df[(df["answer_correct"] == False) & (df["intermediate_correct"] == True)][
            "p_index"].unique().tolist()
        include_idx = pd.read_csv("MATH-lighteval_Llada_original.csv")["p_index"].unique().tolist()
        ds = ds.select((
            i for i in range(len(ds))
            if i in set(include_idx)
        ))
    elif dataset_name == "agentica-org/DeepScaleR-Preview-Dataset":
        ds = load_dataset(dataset_name, cache_dir="./cache")["train"]
        ds = ds.remove_columns(["solution"])
        ds = ds.rename_column("answer", "solution")
    else:
        ds = load_dataset(dataset_name, cache_dir="./cache")[args.split]
        # include_idx = [0,1,2,3,4,5,6,7,8,9] #[6] #93, 46, 19 some hard sample that we can use to test our idea
        # ds = ds.select((
        #     i for i in range(len(ds))
        #     if i in set(include_idx)
        # ))
    all_results = []
    ds = ds.remove_columns([i for i in ds.column_names if i not in ["problem_id", "problem_statement", "verification_info", "gold_standard_solution"]])
    test_reward_fn(ds)
    dataloader = DataLoader(
        ds,
        batch_size=1,
        sampler=CustomDistributedSampler(ds, shuffle=False),
        collate_fn=collate_fn
    )
    for p_index, d in tqdm(enumerate(dataloader), total=len(dataloader)):
        problem, solution, verification = d["problem_statement"][0], d["gold_standard_solution"][0], d["verification_info"][0]
        unique_id = d.get("problem_id", [p_index])[0]
        problem += "The code should be within ```python ... ```."
        # Add special tokens for the Instruct model. The Base model does not require the following two lines.
        m = [
             {"role": "user", "content": problem}, ]
        prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

        input_ids = tokenizer(prompt)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
        # sampling_settings = [(1, 512), (4, 512), (16, 512), (32, 512), (128, 512), (512, 512),
        #                      (4, 256), (16, 256), (32, 256), (128, 256), (512, 256),
        #                      (4, 128), (16, 128), (32, 128), (128, 128), (512, 128),
        #                      (16, 64), (32, 64), (128, 64), (512, 64)
        #                      ] # A list of (block_length, step)
        # sampling_settings = [(128, 64), (128, 128), (128, 256), (32, 64), (32, 128), (32, 256)]
        block_sizes = [128, 512]
        steps = [64, 128, 256]
        for block_length in block_sizes:
            block_length = min(block_length, args.gen_length)
            for step in steps:
                # for block_size, step in sampling_settings:
                if step % (args.gen_length / block_length) != 0:
                    break
                out, intermediates, confidences, intermediate_inputs = diffusion_generate(model, input_ids, mask_id=model.config.mask_token_id, gen_length=args.gen_length, block_length=block_length,
                                         steps=step, temperature=args.temperature, conf_alg=args.conf_alg, rcr=args.rcr, top_p=args.top_p, top_k=args.top_k)
                model_answer = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
                
                
                intermediate_answers = tokenizer.batch_decode(
                    torch.cat(intermediates, dim=0),
                    skip_special_tokens=True)
                answer_rewards = code_reward(intermediate_answers, verification_info=[verification]* len(intermediate_answers))
                answer_correct = (answer_rewards[-1] == 1.0)
                # print(f"Question {problem_index} is {str(answer_correct)}")
                # intermediate_correct = False
                intermediate_correct_cnt = [idx for idx, ans_r in enumerate(answer_rewards) if ans_r == 1.0]
                if (not answer_correct) and len(intermediate_correct_cnt) > 0 and args.log_visualizations:
                    vis_file_name = f"logs/visualizations/htmls/{args.model_path.rstrip('/').split('/')[-1] if args.lora_path is None else args.lora_path.rstrip('/').split('/')[-1]}_prompt_{args.system_prompt_type}_{args.mode}_{step}_{block_length}_{unique_id}_remask_{args.conf_alg}_RCR_{str(args.rcr)}.html"
                    visualize_intermediates(intermediates, intermediate_inputs, intermediate_correct_cnt, vis_file_name)
                all_results.append({"id": unique_id,"problem": problem, "solution": solution, "model_answer": model_answer,
                                    "block_size": block_length, "step": step,
                                    "answer_correct": answer_correct, "intermediate_correct": intermediate_correct_cnt})
    dist.barrier()
    file_name = f"./local_rank_{dist.get_rank()}_{dataset_name.split('/')[-1]}_{args.model_path.rstrip('/').split('/')[-1] if args.lora_path is None else args.lora_path.rstrip('/').split('/')[-1]}_prompt_{args.system_prompt_type}_{args.mode}_{args.gen_length}_remask_{args.conf_alg}_RCR_{str(args.rcr)}.csv"
    pd.DataFrame(all_results).to_csv(os.path.join("./logs", file_name), index=False)
    if dist.get_rank() == 0:
        dfs = []
        all_file_name = file_name = f"./{dataset_name.split('/')[-1]}_{args.model_path.rstrip('/').split('/')[-1] if args.lora_path is None else args.lora_path.rstrip('/').split('/')[-1]}_prompt_{args.system_prompt_type}_{args.mode}_{args.gen_length}_remask_{args.conf_alg}_RCR_{str(args.rcr)}.csv"
        for rank in range(dist.get_world_size()):
            file_name = f"./local_rank_{rank}_{dataset_name.split('/')[-1]}_{args.model_path.rstrip('/').split('/')[-1] if args.lora_path is None else args.lora_path.rstrip('/').split('/')[-1]}_prompt_{args.system_prompt_type}_{args.mode}_{args.gen_length}_remask_{args.conf_alg}_RCR_{str(args.rcr)}.csv"
            dfs.append(pd.read_csv(os.path.join("./logs", file_name)))
            os.remove(os.path.join("./logs", file_name))
        pd.concat(dfs).to_csv(os.path.join("./logs", all_file_name), index=False)
    cleanup_ddp()