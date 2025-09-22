import os.path

import torch
import itertools
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from src.llada.modeling_llada import LLaDAModelLM
from src.dream import DreamModel
from datasets import load_dataset
from math_verify import LatexExtractionConfig, parse, verify
from src.open_r1.utils.trainer_utils import profiling_context, CustomDistributedSampler
import torch.distributed as dist
from src.mdlm_generation_utils import diffusion_generate
import pandas as pd
from latex2sympy2_extended import NormalizationConfig
from tqdm import tqdm
from visualize_diffusion import DiffusionModelVisualizer
from torch.utils.data import DataLoader

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

if __name__ == '__main__':
    local_rank = setup_ddp()
    device = local_rank
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="HuggingFaceH4/MATH-500", choices=["DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/aime_2024", "HuggingFaceH4/MATH-500"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--system_prompt_type", default="normal")
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
    if args.system_prompt_type == "format":
        system_prompt = "Let's first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer> and output the final answer within \\boxed{} inbetween the <answer> </answer> tags"
    elif args.system_prompt_type == "step_by_step":
        system_prompt = "Let's think step by step and output the final answer within \\boxed{}."
    elif args.system_prompt_type == "d1":
        system_prompt = """You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}. Respond in the following format: <reasoning> Your reasoning here </reasoning> <answer> \\boxed{...} </answer>" """
    else:
        system_prompt = "Solve this problem and output the final answer within \\boxed{}."
    # dataset_name = "agentica-org/DeepScaleR-Preview-Dataset" #HuggingFaceH4/MATH-500, HuggingFaceH4/aime_2024, agentica-org/DeepScaleR-Preview-Dataset
    # ds = load_dataset("open-r1/OpenR1-Math-220k", cache_dir="./cache")["train"]
    # ds = load_dataset("HuggingFaceH4/aime_2024", cache_dir="./cache")["train"]
    # ds = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", cache_dir="./cache")["train"]
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
    dataloader = DataLoader(
        ds,
        batch_size=1,
        sampler=CustomDistributedSampler(ds, shuffle=False),
    )
    for p_index, d in tqdm(enumerate(dataloader), total=len(dataloader)):
        # problem_index = random.randint(0, len(ds) - 1)
        # problem_index = 5371
        # problem, answer, solution = ds[problem_index]["problem"], ds[problem_index]["answer"], ds[problem_index]['solution']

        problem, solution = d["problem"][0], d["solution"][0]
        unique_id = d.get("unique_id", [p_index])[0]
        unique_id = unique_id.replace("/", "_").rstrip(".json") if isinstance(unique_id, str) else unique_id
        level = d.get('level', [1])[0]
        p_type = d.get('type', ['math'])[0]
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
        problem += "\n"
        problem += system_prompt
        # Add special tokens for the Instruct model. The Base model does not require the following two lines.
        m = [{"role": "user", "content": problem}, ]
        # inputs = tokenizer.apply_chat_template(
        #     m, return_tensors="pt", return_dict=True, add_generation_prompt=True
        # )
        # input_ids = inputs.input_ids.to(device=device)
        # attention_mask = inputs.attention_mask.to(device=device)
        prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

        input_ids = tokenizer(prompt)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
        # sampling_settings = [(1, 512), (4, 512), (16, 512), (32, 512), (128, 512), (512, 512),
        #                      (4, 256), (16, 256), (32, 256), (128, 256), (512, 256),
        #                      (4, 128), (16, 128), (32, 128), (128, 128), (512, 128),
        #                      (16, 64), (32, 64), (128, 64), (512, 64)
        #                      ] # A list of (block_length, step)
        # sampling_settings = [(128, 64), (128, 128), (128, 256), (32, 64), (32, 128), (32, 256)]
        block_sizes = [512, 128]
        steps = [64, 128, 256]
        for block_length in block_sizes:
            for step in steps:
                # for block_size, step in sampling_settings:
                if step % (args.gen_length / block_length) != 0:
                    break
                out, intermediates, confidences, intermediate_inputs = diffusion_generate(model, input_ids, mask_id=model.config.mask_token_id, gen_length=args.gen_length, block_length=block_length,
                                         steps=step, temperature=args.temperature, conf_alg=args.conf_alg, rcr=args.rcr, top_p=args.top_p, top_k=args.top_k)
                model_answer = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
                if len(gold_parsed) != 0:
                    # We require the answer to be provided in correct latex (no malformed operators)
                    answer_parsed = parse(
                        model_answer,
                        extraction_config=[
                            LatexExtractionConfig(
                                normalization_config=NormalizationConfig(
                                    nits=False,
                                    malformed_operators=False,
                                    basic_latex=True,
                                    equations=True,
                                    boxed="all",
                                    units=True,
                                ),
                                # Ensures that boxed is tried first
                                boxed_match_priority=0,
                                try_extract_without_anchor=False,
                            )
                        ],
                        extraction_mode="first_match",
                    )
                    intermediate_answers = tokenizer.batch_decode(
                        torch.cat(intermediates, dim=0),
                        skip_special_tokens=True)
                    answer_correct = verify(answer_parsed, gold_parsed)
                    # print(f"Question {problem_index} is {str(answer_correct)}")
                    # intermediate_correct = False
                    intermediate_correct_cnt = []
                    for i, intermediate_answer in enumerate(intermediate_answers):
                        intermediate_parsed = parse(
                            intermediate_answer,
                            extraction_config=[
                                LatexExtractionConfig(
                                    normalization_config=NormalizationConfig(
                                        nits=False,
                                        malformed_operators=False,
                                        basic_latex=True,
                                        equations=True,
                                        boxed="all",
                                        units=True,
                                    ),
                                    # Ensures that boxed is tried first
                                    boxed_match_priority=0,
                                    try_extract_without_anchor=False,
                                )
                            ],
                            extraction_mode="first_match",
                        )
                        if verify(gold_parsed, intermediate_parsed):
                            # intermediate_correct = True
                            intermediate_correct_cnt.append(i)
                        # if verify(gold_parsed, intermediate_parsed) and not answer_correct:
                        #     print(f"Correct prediction at timestep {i} for question {problem_index}")
                    if (not answer_correct) and len(intermediate_correct_cnt) > 0 and args.log_visualizations:
                        vis_file_name = f"logs/visualizations/htmls/{args.model_path.rstrip('/').split('/')[-1] if args.lora_path is None else args.lora_path.rstrip('/').split('/')[-1]}_prompt_{args.system_prompt_type}_{args.mode}_{step}_{block_length}_{unique_id}_remask_{args.conf_alg}_RCR_{str(args.rcr)}.html"
                        visualize_intermediates(intermediates, intermediate_inputs, intermediate_correct_cnt, vis_file_name)
                    all_results.append({"id": unique_id,"problem": problem, "solution": solution, "model_answer": model_answer, "level": level,
                                        "p_type": p_type, "block_size": block_length, "step": step,
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