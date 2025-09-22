import contextlib
import functools
import os
import random
import textwrap
import math
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union
from src.mdlm_generation_utils import diffusion_generate
import pandas as pd
import torch.distributed as dist
import time
import shutil
import torch
import torch.utils.data
import transformers
from tqdm import tqdm
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from functools import partial
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed, broadcast
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import Sampler, RandomSampler
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.trainer_utils import (
    TrainOutput,
    speed_metrics,
)
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.trainer_pt_utils import get_model_param_count

from llada.generate import add_gumbel_noise, get_num_transfer_tokens
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled, deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers.utils import is_peft_available, is_accelerate_available, is_sagemaker_mp_enabled, is_torch_xla_available, logging, is_apex_available
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.extras.profiling import profiling_decorator
from trl.import_utils import is_rich_available
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.callbacks import SyncRefModelCallback
from .mdpo_config import MDPOConfig
from trl.trainer.utils import (
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)
from transformers.trainer_callback import (
    ExportableState,
    TrainerCallback,
    TrainerState,
)
from itertools import groupby
from math_verify.parser import get_extraction_regexes, extract_match
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from src.open_r1.utils.trainer_utils import profiling_context, CustomDistributedSampler
import importlib.metadata
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
if is_peft_available():
    from peft import PeftConfig, get_peft_model, PeftModel
if is_apex_available():
    from apex import amp
if is_wandb_available():
    import wandb
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False
logger = logging.get_logger(__name__)
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]
# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCALER_NAME = "scaler.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    from torch_xla import __version__ as XLA_VERSION

    IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(XLA_FSDPV2_MIN_VERSION)
    if IS_XLA_FSDPV2_POST_2_2:
        import torch_xla.distributed.spmd as xs
        import torch_xla.runtime as xr
else:
    IS_XLA_FSDPV2_POST_2_2 = False
if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.state import AcceleratorState
    from accelerate.utils import (
        AutocastKwargs,
        DistributedDataParallelKwargs,
        DistributedType,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper

def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,) if is_peft_available() else ()
        # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
        if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
            from peft import PeftMixedModel

            classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False

def extract_target_from_pred(
    pred: str,
    target_res,
    timeout_seconds: int,
    fallback_mode = "no_fallback",
    extraction_mode = "any_match",
):
    """Extracts targets from a prediction string using regex patterns.
    Returns first sucesffuly extracted match.

    Args:
        pred (str): The prediction string to extract from
        target_res (list[tuple[list[tuple[re.Pattern[str], int]], ExtractionTarget]]): List of regex patterns and their priorities for each target type
        fallback_mode (Literal["no_fallback", "first_match"], optional): How to handle extraction failures. Defaults to "no_fallback".
            - "no_fallback": Return only successfully parsed match
            - "first_match": Additionaly Include the first string match no matter how parsing finished
        extraction_mode (Literal["first_match", "any_match"], optional): How to handle extraction failures. Defaults to "any_match".
            - "first_match": Only tries to extract the first match
            - "any_match": Tries to extract any match

    Returns:
        list: List of extracted predictions, with first fallbac string appended if fallback_mode is "first_match"
    """
    extracted_predictions = []
    fallbacks = []

    # Get all patterns and sort by priority
    all_patterns = [
        (pattern, target_type, priority)
        for target_patterns, target_type in target_res
        for pattern, priority in target_patterns
    ]

    # Group patterns by priority using itertools.groupby
    match_found = False
    string_matches = []
    sorted_patterns = sorted(all_patterns, key=lambda x: x[2])
    grouped_patterns = list((gr, list(val)) for gr, val in groupby(sorted_patterns, key=lambda x: x[2]))
    for _, patterns_group in grouped_patterns:
        # Find all matches for each pattern in this priority group
        matches_with_pos = (
            (match, match.start(), match.end(), target_type)
            for pattern, target_type, _ in patterns_group
            for match in pattern.finditer(pred)
        )

        # Sort matches by end position (rightmost first) and then by start position (leftmost first)
        matches_with_pos = sorted(
            matches_with_pos, key=lambda x: (x[2], -x[1]), reverse=True
        )

        # Try to extract from each match, starting from rightmost
        for match, _, _, target_type in matches_with_pos:
            extracted_match, str_fallback = extract_match(match, target_type, timeout_seconds=timeout_seconds)

            match_found = True
            if str_fallback:
                fallbacks.append(str_fallback)

            if extracted_match is not None:
                string_matches.append(match)
                extracted_predictions.append(extracted_match)
                break

            if extraction_mode == "first_match":
                break

        # If we extracted something or found something and we're in first_match mode, stop processing other priorities
        if extracted_predictions or (match_found and extraction_mode == "first_match"):
            break

    if fallback_mode == "first_match" and fallbacks:
        extracted_predictions += [fallbacks[0]]

    return extracted_predictions, string_matches

def find_subtensor_mask(tensor, subtensor, method='sliding'):
    """
    Find positions of a subtensor within a larger tensor.

    Args:
        tensor (torch.Tensor): Main tensor to search in
        subtensor (torch.Tensor): Subtensor to search for
        method (str): Method to use for subtensor detection

    Returns:
        torch.Tensor: Binary mask indicating subtensor locations
    """
    if method == 'sliding':
        # Approach 2: Sliding window comparison
        mask = torch.zeros_like(tensor, dtype=torch.float)

        for i in range(len(tensor) - len(subtensor) + 1):
            window = tensor[i:i + len(subtensor)]
            if torch.equal(window, subtensor):
                mask[i:i + len(subtensor)] = 1

        return mask

    elif method == 'unique':
        # Approach 3: Using unique value matching strategy
        mask = torch.zeros_like(tensor, dtype=torch.float)

        for i in range(len(tensor) - len(subtensor) + 1):
            if torch.all(tensor[i:i + len(subtensor)] == subtensor):
                mask[i:i + len(subtensor)] = 1

        return mask

class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, batch_size=3, repeat_count=4)
    >>> list(sampler)
    [4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,

     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6]
    ```

    ```txt
    mini_repeat_count = 3
          -   -   -
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                repeat_count = 2
          0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
          ---------   ---------   ---------   ---------
           ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                         batch_size = 12
    ```
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()  # Create a local random generator
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


class MDPOTrainer(Trainer):
    """
    Trainer for the Masked Diffusion Policy Optimization (MDPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    def reward_func(completions, **kwargs):
        # Dummy reward function that rewards completions with more unique letters.
        return [float(len(set(completion))) for completion in completions]

    trainer = MDPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_func,
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`MDPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    _tag_names = ["trl", "mdpo"]

    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            reward_funcs: Union[RewardFunc, list[RewardFunc]],
            args: Optional[MDPOConfig] = None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
            processing_class: Optional[PreTrainedTokenizerBase] = None,
            reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
            callbacks: Optional[list[TrainerCallback]] = None,
            optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
            None, None),
            peft_config: Optional["PeftConfig"] = None,
            incremental_training: bool = False,
            mixture_data: bool = False,
            ab_path: str = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = MDPOConfig(f"{model_name}-MDPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `MDPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model = AutoModel.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `MDPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            if not is_peft_available():
                raise ImportError("PEFT is required to use `peft_config`. Run `pip install peft`.")
            model = get_peft_model(model, peft_config)

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            from llada.modeling_llada import LLaDAModelLM
            self.ref_model = LLaDAModelLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype="bfloat16",
                                                          cache_dir="/weka/geiger/gwb130/dlm-r1/cache/")
        # elif is_peft_model(model):
        #     # If PEFT is used, the reference model is not needed since the adapter can be disabled
        #     # to revert to the initial model.
        #     self.ref_model = None
        else:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.temperature = args.temperature
        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon = args.epsilon
        # Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle.
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = [None] * args.gradient_accumulation_steps

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self.log_completions = args.log_completions

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        self.incremental_training = incremental_training
        self.ab_path = ab_path
        self.mixture_data = mixture_data
        self.current_data = []
        if incremental_training:
            self.incremental_train_dataloader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, sampler=CustomDistributedSampler(train_dataset, shuffle=True))
        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(1, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(1, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)
        self.generation_fn = partial(diffusion_generate, gen_length=self.max_completion_length, temperature=self.temperature,
                                     conf_alg=self.args.conf_alg, mask_id=processing_class.mask_token_id, rcr=self.args.rcr, top_k=self.args.top_k, top_p=self.args.top_p)

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False
        self.eval_dataloader = DataLoader(self.eval_dataset, batch_size=1, sampler=CustomDistributedSampler(self.eval_dataset, shuffle=False),)
        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _get_train_sampler(self) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                     |     GPU 0     |     GPU 1     |     GPU 2    |
        #
        #               global_step   step     <───────>  num_generations=3
        #                                      <───────────> per_device_train_batch_size=4
        #                ▲   0          0      0   0   0   1   1   1   2   2   2   3   3   3  │
        #  grad_accum=3  │   0          1      4   4   4   5   5   5   6   6   6   7   7   7  │ Generate completions for each prompt
        #                ▼   0          2      8   8   8   9   9   9  10  10  10  11  11  11  │
        #
        #                    1          3      0   0   0   1   1   1   2   2   2   3   3   3  │ The sampled prompts are the same as in the first iteration
        #                    1          4      4   4   4   5   5   5   6   6   6   7   7   7  │ Reuse the completions (here, once, because num_iterations=2)
        #                    1          5      8   8   8   9   9   9  10  10  10  11  11  11  │
        #
        #                    2          6     12  12  12  13  13  13  14  14  14  15  15  15
        #                    2          7     16  16  16  17  17  17  18  18  18  19  19  19
        #                    2          8     20  20  20  21  21  21  22  22  22  23  23  23
        #                                          ...
        effective_batch_size = (
                self.args.per_device_train_batch_size
                * self.accelerator.num_processes
                * self.args.gradient_accumulation_steps
        )
        return RepeatRandomSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatRandomSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: MDPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
                "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model

    # Get the per-token log probabilities for the completions for the model and the reference model
    @profiling_decorator
    def _get_per_token_logps(self, model, input_ids, target_ids, attention_mask):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        target_ids = target_ids[:, -self.max_completion_length:]
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # See https://github.com/huggingface/trl/issues/2770
        logits = logits[:, -self.max_completion_length:]
        logits = logits / (self.temperature + torch.finfo(logits.dtype).eps)
        return selective_log_softmax(logits, target_ids)  # compute logprobs for the input tokens

    @profiling_decorator
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            if self.state.global_step % self.num_iterations == 0:
                # inputs = self._generate_and_score_completions(inputs)
                self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
            else:
                inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            self._step += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            inputs = self._generate_and_score_completions(inputs)
        return inputs

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        #TODO: support mini-batch
        all_diffusion_steps = [x.get("diffusion_steps", self.args.diffusion_steps) for x in inputs][0]
        all_block_lengths = [x.get("block_length", self.args.block_length) for x in inputs][0]
        if inputs[0].get("diffusion_steps", None) is None:
            if self.mixture_data:
                all_diffusion_steps, all_block_lengths = random.choice([(self.args.diffusion_steps, 128), (self.args.diffusion_steps, 512)])
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        solutions_text = [i["solution"] for i in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text, return_tensors="pt", padding="max_length", padding_side="left", truncation=True, max_length=self.max_prompt_length,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        # if torch.distributed.get_rank() == 0:
        #     diffusion_steps = random.choice([64, 128, 256])
        #     block_length = random.choice([64, 128, 256, 512])
        #     generation_params = torch.tensor([diffusion_steps, block_length], device=device, dtype=torch.int32)
        # else:
        #     generation_params = torch.empty((2,), device=device, dtype=torch.int32)
        # torch.distributed.broadcast(generation_params, src=0)
        # diffusion_steps, block_length = generation_params.cpu().numpy().tolist()
        # diffusion_steps, block_length = self.args.diffusion_steps, self.args.block_length
        diffusion_steps, block_length = all_diffusion_steps, all_block_lengths
        with unwrap_model_for_generation(self.model_wrapped, self.accelerator, gather_deepspeed3_params=False) as unwrapped_model:
            _, all_steps_output_completion_ids, all_confidence, all_steps_input_ids = self.generation_fn(unwrapped_model, prompt_ids, prompt_mask=prompt_mask, steps=diffusion_steps,
                                     block_length=block_length)
        # Compute prompt length and extract completion ids
        completion_ids = all_steps_output_completion_ids[-1]
        completion_ids = completion_ids.to(device)
        prompt_length = prompt_ids.size(1)
        # prompt_ids = prompt_completion_ids[:, :prompt_length]
        # completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        all_rewards = []
        # last_block_steps = self.args.diffusion_steps // (
        #         self.args.max_completion_length // self.args.block_length)
        for t_completion_ids in all_steps_output_completion_ids:
            # replace out vocab token with eos here becauseof this issue https://github.com/DreamLM/Dream/issues/54
            t_completion_ids = torch.where(t_completion_ids >= len(self.processing_class), self.processing_class.eos_token_id, t_completion_ids)
            t_completions_texts = self.processing_class.batch_decode(t_completion_ids, skip_special_tokens=True)
            t_completions = []
            if is_conversational(inputs[0]):

                for prompt, t_completions_text in zip(prompts, t_completions_texts):
                    bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                    t_completions.append([{"role": "assistant", "content": bootstrap + t_completions_text}])
            else:
                # To align with chat template so reward functions can parse
                for prompt, t_completions_text in zip(prompts, t_completions_texts):
                    t_completions.append([{"content": prompt + t_completions_text}])
            rewards_t = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
            for i, (reward_func, reward_processing_class) in enumerate(
                    zip(self.reward_funcs, self.reward_processing_classes)
            ):
                if isinstance(reward_func,
                              nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                    reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
                else:
                    reward_func_name = reward_func.__name__
                with profiling_context(self, reward_func_name):
                    if isinstance(
                            reward_func, nn.Module
                    ):  # Module instead of PretrainedModel for compat with compiled models
                        raise NotImplementedError
                    else:
                        # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                        keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                        output_reward_func_t = reward_func(prompts=prompts, completions=t_completions,
                                                           **reward_kwargs)
                        rewards_t[:, i] = torch.tensor(output_reward_func_t, dtype=torch.float32, device=device)
            all_rewards.append((rewards_t * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1, keepdim=True))

        all_step_rewards = torch.cat(all_rewards, dim=-1)
        # adv-v1: single step reward
        # all_step_advantages = all_step_rewards
        # adv-v2: average reward following action t
        # all_step_advantages = (all_step_rewards.flip(dims=(-1,)).cumsum(dim=-1) / torch.arange(1, diffusion_steps+1, device=all_step_rewards.device).unsqueeze(0)).flip(dims=(-1,))
        #adv-v3: reward difference
        all_step_advantages = torch.cat([all_step_rewards[:, 0:1], all_step_rewards[:, 1:] - all_step_rewards[:, :-1]], dim=-1) + 1
        #adv-v4: adv-v3 + adv-v2
        all_step_advantages += torch.cat([(all_step_rewards[:, 1:].flip(dims=(-1,)).cumsum(dim=-1) / torch.arange(1, diffusion_steps, device=all_step_rewards.device).unsqueeze(0)).flip(dims=(-1,)), all_step_rewards[:, -1:]], dim=-1)
        gathered_all_advantages = gather(all_step_advantages).detach().cpu()
        gathered_all_advantages = gathered_all_advantages.view(-1, self.num_generations, diffusion_steps)
        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"
        mean_grouped_advantages = gathered_all_advantages.mean(dim=1)
        std_grouped_advantages = gathered_all_advantages.std(dim=1)
        advantages = (all_step_advantages.detach().cpu() - mean_grouped_advantages) / (std_grouped_advantages + 1e-4)
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)

        rewards_per_func = gather(rewards_t)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            # Only calculate mean for samples where this reward function was applied (non-NaN values)
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
            # self._metrics[mode][f"rewards/{reward_func_name}"].append(rewards_per_func[i].item())

        self._metrics[mode]["reward"].append(rewards.mean().item())
        # self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item()
        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts_text)
            # replace out vocab token with eos here becauseof this issue https://github.com/DreamLM/Dream/issues/54
            completion_ids = torch.where(completion_ids >= len(self.processing_class),
                                           self.processing_class.eos_token_id, completion_ids)
            completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
            completions_to_log = gather_object(completions_text)
            solutions_to_log = gather_object(solutions_text)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        self.state.global_step,
                    )
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    import pandas as pd

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "solution": solutions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "all_steps_input_ids": all_steps_input_ids,
            "all_steps_completion_ids": all_steps_output_completion_ids,
            "all_confidence": all_confidence,
            "advantages": advantages,
        }

    def split_into_micro_batches(self, traj):
        prompt_ids, prompt_mask = traj["prompt_ids"], traj["prompt_mask"]
        advantages = traj["advantages"]
        # Split data into manageable micro-batches
        # After testing sort torch.abs() is worse, so we directly pick the highest steps in order
        for i, step in enumerate(torch.topk(torch.abs(advantages).sum(dim=0), k=self.args.sample_train_steps, dim=-1).indices):
            # sample_index = sample_indices[:, i]
            step = step.item()
            input_answer_ids = traj["all_steps_input_ids"][step]
            completion_ids = traj["all_steps_completion_ids"][step]

            # ans_mask = torch.stack([traj["all_correct_token_masks"][idx][b] for b, idx in enumerate(sample_index)], dim=0).to(prompt_ids.dtype).to(prompt_ids.device)
            input_ids = torch.concatenate(
                [prompt_ids, input_answer_ids.to(prompt_ids.dtype).to(prompt_ids.device)], dim=-1)
            target_ids = torch.concatenate(
                [prompt_ids, completion_ids.to(prompt_ids.dtype).to(prompt_ids.device)], dim=-1)
            input_mask = torch.concatenate(
                [prompt_mask, torch.ones_like(input_answer_ids).to(prompt_mask.dtype).to(prompt_mask.device)], dim=-1)
            conf = traj["all_confidence"][step].to(input_ids.device)
            with torch.no_grad():
                if self.beta != 0.0:
                    ref_per_token_logps = self._get_per_token_logps(self.ref_model, input_ids, target_ids, input_mask)
                else:
                    ref_per_token_logps = None
            yield {
                "input_ids": input_ids,
                "input_mask": input_mask,
                "target_ids": target_ids,
                "advantages": (advantages[:, step: step+1]).expand_as(input_answer_ids).to(input_ids.device),
                "conf": conf,
                "ref_per_token_logps": ref_per_token_logps,
                "step": torch.ones_like(input_ids) * step
            }

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the intial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size
        (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            epoch_based,
            len_dataloader,
            max_steps,
        ) = self.set_initial_training_values(args, train_dataloader, total_train_batch_size)

        num_train_tokens = None
        if self.args.include_tokens_per_second:
            num_train_tokens = self.num_tokens(train_dataloader, None if epoch_based else max_steps)
            # If going by epochs, multiply tokens linearly
            if len_dataloader is not None and epoch_based:
                num_train_tokens *= args.num_train_epochs
            # Otherwise since its steps, we just multiply by grad accum
            else:
                num_train_tokens *= args.gradient_accumulation_steps

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        self.state.compute_steps(args, max_steps)

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if use_accelerator_prepare and self.is_fsdp_enabled:
            # In case of auto_find_batch_size=True
            # Remove FSDP wrapping from sub-models.
            self.model = unwrap_model(self.model, recursive=True)

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                # configure fsdp plugin for qlora if any
                self._fsdp_qlora_plugin_updates()
                if self.accelerator.mixed_precision != "fp8":
                    self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # In this case we are in DDP + LOMO, which should be supported
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
        self._load_scaler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.state.init_training_references(self, train_dataloader, max_steps, num_train_epochs, trial)

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        for epoch in range(epochs_trained, num_train_epochs):
            if self.incremental_training:
                if epoch == 0 and self.ab_path is not None and os.path.exists(self.ab_path):
                    ab_data = pd.read_csv(self.ab_path)
                    self.current_data = []
                    for row in range(ab_data.shape[0]):
                        self.current_data.append({"prompt": [{"role": "user", "content": ab_data.loc[row, "Problem"]}],
                                                  "solution": ab_data.loc[row, "Solution"],
                                                  "diffusion_steps": ab_data.loc[row, "diffusion_steps"] if "diffusion_steps" in ab_data.columns else None,
                                                  "block_length": ab_data.loc[
                                                      row, "block_length"] if "block_length" in ab_data.columns else None,
                                                  })
                else:
                    data_per_rank, wrong_per_rank = self._incremental_data_selection()
                    dist.barrier()
                    data_gathered = gather_object(data_per_rank)
                    wrong_gathered = gather_object(wrong_per_rank)
                    # for old_data in self.current_data:
                    #     for wrong_data in wrong_gathered:
                    #         if old_data["prompt"][0]["content"] == wrong_data["prompt"][0]["content"] and old_data["diffusion_steps"] == wrong_data["diffusion_steps"] and old_data["block_length"] == wrong_data["block_length"]:
                    #             data_gathered.append(wrong_data)
                    self.current_data = data_gathered
                    if dist.get_rank() == 0:
                        pd.DataFrame({"Problem": [a["prompt"][0]["content"] for a in data_gathered],
                                      "Solution": [a["solution"] for a in data_gathered]}).to_csv(os.path.join(args.output_dir, f"data_epoch_{epoch+1}.csv"))
                self.train_dataset = Dataset.from_list(self.current_data)
                train_dataloader = self.get_train_dataloader()
                if self.is_fsdp_xla_v2_enabled:
                    train_dataloader = tpu_spmd_dataloader(train_dataloader)

                # Setting up training control variables:
                # number of training epochs: num_train_epochs
                # number of training steps per epoch: num_update_steps_per_epoch
                # total number of training steps to execute: max_steps
                (
                    num_train_epochs,
                    num_update_steps_per_epoch,
                    num_examples,
                    num_train_samples,
                    epoch_based,
                    len_dataloader,
                    max_steps,
                ) = self.set_initial_training_values(args, train_dataloader, total_train_batch_size)
                self.state.compute_steps(args, max_steps)
                self.state.init_training_references(self, train_dataloader, max_steps, num_train_epochs, trial)
                if dist.get_rank() == 0:
                    logger.info(f"***** Epoch {epoch+1} *****")
                    logger.info(f"  Num examples = {num_examples:,}")
                    logger.info(f"  Total optimization steps = {max_steps:,}")
            epoch_dataloader = train_dataloader
            if hasattr(epoch_dataloader, "set_epoch"):
                epoch_dataloader.set_epoch(epoch)
            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_dataloader)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_dataloader = skip_first_batches(epoch_dataloader, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            epoch_iterator = iter(epoch_dataloader)
            # We chunkify the epoch iterator into gradient accumulation steps `n` batches
            remainder = num_examples % args.gradient_accumulation_steps
            if remainder == 0:
                remainder = args.gradient_accumulation_steps
            update_step = -1
            total_updates = steps_in_epoch // args.gradient_accumulation_steps + 1
            if args.gradient_accumulation_steps == 1:
                total_updates -= 1
            for _ in range(total_updates):
                update_step += 1
                # Evaluation every 100 step
                if update_step % 100 == 0:
                    # pass
                    correct_per_rank = self.evaluate()
                    dist.barrier()
                    cpu_pg = dist.new_group(backend="gloo")  # 1-line fix
                    world_size = dist.get_world_size()
                    gathered_corrects = [None] * world_size if dist.get_rank() == 0 else None
                    dist.gather_object(correct_per_rank, gathered_corrects, dst=0, group=cpu_pg)
                    if dist.get_rank() == 0:
                        wandb.log({"eval/accuracy": sum([len(corrects) for corrects in gathered_corrects]) / len(self.eval_dataset)})
                    dist.destroy_process_group(cpu_pg)
                num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
                batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches)
                for i, batch in enumerate(batch_samples):
                    step += 1
                    do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch
                    # Since we perform prefetching, we need to manually set sync_gradients
                    self.accelerator.gradient_state._set_sync_gradients(do_sync_step)
                    diffusion_traj = self._generate_and_score_completions(batch)
                    for inputs in self.split_into_micro_batches(diffusion_traj):
                        if inputs is None:
                            continue
                        if self.args.include_num_input_tokens_seen:
                            main_input_name = getattr(self.model, "main_input_name", "input_ids")
                            if main_input_name not in inputs:
                                logger.warning(
                                    "Tried to track the number of tokens seen, however the current model is "
                                    "not configured properly to know what item is the input. To fix this, add "
                                    "a `main_input_name` attribute to the model class you are using."
                                )
                            else:
                                input_tokens = inputs[main_input_name].numel()
                                input_tokens = torch.tensor(input_tokens, device=self.args.device, dtype=torch.int64)
                                self.state.num_input_tokens_seen += (
                                    self.accelerator.gather(input_tokens).sum().cpu().item()
                                )
                        if rng_to_sync:
                            self._load_rng_state(resume_from_checkpoint)
                            rng_to_sync = False

                        # Skip past any already trained steps if resuming training
                        if steps_trained_in_current_epoch > 0:
                            steps_trained_in_current_epoch -= 1
                            if steps_trained_progress_bar is not None:
                                steps_trained_progress_bar.update(1)
                            if steps_trained_in_current_epoch == 0:
                                self._load_rng_state(resume_from_checkpoint)
                            continue
                        elif steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.close()
                            steps_trained_progress_bar = None

                        # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
                        context = (
                            functools.partial(self.accelerator.no_sync, model=model)
                            if i != len(batch_samples) - 1
                            and self.accelerator.distributed_type != DistributedType.DEEPSPEED
                            else contextlib.nullcontext
                        )
                        with context():
                            tr_loss_step = self.training_step(model, inputs, num_items_in_batch)

                        if (
                            args.logging_nan_inf_filter
                            and not is_torch_xla_available()
                            and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                        ):
                            # if loss is nan or inf simply add the average of previous logged losses
                            tr_loss = tr_loss + tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                        else:
                            if tr_loss.device != tr_loss_step.device:
                                raise ValueError(
                                    f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                                )
                            tr_loss = tr_loss + tr_loss_step

                        self.current_flos += float(self.floating_point_ops(inputs))
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                        # PyTorch/XLA relies on the data loader to insert the mark_step for
                        # each step. Since we are breaking the loop early, we need to manually
                        # insert the mark_step here.
                        if self.control.should_epoch_stop or self.control.should_training_stop:
                            if is_torch_xla_available():
                                xm.mark_step()
                            break
                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
                    if do_sync_step:
                        # Since we perform prefetching, we need to manually set sync_gradients to True
                        self.accelerator.gradient_state._set_sync_gradients(True)

                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            if is_sagemaker_mp_enabled() and args.fp16:
                                _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                            elif self.use_apex:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                _grad_norm = nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer),
                                    args.max_grad_norm,
                                )
                            else:
                                _grad_norm = self.accelerator.clip_grad_norm_(
                                    model.parameters(),
                                    args.max_grad_norm,
                                )

                            if (
                                    is_accelerate_available()
                                    and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                            ):
                                grad_norm = model.get_global_grad_norm()
                                # In some cases the grad norm may not return a float
                                if hasattr(grad_norm, "item"):
                                    grad_norm = grad_norm.item()
                            else:
                                grad_norm = _grad_norm

                        self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

                        self.optimizer.step()

                        self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                        if not self.accelerator.optimizer_step_was_skipped:
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                self.lr_scheduler.step()

                        model.zero_grad()
                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                        self._maybe_log_save_evaluate(
                            tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time
                        )
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)
                # We also need to break out of the nested loop
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            if epoch == (num_train_epochs - 1):
                # pass
                correct_per_rank = self.evaluate()
                dist.barrier()
                cpu_pg = dist.new_group(backend="gloo")  # 1-line fix
                world_size = dist.get_world_size()
                gathered_corrects = [None] * world_size if dist.get_rank() == 0 else None
                dist.gather_object(correct_per_rank, gathered_corrects, dst=0, group=cpu_pg)
                if dist.get_rank() == 0:
                    wandb.log({"eval/accuracy": sum([len(corrects) for corrects in gathered_corrects]) / len(
                        self.eval_dataset)})
                dist.destroy_process_group(cpu_pg)
            if step < 0:
                logger.warning(
                    "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GDPOTrainer does not support returning outputs")
        # refer to the guideline here https://github.com/ML-GSAI/LLaDA/blob/main/GUIDELINES.md
        per_token_logps = self._get_per_token_logps(model, inputs["input_ids"], inputs["target_ids"], inputs["input_mask"])
        confidence = inputs["conf"]
        # # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            # Refer to http://joschu.net/blog/kl-approx.html
            logr = ref_per_token_logps - per_token_logps
            # k1 = -logr
            # k2 = logr ** 2 / 2
            # k3 = (logr.exp() - 1) - logr
            # k4 = torch.where(logr < 0, k3, torch.min(k1, k3))
            # r = torch.exp(logr)
            # clip_r = torch.clamp(r, max=6)
            # k3_clip = clip_r - 1 - logr
            # k3_clip_pos = torch.where(k3_clip < 0, 0, k3_clip)
            # k5 = (logr) ** 2 / 2.
            per_token_kl = logr ** 2 / 2

        # Compute the loss
        completion_mask = inputs["input_ids"] == self.processing_class.mask_token_id
        completion_mask = completion_mask[:, -self.max_completion_length:]
        lambda_t = self.max_completion_length / (completion_mask.sum(dim=-1, keepdim=True) + 1e-4)

        # # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # # _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        per_token_loss1 = coef_1 * inputs["advantages"]
        per_token_loss2 = coef_2 * inputs["advantages"]
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2) * lambda_t
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        if confidence is None:
            confidence = torch.ones_like(per_token_loss)
        # loss = (per_token_loss * completion_mask).sum() / (completion_mask.sum() + 1e-4)
        if model.module.config.model_type == "Dream":
            per_token_loss = torch.cat([per_token_loss[:, 1:], per_token_loss[:, -1:]], dim=1)
            completion_mask = torch.cat([completion_mask[:, 1:], completion_mask[:, :1]], dim=1)
            completion_mask[:, -1] = 0
            confidence = torch.cat([confidence[:, 1:], confidence[:, :1]], dim=1)
        loss = (per_token_loss * completion_mask * confidence).sum() / (completion_mask.sum() + 1e-4)
        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        # is_clipped = (per_token_loss1 < per_token_loss2).float()
        # clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        # self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(loss)
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def _incremental_data_selection(self):
        all_selected_samples = []
        all_wrong_samples = []
        for batch in tqdm(self.incremental_train_dataloader, disable=(dist.get_rank() != 0), desc="Test on the dataset to find ICFW samples"):
            if self.mixture_data:
                diffusion_steps, block_length = random.choice([(self.args.diffusion_steps, 32), (self.args.diffusion_steps, 512)])
            else:
                diffusion_steps, block_length = self.args.diffusion_steps, self.args.block_length
            prompts_texts = [self.processing_class.apply_chat_template([{"role": "user", "content": problem}, ], add_generation_prompt=True, tokenize=False) for problem in batch["problem"]]
            for prompts_text, problem, solution in zip(prompts_texts, batch["problem"], batch["solution"]):
                prompt_inputs = self.processing_class(
                    text=prompts_text, return_tensors="pt"
                )
                prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
                prompt_ids = prompt_ids.to(self.accelerator.device)
                prompt_mask = prompt_mask.to(self.accelerator.device)
                with (unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model):
                    _, all_steps_output_completion_ids, _, all_steps_input_ids = diffusion_generate(unwrapped_model,
                                                                                       prompt_ids,
                                                                                       prompt_mask=prompt_mask,
                                                                                       steps=diffusion_steps,
                                                                                       gen_length=self.max_completion_length,
                                                                                       block_length=block_length,
                                                                                       temperature=0.1 if unwrapped_model.config.model_type == "Dream" else 0.0,
                                                                                       mask_id=self.processing_class.mask_token_id,
                                                                                       conf_alg=self.args.conf_alg,
                                                                                       rcr=self.args.rcr,
                                                                                       top_k=self.args.top_k, top_p=self.args.top_p)
                # Compute prompt length and extract completion ids
                cat_all_steps_output_completion_ids = torch.cat(all_steps_output_completion_ids, dim=0)
                # replace out vocab token with eos here becauseof this issue https://github.com/DreamLM/Dream/issues/54
                cat_all_steps_output_completion_ids = torch.where(cat_all_steps_output_completion_ids >= len(self.processing_class),
                                             self.processing_class.eos_token_id, cat_all_steps_output_completion_ids)
                intermediate_answers = self.processing_class.batch_decode(cat_all_steps_output_completion_ids,
                                           skip_special_tokens=True)
                # print(f"Question {problem_index} is {str(answer_correct)}")
                # intermediate_correct = False
                intermediate_correct_cnt = []
                for i, intermediate_answer in enumerate(intermediate_answers):
                    intermediate_correct = self.incremental_train_dataloader.dataset.verify_fn(intermediate_answer, ([nums.item() for nums in solution["nums"]], solution["target"].item()) if isinstance(solution, dict) else solution, question=batch.get("question", [None])[0])
                    if intermediate_correct:
                        intermediate_correct_cnt.append(i)
                answer_correct = len(intermediate_correct_cnt) > 0 and intermediate_correct_cnt[-1] == (self.args.diffusion_steps - 1)
                if len(intermediate_correct_cnt) > 0 and not answer_correct:
                    all_selected_samples.append({"prompt":[{"role": "user", "content": problem}, ], "solution": [{"nums": [i.item() for i in solution["nums"]], "target": solution["target"].item()}] if isinstance(solution, dict) else solution, "rank": dist.get_rank(), "diffusion_steps": diffusion_steps, "block_length": block_length})
                if len(intermediate_correct_cnt) == 0:
                    all_wrong_samples.append({"prompt":[{"role": "user", "content": problem}, ], "solution": [{"nums": [i.item() for i in solution["nums"]], "target": solution["target"].item()}] if isinstance(solution, dict) else solution, "rank": dist.get_rank(), "diffusion_steps": diffusion_steps, "block_length": block_length})
        return all_selected_samples, all_wrong_samples

    def evaluate(self):
        correct_per_rank = []
        for batch in tqdm(self.eval_dataloader, desc="Evaluation"):
            solution = batch["solution"][0]
            if isinstance(solution, dict):
                solution = ([i.item() for i in solution["nums"]], solution["target"].item())
            m = [{"role": "user", "content": batch["problem"][0]}, ]
            prompt = self.processing_class.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            prompt_inputs = self.processing_class(
                text=prompt, return_tensors="pt"
            )
            prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
            with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
                completion_ids, _, _, _ = diffusion_generate(unwrapped_model,
                 prompt_ids.to(unwrapped_model.device), prompt_mask=prompt_mask.to(unwrapped_model.device),
                 steps=self.args.diffusion_steps, gen_length=self.max_completion_length,
                 block_length=self.args.block_length, temperature=0.1 if unwrapped_model.config.model_type == "Dream" else 0.0,
                 conf_alg=self.args.conf_alg, mask_id=self.processing_class.mask_token_id, rcr=self.args.rcr,
                 top_k=self.args.top_k, top_p=self.args.top_p)
            # replace out vocab token with eos here becauseof this issue https://github.com/DreamLM/Dream/issues/54
            completion_ids = torch.where(completion_ids >= len(self.processing_class),
                                         self.processing_class.eos_token_id, completion_ids)
            # Compute prompt length and extract completion ids
            model_answer = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)[0]
            answer_correct = self.eval_dataloader.dataset.verify_fn(model_answer, solution, question=batch.get("question", [None])[0])
            if answer_correct:
                correct_per_rank.append(batch.get("unique_id", [1])[0])
        return correct_per_rank
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "eval" if self.control.should_evaluate else "train"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics[mode].clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GDPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))