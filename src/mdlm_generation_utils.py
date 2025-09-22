import torch
import numpy as np
import torch.nn.functional as F
import torch.distributions as dists
from src.llada.generate import add_gumbel_noise, get_num_transfer_tokens_maskgit

def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits

@torch.no_grad()
def diffusion_generate(model, prompt, mask_id, prompt_mask=None, steps=64, gen_length=128, block_length=32, temperature=0.,
                 conf_alg='random', mode="linear", rcr=False, top_p=None, top_k=None):
    '''
    Optimized version of the generate function.
    '''
    if prompt_mask == None:
        prompt_mask = torch.ones_like(prompt) 
    # Use mixed precision for faster computation
    with torch.amp.autocast("cuda", enabled=True):
        x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=prompt.device)
        x[:, :prompt.shape[1]] = prompt.clone()
        attn_mask = torch.ones_like(x)
        attn_mask[:, :prompt_mask.shape[1]] = prompt_mask.clone()
        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        intermediate_inputs = []
        intermediate_results = []
        intermediate_confidence = []
        # Adjust steps if needed
        steps_per_block = max(1, steps // num_blocks)
        overtime_confidence = torch.zeros_like(x, dtype=torch.float32)
        for num_block in range(num_blocks):
            start_idx = prompt.shape[1] + num_block * block_length
            end_idx = prompt.shape[1] + (num_block + 1) * block_length

            block_mask_index = (x[:, start_idx:end_idx] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens_maskgit(block_mask_index, steps_per_block, mode=mode)

            for i in range(steps_per_block):
                mask_index = (x == mask_id)
                intermediate_inputs.append(x.clone().cpu()[:, -gen_length:])
                # Handle classifier-free guidance more efficiently
                logits = model(x, attention_mask=attn_mask).logits
                # Apply Gumbel noise for sampling, commented for now as Dream 7B crashes with gumbel noise
                # logits = logits(logits, temperature)
                if temperature > 0:
                    logits = logits / temperature
                if top_p is not None and top_p < 1:
                    logits = top_p_logits(logits, top_p)
                if top_k is not None:
                    logits = top_k_logits(logits, top_k)
                # x0 = torch.argmax(logits_with_noise, dim=-1)
                # Handle remasking strategy
                if conf_alg == 'random':
                    p = torch.rand(x0.shape, device=x0.device)
                else:
                    # Use float32 instead of float64 for better performance
                    p = F.softmax(logits, dim=-1)
                if temperature > 0:
                    try:
                        x0 = dists.Categorical(probs=p).sample()
                        confidence = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
                    except:
                        confidence, x0 = p.max(dim=-1)
                else:
                    confidence, x0 = p.max(dim=-1)
                if not rcr:
                    intermediate_confidence.append(confidence.clone().cpu()[:, -gen_length:])
                if conf_alg == 'entropy':
                    epsilon = 1e-10
                    log_probs = torch.log(p + epsilon)
                    confidence = torch.sum(p * log_probs, dim=-1)
                elif conf_alg == "topk_margin":
                    sorted_probs, _ = torch.sort(p, dim=-1, descending=True)
                    # Extract top1 and top2 probabilities
                    top1_probs = sorted_probs[:, :, 0]
                    top2_probs = sorted_probs[:, :, 1]
                    # Calculate confidence as top1 - top2
                    confidence = top1_probs - top2_probs
                
                # Ensure we don't process tokens beyond the current block
                confidence[:, end_idx:] = -np.inf
                # Update masked tokens
                x0 = torch.where(mask_index, x0, x)
                intermediate_results.append(x0.clone().cpu()[:, -gen_length:])
                # valid_token_mask = x0 != 198
                # confidence = torch.where(torch.logical_and(mask_index, valid_token_mask), x0_p, torch.tensor(-np.inf, device=x0.device))
                confidence = torch.where(mask_index, confidence, torch.tensor(-np.inf, device=x0.device))

                # Select tokens to transfer based on confidence
                for j in range(confidence.shape[0]):
                    num_tokens = num_transfer_tokens[j, i].item()
                    if rcr:
                        _, select_indices = torch.topk(confidence[j], k=num_transfer_tokens[j, i:].sum().item())
                        x[j, select_indices] = x0[j, select_indices]
                        overtime_confidence[j, select_indices] = confidence[j, select_indices].clone()
                        # if (x[j,:] == mask_id).sum() <= 0:
                        if i != (steps_per_block - 1):
                            overtime_conf_wo_zeros = \
                                torch.where(overtime_confidence == 0.0, 1.0, overtime_confidence)[j]
                            num_tokens_to_mask = num_transfer_tokens[j, i + 1:].sum().item()
                            _, mask_select_indices = torch.topk(overtime_conf_wo_zeros, k=num_tokens_to_mask,
                                                                largest=False)
                            if len(mask_select_indices) == 0:
                                break
                            x[j, mask_select_indices] = mask_id
                    else:
                        if num_tokens > 0:
                            _, select_indices = torch.topk(confidence[j], k=num_tokens)
                            x[j, select_indices] = x0[j, select_indices]
                if rcr:
                    intermediate_confidence.append(overtime_confidence.clone().cpu()[:, -gen_length:])
        return x[:, -gen_length:], intermediate_results, intermediate_confidence, intermediate_inputs