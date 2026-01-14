import os
import argparse
import warnings
import torch

from .model import GPT
from .data import build_datasets
from .tokenizer import load_tokenizer

# suppress specific noisy warnings from torch during quantized ckpt load
warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

def load_checkpoint(path):
    return torch.load(path, map_location="cpu")

PUNCT = set(list(",，。．、：:；;！!？?…"))

def _is_punct_token(tok, tid: int) -> bool:
    try:
        s = tok.decode([tid])
        return len(s) > 0 and s[0] in PUNCT
    except Exception:
        return False

def _trim_leading_punct(s: str) -> str:
    i = 0
    while i < len(s) and (s[i].isspace() or s[i] in PUNCT):
        i += 1
    return s[i:]

def generate(model, tok, prompt, max_new_tokens=64, temperature=1.0, top_k=0, top_p=1.0, repetition_penalty: float = 1.0, stop_strings=None, min_tokens: int = 5, device=None):
    model.eval()
    # normalize prompt: collapse or remove spaces commonly inserted in Chinese
    norm = prompt.replace(" ", "").replace("\u3000", "")
    prefix = tok.encode("用户:" + norm + "\n助手:", add_special_tokens=True)
    x = torch.tensor(prefix, dtype=torch.long, device=device).unsqueeze(0)
    recent = []
    with torch.no_grad():
        for step in range(max_new_tokens):
            logits = model(x)
            logits = logits[:, -1, :] / max(1e-6, temperature) # 根据temperature缩放生成的token的候选列表，[:,-1,:]是取最后一个位置的输出（下一个token的候选列表）
            # 屏蔽特殊token
            if hasattr(tok, 'pad_id') and tok.pad_id is not None and tok.pad_id >= 0:
                logits[0, tok.pad_id] = -float('inf')
            if hasattr(tok, 'bos_id') and tok.bos_id is not None and tok.bos_id >= 0:
                logits[0, tok.bos_id] = -float('inf')
            if hasattr(tok, 'unk_id') and tok.unk_id is not None and tok.unk_id >= 0:
                logits[0, tok.unk_id] = -float('inf')
            if step == 0 and hasattr(tok, 'eos_id') and tok.eos_id is not None and tok.eos_id >= 0:
                logits[0, tok.eos_id] = -float('inf') # 第一个token不能是eos (结束token)
            if repetition_penalty > 1.0 and len(recent) > 0:
                for tid in recent[-16:]: # recent是最近生成的16个token列表，对新token的候选列表中recent中出现的token进行缩放，防止重复生成
                    logits[0, tid] = logits[0, tid] / repetition_penalty
            if step < min_tokens and hasattr(tok, 'eos_id') and tok.eos_id is not None and tok.eos_id >= 0:
                logits[0, tok.eos_id] = -float('inf') # 生成的token数小于min_tokens时，不能是eos (结束token)
            probs = torch.softmax(logits, dim=-1) # 对logits进行softmax，得到下一个token的概率分布
            if top_k > 0: # 取前k个概率最高的候选token
                '''
                原始概率: [0.5, 0.3, 0.1, 0.05, 0.05]
                top_k = 2
                保留后: [0.5, 0.3, 0, 0, 0]
                归一化后: [0.625, 0.375, 0, 0, 0]
                '''
                v, i = torch.topk(probs, top_k) 
                p = torch.zeros_like(probs).scatter_(1, i, v) # 其余设为0
                s = p.sum(dim=-1, keepdim=True)
                probs = torch.where(s > 0, p / s, probs) # 只保留前K个再进行归一化
            if top_p < 1.0: # 累计概率
                '''
                排序后概率: [0.4, 0.3, 0.2, 0.05, 0.05]
                累积概率: [0.4, 0.7, 0.9, 0.95, 1.0]
                top_p = 0.8
                保留: [0.4, 0.3, 0.2, 0, 0] (累积 0.9 > 0.8，所以保留前 3 个)
                归一化后: [0.444, 0.333, 0.222, 0, 0]
                '''
                srt, idx = torch.sort(probs, descending=True) # 对概率从大到小排序
                c = torch.cumsum(srt, dim=-1)
                m = c <= top_p # 只取累计概率达到top_p的最小token集合 
                srt = srt * m
                p = torch.zeros_like(probs).scatter_(1, idx, srt) # 其余设为0
                s = p.sum(dim=-1, keepdim=True)
                probs = torch.where(s > 0, p / s, probs) # 重新归一化
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0) # 将 NaN、正无穷、负无穷替换为0
            if probs.sum() == 0: # 如果候选token的概率和为0，异常情况（可能含有负值），直接取logits中最大logit的token
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                next_id = torch.multinomial(probs, 1)
            x = torch.cat([x, next_id], dim=1) # 拼接新生成的token(目前都是张量)
            recent.append(next_id.item())
            if next_id.item() == tok.eos_id:
                break
            if stop_strings: # 检查输出是否包含停止字符串（如 \n\n 、 ### 等)
                out_ids = x[0].tolist()[len(prefix):]
                out_text = tok.decode(out_ids)
                if any(out_text.endswith(ss) for ss in stop_strings):
                    break
    out_ids = x[0].tolist()[len(prefix):]
    return _trim_leading_punct(tok.decode(out_ids))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/last.pt")
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--stop_strings", nargs='*', default=None)
    ap.add_argument("--show_label", action="store_true")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu"]) 
    args = ap.parse_args()
    obj = load_checkpoint(args.ckpt)
    cfg = obj["cfg"]
    tok = load_tokenizer(cfg.get("tokenizer", {}).get("type", "byte"), cfg.get("tokenizer", {}).get("path"))
    m = GPT(
        vocab_size=tok.vocab_size,
        n_layer=cfg["model"]["n_layer"],
        n_head=cfg["model"]["n_head"],
        n_embd=cfg["model"]["n_embd"],
        seq_len=cfg["model"]["seq_len"],
        dropout=cfg["model"]["dropout"],
    )
    sd = obj["model"]
    packed = any("_packed_params" in k for k in sd.keys())
    if packed:
        device = torch.device("cpu")
        m = torch.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
    else:
        device = torch.device("cuda") if (args.device == "auto" and torch.cuda.is_available()) else torch.device("cpu")
    m.load_state_dict(sd)
    m.to(device)
    text = generate(m, tok, args.prompt, args.max_new_tokens, args.temperature, args.top_k, args.top_p, args.repetition_penalty, args.stop_strings, device=device)
    if args.show_label:
        print("回答:" + text)
    else:
        print(text)

if __name__ == "__main__":
    main()
