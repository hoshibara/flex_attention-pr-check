import os
import sys
import logging
# Robust logging setup: remove all root handlers and configure logging before any other imports
for handler in logging.root.handlers[:]:
    print("Before setup, remove log root handler: ", handler)
    logging.root.removeHandler(handler)
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=LOG_LEVEL,
    stream=sys.stdout,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger()  # Use root logger for reliability

import contextlib
import torch
import time
import json
import pathlib
import argparse
import csv
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from typing import Any, Callable, Dict, Optional, Tuple, Union

import gc
# Set console encoding to UTF-8
if os.name == 'nt':
    os.system('chcp 65001')
    sys.stdout.reconfigure(encoding='utf-8')

RESULTS_DIR = os.environ.get("RESULTS_DIR", "./logs")
PROFILE_COUNT = 0






# --- Single parser: all arguments, no subparser ---
import argparse

parser = argparse.ArgumentParser("Generation script (fp32/bf16/fp16 path)")
parser.add_argument("-m", "--model-id", type=str, help="the huggingface model id")
parser.add_argument('--sub-model-name', type=str, default="", help="the sub model name for accuracy check")
parser.add_argument("--device", type=str, choices=["cpu", "xpu", "cuda"], default="xpu", help="xpu, cuda or cpu")
parser.add_argument("--dtype", type=str, choices=["float32", "bfloat16", "float16"], default="bfloat16", help="float16, bfloat16, float32")
parser.add_argument("--input-tokens", default="32", type=str, help="input tokens length if needed from prompt.json")
parser.add_argument("--max-new-tokens", default=32, type=int, help="output max new tokens")
parser.add_argument("--prompt", default=None, type=str, help="input prompt for self-defined if needed")
parser.add_argument("--batch-size", default=1, type=int, help="batch size")
parser.add_argument("--num-iter", default=10, type=int, help="num of steps for iteration")
parser.add_argument("--num-warmup", default=3, type=int, help="num of steps for warming up")
parser.add_argument("--num-profile", default=1, type=int, help="num of steps for profiling")
parser.add_argument("--num-beams", default=1, type=int, help="beam width")
parser.add_argument("--greedy", action="store_true")
parser.add_argument("--use-hf-code", default="False", choices=["True", "False"], type=str, help="use hf transformers code")
parser.add_argument("--use-static-cache", action="store_true", help="use static kv cache")
parser.add_argument("--amp", action="store_true", help="whether to enable auto-mixed-precision feature")
parser.add_argument("--inductor", action="store_true")
# debug related args.
parser.add_argument("--profile", action="store_true")
parser.add_argument("--unitrace", action="store_true")
# accuracy related args.
parser.add_argument("--accuracy-only", action="store_true")
parser.add_argument("--acc-tasks", default="gsm8k", type=str, help="tasks list for accuracy validation")
parser.add_argument("--acc-iter", default=-1, type=int)
# Log related args.
parser.add_argument("--print-memory", action="store_true")
parser.add_argument("--token-latency", action="store_true", help="get token latency breakdown")
parser.add_argument("--output-csv-path", default="output.csv", type=str, help="path to output CSV file (default: output.csv)")
# Quantization-specific args
parser.add_argument("--quant-mode", type=str, default=None, choices=["woq", "dynamic_quant"], help="Quantization Mode. (default: None)")
parser.add_argument("--woq", default=False, action="store_true", help="Weight Only Quantization shortcut")
parser.add_argument("--group-size", default=128, type=int, help="group size, default is 128")
parser.add_argument("--ZPFLOAT", action="store_true", help="use float zero point. If not set, it will use ZeroPointDomain.INT")
parser.add_argument("--calibration-samples", type=int, default=10, help="Number of samples for calibration. Default is 10")
parser.add_argument("--model-save-path", type=str, default=None, help="Path to store the scale values")
parser.add_argument("--load-quantize-model", action="store_true", help="Load quantized model. If set, it will load the model from the specified path and apply quantization.")
parser.add_argument("--woq-type", choices=["rtn", "awq"], default="rtn", help="WOQ quantization type, by default, it will be rtn")
parser.add_argument("--quant-dtype", type=str, default="uint4", choices=["unit1", "int4","uint4", "uint8", "int8"], help="The data type of the quantized weights.")
parser.add_argument("--use-hqq", action="store_true", help="Enable HQQ quantization for AWQ (default: False)")
parser.add_argument(
    "--attn-type",
    default="sdpa",
    choices=["flex_attention", "paged_attention", "sdpa"],
    type=str,
)
parser.add_argument(
    "--disable-cpp-wrapper",
    action="store_true",
    help="Disable C++ wrapper for inductor (default: False)",
)
parser.add_argument(
    "--disable-skip-guard-eval",
    action="store_true",
    help="Disable skip guard evaluation (default: False)",
)

args = parser.parse_args()

# Shortcut for setting the quant_mode
if args.woq:
    args.quant_mode = "woq"

args.use_hf_code = args.use_hf_code == "True"
logger.info(args)

do_profiling = os.environ.get("PROFILE", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]
do_profiling = args.profile or do_profiling

# device
device = torch.device(args.device)
device_interface = getattr(torch, args.device, None)
if device_interface is None:
    raise SystemExit(f"Device {args.device} is not supported.")

# adapted from: https://github.com/mit-han-lab/llm-awq/blob/main/awq/entry.py#L255
def get_calib_dataset(tokenizer=None, n_samples=100, block_size=512):
    from datasets import load_dataset
    dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    samples = []
    n_tokens = n_samples * block_size
    n_run = n_tokens
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run -= len(line_encoded)
        if n_run <= n_samples:
            break

    cat_samples = torch.cat(samples, dim=1)
    return [
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_samples)
    ]

# dtype
amp_enabled = True if args.dtype != "float32" and args.amp else False
amp_dtype = getattr(torch, args.dtype)
load_dtype = amp_dtype

# load model
MODEL_CLASSES = {
    "auto": (AutoModelForCausalLM, AutoTokenizer),
    "gpt-j": (AutoModelForCausalLM, AutoTokenizer),
    "gpt-neox": (AutoModelForCausalLM, AutoTokenizer),
    "gpt2": (AutoModelForCausalLM, AutoTokenizer),
    "llama": (AutoModelForCausalLM, AutoTokenizer),
    "opt": (AutoModelForCausalLM, AutoTokenizer),
    "falcon": (AutoModelForCausalLM, AutoTokenizer),
    "bloom": (AutoModelForCausalLM, AutoTokenizer),
    "baichuan2": (AutoModelForCausalLM, AutoTokenizer),
    "qwen": (AutoModelForCausalLM, AutoTokenizer),
    "chatglm3": (AutoModelForCausalLM, AutoTokenizer),
    "phi-3": (AutoModelForCausalLM, AutoTokenizer),
    "mistral": (AutoModelForCausalLM, AutoTokenizer),
    "glm-4": (AutoModelForCausalLM, AutoTokenizer),
    "phi-4": (AutoModelForCausalLM, AutoTokenizer),
}

model_type = next(
    (x for x in MODEL_CLASSES.keys() if x in args.model_id.lower()), "auto"
)
model_class = MODEL_CLASSES[model_type]
config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=args.use_hf_code)
quantization_config = None

# We set it to None by default, so that when model is too large, it will loaded in CPU memory first to avoid OOM.
device_map = None
# Set quantization_config only for RTN
if args.quant_mode == "woq" and args.woq_type == "rtn":
    from torchao.dtypes import Int4XPULayout
    from torchao.quantization.quant_primitives import ZeroPointDomain
    from torchao.quantization import Int4WeightOnlyConfig, quantize_
    from transformers import TorchAoConfig

    if args.device == 'xpu':
        zero_point_domain = ZeroPointDomain.FLOAT if args.ZPFLOAT else ZeroPointDomain.INT
        quant_config = Int4WeightOnlyConfig(group_size=args.group_size, int4_packing_format="plain_int32")
    elif args.device == "cuda":
        quant_config = Int4WeightOnlyConfig(group_size=args.group_size, int4_packing_format="tile_packed_to_4d")
    
    quantization_config = TorchAoConfig(quant_config)

    logger.info(f"Using {args.device} device for int4_weight_only RTN mode, Using TorchAoConfig: {quantization_config}")
    device_map = args.device

# Always load model/tokenizer here, quantization_config is None for AWQ
model = model_class[0].from_pretrained(
    args.model_id,
    torch_dtype=load_dtype,
    config=config,
    low_cpu_mem_usage=True,
    trust_remote_code=args.use_hf_code,
    device_map=device_map,
    quantization_config=quantization_config,
    attn_implementation=args.attn_type,
)
tokenizer = model_class[1].from_pretrained(args.model_id, trust_remote_code=args.use_hf_code)

# For AWQ, quantize after loading
if args.quant_mode == "woq" and args.woq_type == "awq":
    from torchao.quantization.quant_api import (
        _linear_extra_repr,
        _is_linear,
    )
    from torchao.quantization import Int4WeightOnlyConfig, quantize_
    from torchao.prototype.awq import AWQConfig

    if args.load_quantize_model:
        import types
        def load_awq_weight(module, state_dict, name=""):
            observed_linear = module
            qw = state_dict[name + ".weight"]
            linear = torch.nn.Linear(
                observed_linear.in_features,
                observed_linear.out_features,
                observed_linear.bias is not None,
                device=observed_linear.weight.device,
                dtype=observed_linear.weight.dtype,
            )
            linear.weight = torch.nn.Parameter(qw, requires_grad=False)
            linear.extra_repr = types.MethodType(_linear_extra_repr, module)
            linear.bias = observed_linear.bias
            return linear

        def _replace_with_custom_fn_if_matches_filter(
            model, replacement_fn, filter_fn, cur_fqn="", device=None, extra_args: Dict = {}
        ):
            if filter_fn(model, cur_fqn[:-1]):
                model = replacement_fn(model, extra_args, name=cur_fqn[:-1])
                return model
            for name, child in list(model.named_children()):
                new_child = _replace_with_custom_fn_if_matches_filter(
                    child, replacement_fn, filter_fn, f"{cur_fqn}{name}.", device, extra_args
                )
                if new_child is not child and new_child is not None:
                    setattr(model, name, new_child)
            return model

        def load_awq(model, state_dict, filter_fn=None):
            filter_fn = _is_linear if filter_fn is None else filter_fn
            return _replace_with_custom_fn_if_matches_filter(
                model, replacement_fn=load_awq_weight, filter_fn=filter_fn, extra_args=state_dict
            )

        logger.info(f"load awq model from {args.model_save_path}")
        data = torch.load(args.model_save_path)
        model = load_awq(model, data)
        del data
        model = model.to(device)
    else:
        if args.ZPFLOAT:
            raise SystemExit("AWQ does not support float zero point domain")
        # TODO: This will OOM for large models, we need to quant them layer by layer.
        model.eval().to(device)
        quant_dtype = getattr(torch, args.quant_dtype)
        group_size = args.group_size
        logger.info(f"running {quant_dtype} calibration")
        if "cuda" in device.type:
            base_config = Int4WeightOnlyConfig(group_size=group_size)
        elif "xpu" in device.type:
            base_config = Int4WeightOnlyConfig(
                group_size=group_size, int4_packing_format="plain_int32"
            )
        elif "cpu" in device.type:
            base_config = Int4WeightOnlyConfig(
                group_size=group_size, int4_packing_format="opaque"
            )
        else:
            assert False, "Unsupported device: {}".format(device)
        base_config.use_hqq = args.use_hqq
        quant_config = AWQConfig(base_config, step="prepare")

        quantize_(
            model,
            quant_config,
        )
        calibration_data = get_calib_dataset(tokenizer=tokenizer, n_samples=args.calibration_samples, block_size=512)
        for batch in calibration_data:
            if batch.numel() == 0:
                continue
            model(batch.to(device))
        quant_config = AWQConfig(base_config, step="convert")
        quantize_(model, quant_config)
        if args.model_save_path:
            logger.info(f"Saving model to {args.model_save_path}")
            torch.save(model.state_dict(), args.model_save_path)

elif args.quant_mode == "dynamic_quant":
    from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
    config_map = {"int8": Int8DynamicActivationInt8WeightConfig(),}
    quantize_(model, config_map[args.quant_dtype])

model = model.eval().to(device)
def get_memory_usage(name, args):
    if args.print_memory:
        memory_allocated = round(device_interface.memory_reserved() / 1024**3, 3)
        logger.debug(f"{name} memory used total: {memory_allocated} GB")
get_memory_usage("model", args)

if args.inductor:
    from torch._functorch._aot_autograd.subclass_parametrization import unwrap_tensor_subclass_parameters
    # To optimize model with quantized by torchao, which introduce subclass and has more host overhead
    # in dynamo. This function can unwarp subclass and make torch.compile faster.
    # Skip the accuracy only and config.tie_word_embeddings to avoid error.
    if (not args.accuracy_only) or (not config.tie_word_embeddings):
        logger.info(f"For model {args.model_id}, skipped the unwarp_tensor_subclass_parameters. This may affect performance")
        unwrap_tensor_subclass_parameters(model)

    # cpp_warpper makes inductor generate pure C++ code instead of python code for graph execution,
    # which can reduce host overhead. Current AOTI only support Linux platform.
    if sys.platform.startswith("linux"):
        import torch._inductor.config as inductor_config
        inductor_config.cpp_wrapper = not args.disable_cpp_wrapper
    
    model.forward = torch.compile(model.forward)

######################## run lm eval accuracy check ########################

def run_accuracy(model, tokenizer, max_length, tasks=["gsm8k"], device="xpu"):
    import lm_eval
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    from lm_eval.utils import make_table

    model.eval()
    model.config.use_cache = False

    model_eval = lm_eval.models.huggingface.HFLM(pretrained=model, tokenizer=tokenizer)

    results = {}

    # Define tasks and their fewshot values
    eval_tasks = [
        ("truthfulqa_mc2", 0),
        ("winogrande", 5),
        ("arc_challenge", 25),
        ("hellaswag", 10),
        ("gsm8k", 5),
        ("lambda_standard", 0),
    ]

    for tag, fewshot in eval_tasks:
        if tag in tasks:
            eval_kwargs = dict(
                model=model_eval,
                tasks=[tag],
                num_fewshot=fewshot,
                batch_size=args.batch_size,
            )
            if args.acc_iter > 0:
                eval_kwargs["limit"] = args.acc_iter
            results[tag] = lm_eval.evaluator.simple_evaluate(**eval_kwargs)
            logger.info(f"Accuracy Result on {args.model_id}\n{make_table(results[tag])}")

    return results

if args.accuracy_only:
    run_accuracy(model, tokenizer, 128, tasks=args.acc_tasks)
    sys.exit(0)

######################## run generation benchmark ########################
# generate args
generation_config= GenerationConfig(
    cache_implementation="static" if args.use_static_cache else None,
    do_sample=False,
    num_beams=1 if args.greedy else args.num_beams,
    max_new_tokens=int(args.max_new_tokens),
    min_new_tokens=int(args.max_new_tokens),
)

# load prompt
current_path = pathlib.Path(__file__).parent.resolve()
with open(str(current_path) + "/prompt.json", encoding="utf8") as f:
    prompt_pool = json.load(f)

def run_generate(num_tokens, num_input_tokens, num_beams):
    logger.debug(f"*** Starting to generate {num_tokens} tokens for {num_input_tokens} tokens with num_beams={num_beams}")
    if args.prompt is not None:
        prompt = args.prompt
    elif model_type == "auto":
        raise SystemExit(
            "[ERROR] model prompt is not supported, please use --prompt for this model: "
            + args.model_id
        )
    elif int(args.input_tokens) > 8192:
        prompt = prompt_pool[model_type]["8192"] * int(int(args.input_tokens) / 8192)
    elif args.input_tokens in prompt_pool[model_type]:
        prompt = prompt_pool[model_type][args.input_tokens]
    else:
        raise SystemExit("[ERROR] Plese use --prompt if want to use custom input.")

    input_size = tokenizer(prompt, return_tensors="pt").input_ids.size(dim=1)
    logger.debug(f"---- Prompt size: {input_size}")

    generate_kwargs = {}
    if args.token_latency:
        generate_kwargs["token_latency"] = True

    # take the ref_prompt as reference for accuracy check
    f1 = open(os.path.join(os.path.dirname(__file__), "ref_prompt.json"), encoding="utf8")
    prompt_json = json.load(f1)
    f1.close()
    ref_prompt=None
    ref_prompt_cuda=None
    token_support = [(32, 32), (1024, 128)]
    # we take cpu as ref_propmt, hence skip the accuracy check when using cpu.
    if (int(num_input_tokens), num_tokens) in token_support and args.device == "xpu":
        if args.sub_model_name in prompt_json:
            ref_prompt = prompt_json[args.sub_model_name][f"{num_input_tokens}-{num_tokens}"][f"{num_beams}"]
        try:
            ref_prompt_cuda = prompt_json[args.sub_model_name][f"{num_input_tokens}-{num_tokens}"][f"cuda-result: {num_beams}"]
        except Exception:
            pass
    acc_pass = 0

    # profiling context
    sort_by_keyword = "self_" + args.device + "_time_total"
    def trace_handler(p):
        output = p.key_averages(group_by_input_shape=True).table(
            sort_by=sort_by_keyword,
            row_limit=-1,
            max_name_column_width=150,
            max_shapes_column_width=300,
        )
        logger.info(output)
        global PROFILE_COUNT
        PROFILE_COUNT += 1
        p.export_chrome_trace(
            os.path.join(RESULTS_DIR, "trace_step" + str(PROFILE_COUNT) + ".json")
        )

    num_iter = args.num_iter - args.num_warmup
    num_warmup = args.num_warmup
    num_profile = args.num_profile

    profling_context = contextlib.nullcontext()
    if do_profiling:
        if args.unitrace:
            profling_context = torch.autograd.profiler.emit_itt()
        else:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if "xpu" in args.device:
                activities.append(torch.profiler.ProfilerActivity.XPU)
            elif "cuda" in args.device:
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            profling_context = torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                schedule=torch.profiler.schedule(
                    wait=num_iter,
                    warmup=0,
                    active=num_profile,
                    skip_first=num_warmup,
                    repeat=1,
                ),
                on_trace_ready=trace_handler,
            )

    # start
    total_time = 0.0
    prompt = [prompt] * args.batch_size
    total_list = []

    # skip part of guards which is always the same during dynamo cache lookup stage to save host overhead.
    if args.inductor:
        # warmup one iteration to record all guards
        with torch.inference_mode(), torch.no_grad(), torch.autocast(
            device_type=args.device,
            enabled=amp_enabled,
            dtype=amp_dtype if amp_enabled else None,
        ):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            output = model.generate(
                input_ids, generation_config, **generate_kwargs
            )
            device_interface.synchronize()
        # make dynamo clean redundant guards
        torch.compiler.set_stance(skip_guard_eval_unsafe=not args.disable_skip_guard_eval)

    with torch.inference_mode(), torch.no_grad(), torch.autocast(
        device_type=args.device,
        enabled=amp_enabled,
        dtype=amp_dtype if amp_enabled else None,
    ):
        with profling_context as prof:
            for i in range(num_warmup + num_profile + num_iter):
                # Clean the cache to avoid potential OOM
                device_interface.empty_cache()
                gc.collect()
                tic = time.time()
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                output = model.generate(input_ids, generation_config, **generate_kwargs)
                gen_ids = output[0] if args.token_latency else output
                gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                device_interface.synchronize()
                toc = time.time()
                input_tokens_lengths = [x.shape[0] for x in input_ids]
                output_tokens_lengths = [x.shape[0] for x in gen_ids]
                total_new_tokens = [
                    o - i if model.config.model_type != "t5" else o
                    for i, o in zip(input_tokens_lengths, output_tokens_lengths)
                ]
                logger.info(
                    f" Generated text:\n{gen_text},\n Total New Tokens: {total_new_tokens}"
                )
                logger.info(f"Iteration: {i}, Time: {toc - tic:.6f} sec")
                if i >= num_warmup and i < num_warmup + num_iter:
                    total_time += toc - tic
                    if args.token_latency:
                        total_list.append(output[1])
                    if ref_prompt is not None and ref_prompt in gen_text:
                        acc_pass += 1
                    elif ref_prompt_cuda is not None and ref_prompt_cuda in gen_text:
                        acc_pass += 1
                if do_profiling and not args.unitrace:
                    prof.step()

    logger.info("\n" + "-" * 10 + f" {args.model_id} Summary: " + "-" * 10)
    latency = total_time / num_iter 
    logger.info(f"Inference Total Latency: {latency:.5f} sec.")

    first_latency=0
    average_2n_latency=0

    if args.token_latency:
        import numpy as np
        from itertools import chain

        first_latency = np.mean([x[0] for x in total_list])
        average_2n = list(chain(*[x[1:] for x in total_list]))
        average_2n.sort()
        average_2n_latency = np.mean(average_2n)
        logger.info(f"First token latency: {first_latency:.5f} sec.")
        logger.info(f"Next token latency: {average_2n_latency:.5f} sec.")
        logger.info(
            f"First token latency list: {list([x[0] for x in total_list])}"
        )
        logger.info(f"Next token latency list: {average_2n}")

    output_data = {
        "model_name": args.model_id,
        "total_latency": latency,
        "first_token": first_latency if args.token_latency else "N/A",
        "next_token": average_2n_latency if args.token_latency else "N/A",
        "data_type": args.dtype,
        "quant_mode": args.quant_mode if hasattr(args, "quant_mode") else None,
        "woq_type": args.woq_type if hasattr(args, "woq_type") and args.quant_mode == "woq" else None,
        "group_size": args.group_size if hasattr(args, "group_size") and args.quant_mode == "woq" else None,
        "quant_dtype": args.quant_dtype if hasattr(args, "quant_dtype") and args.quant_mode == "woq" else None,
        "ZPFLOAT": args.ZPFLOAT if hasattr(args, "ZPFLOAT") and args.quant_mode == "woq" else None,
        "use_hqq": args.use_hqq if hasattr(args, "use_hqq") and args.quant_mode == "woq" and args.woq_type == "awq" else None,
        "load_quantize_model": args.load_quantize_model if hasattr(args, "load_quantize_model") and args.quant_mode == "woq" and args.woq_type == "awq" else None,
        "input_tokens": args.input_tokens,
        "max_next_tokens": args.max_new_tokens,
        "amp": args.amp,
        "inductor": args.inductor,
        "use_hf_code": args.use_hf_code,
        "use_static_cache": args.use_static_cache,
        "run_date": time.strftime("%Y-%m-%d-%H-%M"),
    }

    def write_to_csv(output_data, csv_file_path):
        file_exists = os.path.isfile(csv_file_path)
        with open(csv_file_path, mode='a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=output_data)
            if not file_exists:
                writer.writeheader()
            writer.writerow(output_data)
    write_to_csv(output_data, args.output_csv_path)

    if args.device == "xpu":
        if ref_prompt is None:
            logger.debug("Accuracy check skip")
        elif acc_pass==num_iter:
            logger.debug("Accuracy check pass")
        else:
            logger.debug(f"Accuracy check fail, the wrong iteration number is: {num_iter - acc_pass}")

def to_list(obj):
    if not isinstance(obj, list):
        return [obj]
    else:
        return obj

for o, i, g in zip(to_list(args.max_new_tokens), to_list(args.input_tokens), to_list(args.num_beams)):
    run_generate(o, i, g)
