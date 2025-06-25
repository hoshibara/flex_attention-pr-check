## For profiling llama3.1
```bash
python run_llm_inductor_greedy.py -m meta-llama/Meta-Llama-3.1-8B --max-new-tokens 128 \
  --input-tokens 1024 --num-warmup 2 --num-iter 7 --compile --profile
```
