## For profiling llama3.1
```bash
python run_llm_inductor_greedy.py -m meta-llama/Meta-Llama-3.1-8B --max-new-tokens 128 \
  --input-tokens 1024 --num-warmup 2 --num-iter 7 --compile --profile
```

## Full steps for profiling llama3.1
1. Install pytorch & triton:
```bash
## install pytorch
git clone https://github.com/pytorch/pytorch.git
cd pytorch
gh pr checkout 143553
cd ..

cd pytorch
pip uninstall torch -y
pip uninstall torch -y
pip uninstall torch -y
git submodule sync
git submodule update --init --recursive
pip install -r requirements.txt
python setup.py develop
cd ..

## install triton
git clone https://github.com/intel/intel-xpu-backend-for-triton.git triton
pip uninstall triton -y
pip uninstall triton -y
pip uninstall triton -y
cd triton
git submodule sync
git submodule update --init --recursive
scripts/compile-triton.sh
cd ..

## install deps
pip uninstall torchvision torchaudio -y
pip uninstall torchvision torchaudio -y
pip uninstall torchvision torchaudio -y
pip install --pre torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu --no-deps
```

2. install transformers
```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
# git checkout xxx  # can be found transformers-commit.txt
git apply ../transformers-patch-for-timing.diff
git submodule sync
git submodule update --init --recursive
python setup.py develop
cd ..
```
