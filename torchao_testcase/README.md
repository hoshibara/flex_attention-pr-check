## Install transformers
```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
git remote add hoshibara https://github.com/hoshibara/transformers.git
git remote -v
git fetch hoshibara
git branch -vv
# git checkout v4.51.3
# git apply --ignore-space-change --ignore-whitespace /home/xingyuan/projects/20250910-FA/flex_attention-pr-check/torchao_testcase/transformers.patch
git checkout hoshibara/FA-latency
cd ..

pip uninstall transformers -y
pip uninstall transformers -y
pip uninstall transformers -y
cd transformers
git submodule sync
git submodule update --init --recursive
python setup.py develop
cd ..
```

```bash
# self build transformers for collecting token latency
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout v4.55.4
git apply --ignore-space-change --ignore-whitespace ../patches/transformers.patch 
python setup.py install
cd ..
pip install -r requirements.txt
```
