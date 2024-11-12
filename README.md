# Learning Tools

The goal of this repo is _finally_ make public versions of the tools I've been using for machine learning experiments throughout my career. These include:
- Useful config-based utilities for training models
- Reasonable patterns for downloading data, data processing, model configuration, and logging.
- A few toy models to test new ideas


I make use of comments throughout the codebase to help explain why I've made certain design decisions, or otherwise explain tradeoffs in the hope that it will help others learn from my experiences.

# Highlights

- We use `uv` for package management. To install the development version, run:

```
uv pip install -e .
```

# Quickstart

To execute the training, run the following commands.

```
uv run src/nanogpt_uniprot/data/cached_fineweb10B.py
bash run.sh
```

## Prerequesites
### On WSL
install GCC
```
sudo apt update
sudo apt install build-essential
```
After installation, you can verify that GCC is installed by running:
```
gcc --version
```


# TODOs

- [x] Reproduce the original modded-nanogpt demo
- [ ] Set up config to support
  - [ ] Different models
  - [ ] Different datasets
  - [ ] Optimization schemas
- [ ] Reproduce the original modded-nanogpt with UniRef data
  - [ ] Download scripts for UniRef 100 and 50
    - [ ] Splits for train/val/test
  - [ ] Create dataloaders for uniref
    - [ ] Pre-tokenized sequences
  - [ ] Performance evaluation
- [ ] Get running demo
 