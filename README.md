# FunGPT

FunGPT is a lightweight GPT-2 style language model built from scratch in PyTorch, 
designed for educational exploration and efficient training on limited hardware. 
It supports both character-level and token-level autoregressive modeling, and is 
inspired by my deep learning class,Andrej Karpathy's work on nanogpt and 
the original GPT-2 architecture.

## ✨ Features

- ✅ Autoregressive transformer architecture (GPT-style)
- ✅ Character-level and token-level modeling modes
- ✅ Masked self-attention with causal masking
- ✅ Positional Encoding (PE) support
- ✅ Clean modular code, easy to extend or fine-tune
- ⏳ Rotary Positional Embedding (RoPE) — *coming soon*
- ⏳ Efficient training with batching and gradient accumulation
- ⏳ Trainer and logging system for quick experimentation


## 🚀 Motivation

This project was born out of curiosity and a desire to deeply understand the internals of transformer-based language models. 
It started as a school homework on transformers at CMU
and evolved into a hands-on learning tool inspired by Andrej Karpathy's educational examples.

FunGPT is **not** intended to compete with production-grade models — 
it's designed to be minimal, educational, and run on modest hardware (e.g., a single GPU or even CPU with patience!).
The code is full of comments and explanations to help you and ME grasp the concepts behind the architecture and training process.

## 🛠 Architecture Overview

The core architecture mimics GPT-2:

- Multi-layer Transformer decoder stack
- Causal (masked) self-attention
- LayerNorm and GELU activations
- Dropout for regularization
- Learned token/letter embeddings and positional embeddings

Currently uses standard sinusoidal or learned positional encodings. 
Support for [RoPE (Rotary Positional Embeddings)](https://arxiv.org/abs/2104.09864) is planned in a future update.

## 📚 Training

FunGPT includes a flexible training pipeline with:

- Dataset loading (plain text or tokenized)
- Efficient data batching
- Gradient accumulation and checkpointing
- Logging training/validation loss

Example training scripts are included for character-level modeling and custom corpora.

