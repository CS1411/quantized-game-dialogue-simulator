# Quantized Game Dialogue Simulator

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/8rjg8qhkbp/badge.svg)](https://ask.thili.ai/b/8rjg8qhkbp)

Train a small GPT on game dialogue, shrink it with post-training quantization, and simulate what that does to memory footprint and generation speed on resource-constrained devices.

## What you are building

**Goal.** A small quantized GPT for in-game dialogue, plus a simulation of its memory footprint and generation speed on constrained devices.

**Done when.** A small GPT is trained on a custom game dialogue corpus, dynamic and static post-training quantization are applied, and the effect on model size and simulated inference speed is measured.

## Milestones

- [x] **1. Prepare a small game dialogue text corpus.**
- [x] **2. Implement a basic GPT architecture and train it on the corpus.**
- [x] **3. Generate sample dialogue from the trained GPT.**
- [ ] **4. Apply dynamic and static post-training quantization to the model.**
- [ ] **5. Measure and compare the original and quantized model sizes.**
- [ ] **6. Simulate inference speed by comparing operations on original versus quantized weights.**
