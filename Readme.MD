# Inferra-LLM

Inferra is a high-performance adaptive inference runtime built for maximizing large language model throughput under real-world workload variability.

It is designed for the **FlagOS Open Computing Global Challenge (Track 2: Inference Optimization)** and targets efficient serving of **Qwen3-4B** using a dynamic, workload-aware execution strategy built on top of a vLLM backend.

---

## Overview

Modern LLM serving systems typically rely on static assumptions:
- fixed batching strategies  
- static decoding configurations  
- manually tuned performance knobs  

Inferra replaces this with a runtime that continuously adapts to workload conditions in real time.

Instead of forcing the workload into a fixed execution pattern, Inferra reshapes execution around the workload itself.

---

## Key Idea

At its core, Inferra behaves like a lightweight inference control system:

- observes incoming request patterns  
- adjusts batching and scheduling dynamically  
- switches execution behavior based on load conditions  
- continuously optimizes for throughput without sacrificing latency stability  

The result is a system that behaves more like a **self-tuning inference runtime** than a traditional pipeline.

---

## What It Optimizes

Inferra focuses on four key system dimensions:

- **Throughput efficiency** (tokens/sec maximization)  
- **Latency stability** (avoiding spikes under load)  
- **Memory utilization** (KV cache efficiency)  
- **Runtime adaptability** (workload-aware execution changes)  

---

## System Behavior

Rather than using fixed policies, Inferra continuously adapts between two execution styles:

### Latency-Oriented Mode
Used when the system is lightly loaded or serving interactive traffic:
- smaller batches  
- faster response prioritization  
- reduced queuing delay  

### Throughput-Oriented Mode
Activated under higher load conditions:
- larger, more efficient batches  
- improved GPU utilization  
- optimized token processing density  

Mode selection is driven entirely by runtime signals.

---

## Benchmarking & Evaluation

Inferra includes a full evaluation suite designed to measure real system behavior under different workloads:

- controlled throughput benchmarking  
- stress testing under increasing load pressure  
- deterministic replay-based evaluation  
- formal scoring pipeline for submission consistency  

This ensures that performance improvements are measurable, reproducible, and comparable across runs.

---

## Design Principles

Inferra is built with a systems-first mindset:

- **Adaptivity over static tuning**  
- **Stability under load over aggressive optimization tricks**  
- **Modularity for experimentation and extension**  
- **Reproducible evaluation for fair benchmarking**  

The goal is not just to optimize inference speed, but to demonstrate a runtime that reacts intelligently to workload dynamics.

---

## Getting Started

Install dependencies:

```bash
pip install -r requirements.txt