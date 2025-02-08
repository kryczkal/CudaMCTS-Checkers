# MCTS Checkers

<div style="text-align: center;">
  <img src="https://github.com/user-attachments/assets/c784bb6c-1472-4d0e-acb9-e3b773ac6281" alt="3">
  <img src="https://github.com/user-attachments/assets/4d26cc80-48de-4cb8-a4e9-9f5dc510efa7" alt="4">
  <img src="https://github.com/user-attachments/assets/712a14ed-6aad-470a-956b-bf62a082a8dd" alt="5">
  <img src="https://github.com/user-attachments/assets/73b253a7-91cb-4247-81ca-2819e0b4fd6d" alt="2">
</div>

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)

## Overview
Repository implements a Monte Carlo Tree Search (MCTS) engine for Checkers. It supports multi-threaded CPU and GPU (CUDA) simulation. The system builds an MCTS tree concurrently and simulates game states using the selected backend.

## Repository Structure
- **src/**  
  - Contains C++ and CUDA source files.  
  - CUDA files (.cu): Kernels for game simulation, move generation, move selection, board operations, and CUDA utilities.  
  - C++ files (.cpp): CPU implementations for move generation, MCTS tree construction, game engine, and related utilities.
- **include/**  
  - Header files for CPU modules, CUDA kernels, MCTS tree, game engine, board helpers, and common definitions.
- **tests/**  
  - Unit tests for individual modules.
- **CMakeLists.txt**  
  - Build configuration for CMake.
- **main.cu**  
  - Application entry point and interactive command-line interface.

## Features
- **Backend Options**  
  - Multi-threaded CPU version.  
  - GPU version using CUDA.
- **CUDA Kernels**  
  - Optimized with shared memory to reduce global memory accesses.  
  - Use of constant memory for lookup tables.  
  - Branchless move generation to minimize divergence.
- **MCTS Implementation**  
  - Concurrent tree construction.  
  - Simulation of game states using the chosen backend.  
  - UCT-based move selection.
- **Game Engine**  
  - Implements Checkers rules: multi-capture, king promotion, and draw conditions.
- **Interactive CLI**  
  - Allows selection of game mode, backend, and simulation parameters.

## Requirements
- C++ compiler with C++20 support.
- NVIDIA CUDA Toolkit (compatible with CUDA architectures 75, 80, 86).
- CMake version 3.20 or higher.
- Git.
- Make

## Installation
1. Clone repository:
   
   ```bash
   git clone https://github.com/kryczkal/CudaMCTS-Checkers
   cd CudaMCTS-Checkers
   ```

2. Create build directory and run CMake:
   
   ```bash
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   ```

3. Build project:
   
   ```bash
   make
   ```

## Usage
Run the executable from the build directory:
   
   ```bash
   ./MCTS_CHECKERS
   ```

The executable prompts for backend selection, game mode, time limits, and player configuration. Simulations and MCTS tree expansions are processed concurrently using the chosen backend.
