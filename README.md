# RL for Token Swapping Problem

This repository compares different RL methods for solving the Token Swap Problem, including curriculum learning approaches.

## Overview

The Token Swap Problem involves rearranging tokens on a graph by swapping adjacent tokens until reaching a target configuration. This repository implements various reinforcement learning approaches to solve this problem efficiently.

## Features

- **Multiple RL Algorithms**: PPO-based solutions with different wrapper strategies
- **Curriculum Learning**: Progressive difficulty scaling by increasing node count
- **Experimental Framework**: Comprehensive experimentation with Optuna optimization
- **Visualization**: TensorBoard integration for training monitoring

## Curriculum Learning

The `CurriculumCallback` enables progressive learning by gradually increasing the complexity of the token swap environment:

### How it Works
1. **Start Simple**: Begin training with fewer nodes (e.g., 4 nodes)
2. **Monitor Performance**: Track success rate and episode rewards
3. **Increase Difficulty**: When success rate exceeds threshold (e.g., 70%), increase node count
4. **Progressive Scaling**: Continue until reaching target complexity (e.g., 16+ nodes)

### Benefits
- **Faster Convergence**: Learn basic strategies on simple problems first
- **Better Generalization**: Gradually build complexity understanding
- **Stable Training**: Avoid overwhelming the agent with complex scenarios initially

## Usage

### Standard Training
```bash
# Run existing experiments
run_rl.bat
```

### Curriculum Learning
```bash
# Run curriculum learning experiment
run_curriculum.bat
```

### Custom Curriculum Setup
```python
from src.callbacks.Curriculum_callback import CurriculumCallback

# Create curriculum callback
curriculum_callback = CurriculumCallback(
    initial_node_num=4,      # Start with 4 nodes
    target_node_num=16,      # Scale up to 16 nodes
    success_threshold=0.7,   # Move to next level at 70% success
    min_episodes_before_increase=50,  # Minimum episodes per level
    check_freq=5000,         # Evaluation frequency
    verbose=2                # Logging level
)

# Use in training
model.learn(total_timesteps=1000000, callback=curriculum_callback)
```

## Directory Structure

- `src/envs/`: Environment implementations
- `src/wrappers/`: Environment wrappers for different approaches
- `src/callbacks/`: Training callbacks including curriculum learning
- `src/exps/`: Experimental scripts
- `result/`: Training results and saved models

## Experiments

1. **candidateDistance_experiment**: Distance-based candidate selection
2. **candidateGraph_experiment**: Graph-based candidate selection  
3. **curriculum_experiment**: Progressive curriculum learning
4. **candidate_experiment**: Basic candidate approach

## Requirements

See `requirements.txt` for detailed dependencies.