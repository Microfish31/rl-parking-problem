# rl-parking-problem

## Introduction
This project aims to solve the parking problem automatically using reinforcement learning.

## Definition

### Environment
The environment consists of a car that needs to find a parking spot within a parking lot while avoiding collisions with obstacles.

- **30 parking spaces** (1 available)
- **3 fixed obstacles**
- **1 car**

### State Space
- The **observation** is the same as the **achieved goal**.  
- The **desired goal** is the position of the parking space.

### Action Space
- **Steering range:** \(-0.5\) to \(0.5\)  
- **Acceleration range:** \(-0.4\) to \(0.4\)  

To convert continuous actions into discrete actions:  

|   | **Steering (m)** | **Acceleration (n)** |
|---|------------------|----------------------|
| 1 | 11              | 4                    |

### Reward Function
- **Collision penalty:** \(-5\)  
- **Distance-based reward:** The reward is determined by the distance to the target:

$$ R = -(|a - d| * w)^P $$