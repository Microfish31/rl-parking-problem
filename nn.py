import torch.nn as nn
import torch.nn.functional as F

# class DeepQNetwork(nn.Module):
#     def __init__(self, frame_height, frame_width, n_observations,action_count):
#         super(DeepQNetwork, self).__init__()
        
#         # Define convolutional layers
#         self.conv_layer1 = nn.Conv2d(n_observations, 16, kernel_size=5, stride=2)  # If grayscale, input channels = 1
#         self.batch_norm1 = nn.BatchNorm2d(16)
#         self.conv_layer2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
#         self.batch_norm2 = nn.BatchNorm2d(32)
#         self.conv_layer3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
#         self.batch_norm3 = nn.BatchNorm2d(32)

#         # Compute the output dimensions of the convolutional layers
#         def compute_output_size(size, kernel_size=5, stride=2):
#             return (size - (kernel_size - 1) - 1) // stride + 1

#         processed_width = compute_output_size(compute_output_size(compute_output_size(frame_width)))
#         processed_height = compute_output_size(compute_output_size(compute_output_size(frame_height)))
        
#         # Flattened size after convolutional layers
#         flattened_input_size = processed_width * processed_height * 32
        
#         # Define the final fully connected layer
#         self.final_layer = nn.Linear(flattened_input_size, action_count)

#     def forward(self, x):
#         # Apply convolutional layers with ReLU and batch normalization
#         x = F.relu(self.batch_norm1(self.conv_layer1(x)))
#         x = F.relu(self.batch_norm2(self.conv_layer2(x)))
#         x = F.relu(self.batch_norm3(self.conv_layer3(x)))
        
#         # Flatten the tensor
#         try:
#             x = x.view(x.size(0), -1)
#         except RuntimeError as e:
#             raise ValueError(f"Error in flattening: {e}. Check input dimensions.")
        
#         # Apply the final layer
#         return self.final_layer(x)


class DeepQNetwork(nn.Module):
    def __init__(self, n_observations,n_actions):
        super(DeepQNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        # Ensure the input shape matches
        x = x.view(x.size(0), -1)  # Ensure batch dimension remains unchanged, flatten the other dimensions
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)