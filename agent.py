import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from replay_buffer import ReplayBuffer, ReplayBufferNumpy
from collections import deque
import pickle

import torch
import numpy as np
import pickle
import torch.nn.functional as F


class Agent:
    def __init__(
        self,
        board_size=10,
        frames=2,
        buffer_size=10000,
        gamma=0.99,
        n_actions=3,
        use_target_net=True,
        version="",
    ):
        self._board_size = board_size
        self._frames = frames
        self._buffer_size = buffer_size
        self._n_actions = n_actions
        self._gamma = gamma
        self._use_target_net = use_target_net
        self._version = version
        # self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        self._buffer = ReplayBuffer(
            self._buffer_size, self._board_size, self._frames, self._n_actions
        )

        self._model = None
        self._target_net = None

        self._optimizer = None

    def save_buffer(self, file_path="", iteration=None):
        if iteration is not None:
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open(f"{file_path}/buffer_{iteration:04d}.pkl", "wb") as f:
            pickle.dump(self._buffer, f)

    def load_buffer(self, file_path="", iteration=None):
        if iteration is not None:
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open(f"{file_path}/buffer_{iteration:04d}.pkl", "rb") as f:
            self._buffer = pickle.load(f)

    def get_gamma(self):
        return self._gamma

    def _point_to_row_col(self, point):
        """Converts a point into row and column coordinates."""
        return divmod(point, self._board_size)

    def _row_col_to_point(self, row, col):
        """Converts row and column coordinates into a point."""
        return row * self._board_size + col

    def reset_buffer(self, buffer_size=None):
        """Resets the replay buffer with a new size if provided."""
        if buffer_size is not None:
            self._buffer_size = buffer_size
        self._buffer = ReplayBuffer(
            self._buffer_size, self._board_size, self._frames, self._n_actions
        )

    def get_buffer_size(self):
        return self._buffer.get_current_size()

    def add_to_buffer(
        self, state, action, reward, next_state, done, next_legal_moves=None
    ):
        """Adds an experience to the replay buffer."""
        if next_legal_moves is not None:
            # If next_legal_moves is provided, pass it to the buffer
            self._buffer.add_to_buffer(
                state, action, reward, next_state, done, next_legal_moves
            )
        else:
            # If next_legal_moves is not provided, use a default value (e.g., all zeros)
            legal_moves_default = np.zeros((1, self._n_actions))
            self._buffer.add_to_buffer(
                state, action, reward, next_state, done, legal_moves_default
            )


# Q-Network for DeepQLearningAgent
class QNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(QNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, n_actions),
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        # print("In forward, input shape:", x.shape)
        # permute the input if it's not in the 'channel-first' format
        if x.size(1) != self.conv[0].in_channels:
            x = x.permute(
                0, 3, 1, 2
            )  # change from [batch, H, W, C] to [batch, C, H, W]

        conv_out = self.conv(x)
        conv_out = conv_out.reshape(conv_out.size(0), -1)
        return self.fc(conv_out)


class DeepQLearningAgent(Agent):
    def __init__(
        self,
        board_size=10,
        frames=2,
        buffer_size=10000,
        gamma=0.99,
        n_actions=3,
        use_target_net=True,
        version="",
        lr=0.0005,
    ):
        super().__init__(
            board_size,
            frames,
            buffer_size,
            gamma,
            n_actions,
            use_target_net,
            version,
        )

        self._model = self._get_model()
        self._model.to(self._device)

        if self._use_target_net:
            self._target_net = self._get_model()
            self._target_net.to(self._device)
            self._target_net.load_state_dict(self._model.state_dict())
            self._target_net.eval()

        self._optimizer = optim.Adam(self._model.parameters(), lr=lr)

    def _get_model(self):
        return QNetwork(
            (self._frames, self._board_size, self._board_size), self._n_actions
        )

    def _prepare_input(self, s):
        # normalize the board state
        s_normalized = self._normalize_board(s)

        # convert to PyTorch tensor
        s_tensor = torch.tensor(s_normalized, dtype=torch.float32).to(self._device)

        # Permute the dimensions to match [batch, channels, height, width]
        if s_tensor.ndim == 4 and s_tensor.shape[-1] in [
            self._frames,
            2,
        ]:
            s_tensor = s_tensor.permute(0, 3, 1, 2)

        return s_tensor

    def _get_model_outputs(self, state):
        state_tensor = self._prepare_input(state)
        # get the model's prediction
        with torch.no_grad():
            model_outputs = self._model(state_tensor).cpu().numpy()
        return model_outputs

    def _get_max_output(self):
        """Get the maximum output of Q values from the model."""
        s, _, _, _, _, _ = self._buffer.sample(self._buffer.get_current_size())
        s = self._prepare_input(s)
        max_value = self._model(s).max().item()
        return max_value

    def train_agent(self, batch_size=32, num_games=1, reward_clip=False):
        """Train the model by sampling from buffer and return the loss."""
        s, a, r, next_s, done, _ = self._buffer.sample(batch_size)

        s = self._normalize_board(s)
        s = self._prepare_input(s)
        next_s = self._prepare_input(next_s)

        # Convert 'a' from NumPy array to PyTorch tensor
        if isinstance(a, np.ndarray):
            a = torch.tensor(a, dtype=torch.long).to(self._device)
        else:
            # making sure it's a long type tensor and on the correct device
            a = a.to(dtype=torch.long, device=self._device)

        # convert 'r' and 'done' to PyTorch tensors
        r = torch.tensor(r, dtype=torch.float32).to(self._device)
        done = torch.tensor(done, dtype=torch.float32).to(self._device)

        # get the Q values for current states
        q_values = self._model(s)

        #  convert 'a' to indices
        if a.ndim > 1 and a.shape[1] > 1:
            a = torch.argmax(a, dim=1)

        # Unsqueeze 'a' to match the Q-values dimension for gather
        a = a.unsqueeze(-1)

        # Gather the Q values corresponding to the actions taken
        q_values = q_values.gather(1, a).squeeze(-1)

        # get the next Q values
        if self._use_target_net:
            next_q_values = self._target_net(next_s).max(1)[0]
        else:
            next_q_values = self._model(next_s).max(1)[0]

        r = r.squeeze()  # r should be a 1D tensor
        done = done.squeeze()  # done should be a 1D tensor

        # compute the expected Q values
        expected_q_values = r + self._gamma * next_q_values * (1 - done)

        q_values = q_values.squeeze()
        expected_q_values = expected_q_values.squeeze()

        # loss using Mean Squared Error
        loss = nn.MSELoss()(q_values, expected_q_values)

        # optimize the model
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return loss.item()

    def _update_target(self):
        """Update the target network by copying the model's weights."""
        if self._use_target_net:
            self._target_net.load_state_dict(self._model.state_dict())

    def epsilon_greedy_action(self, state, epsilon=0.1):
        """Select action using epsilon-greedy policy."""
        if np.random.random() < epsilon:
            return np.random.choice(self._n_actions)
        else:
            state_tensor = self._prepare_input(np.array([state]))
            q_values = self._model(state_tensor)
            return q_values.argmax().item()

    def adjust_epsilon(self, initial_epsilon, final_epsilon, decay_steps, step):
        """Decay epsilon over time."""
        return final_epsilon + (initial_epsilon - final_epsilon) * np.exp(
            -step / decay_steps
        )

    def save_model(self, file_path="models/v17.1/model", iteration=None):
        """Save the model weights to the specified path."""
        if iteration is not None:
            # If an iteration number is provided, include it in the filename
            full_path = f"{file_path}_iteration_{iteration}.pth"
        else:
            # Otherwise, use the file_path with .pth extension
            full_path = f"{file_path}.pth"

        torch.save(self._model.state_dict(), full_path)

    def load_model(self, file_path, iteration=None):
        """Load the model weights from the specified path."""
        if iteration is not None:
            # If an iteration number is provided, include it in the filename
            full_path = f"{file_path}_iteration_{iteration}.pth"
        else:
            # Otherwise, use the file_path with .pth extension
            full_path = f"{file_path}.pth"

        self._model.load_state_dict(torch.load(full_path))
        if self._use_target_net:
            self._target_net.load_state_dict(self._model.state_dict())

    def _normalize_board(self, board):
        # Assuming the board is a NumPy array that needs to be normalized before conversion to a tensor.
        # Adjust the normalization to match the TensorFlow version.
        return (board / 4.0).astype(np.float32)

    def move(self, board, legal_moves, values=None):
        # make sure 'board' is a numpy array
        assert isinstance(board, np.ndarray), "Board must be a numpy array"
        # Add batch dimension
        if len(board.shape) == 3:
            board = np.expand_dims(board, axis=0)
        assert (
            len(board.shape) == 4
        ), "Board must be 4D with shape [batch, height, width, channels]"

        # Prepare the state tensor by converting the numpy array to a PyTorch tensor and moving it to the correct device
        state_tensor = torch.tensor(board, dtype=torch.float32).to(self._device)

        # Permute the dimensions to match [batch, channels, height, width]
        state_tensor = state_tensor.permute(0, 3, 1, 2)

        # passing the state tensor to the model
        with torch.no_grad():
            q_values = self._model(state_tensor)

        q_values_np = q_values.cpu().numpy()

        assert isinstance(legal_moves, np.ndarray), "Legal moves must be a numpy array"
        assert legal_moves.shape == (
            board.shape[0],
            self._n_actions,
        ), "Legal moves should have shape [batch, n_actions]"

        # masking the illegal actions by setting their Q values to -inf
        masked_q_values = np.where(legal_moves, q_values_np, -np.inf)

        # select the action with the highest Q value for each batch
        return np.argmax(masked_q_values, axis=1)

    def update_target_net(self):
        if self._use_target_net:
            self._target_net.load_state_dict(self._model.state_dict())

    def get_action_proba(self, state):
        state_tensor = self._prepare_input(np.array([state]))
        with torch.no_grad():
            q_values = self._model(state_tensor)
            action_probabilities = F.softmax(q_values, dim=1)
        return action_probabilities.cpu().numpy()
