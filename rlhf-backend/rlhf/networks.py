import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Params Actor
LOG_STD_MAX = 2
LOG_STD_MIN = -5

# Q-Netzwerk
class SoftQNetwork(nn.Module):
    # Netzarchitektur
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256) # erste Schicht
        self.fc2 = nn.Linear(256, 256) # zweite Schicht
        self.fc3 = nn.Linear(256, 1) # dritte Schicht -> eindeutiger Wert

    # Q(s,a) -> Q-Wert
    def forward(self, x, a):
        x = torch.cat([x, a], 1) # Verknüpfung von Zustand und Aktion
        x = F.relu(self.fc1(x)) # erste Schicht - ReLU
        x = F.relu(self.fc2(x)) # zweite Schicht - ReLU
        x = self.fc3(x) # dritte Schicht -> Q-Wert
        return x



class Actor(nn.Module):
    # Netzarchitektur
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256) # erste Schicht
        self.fc2 = nn.Linear(256, 256) # zweite Schicht

        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape)) # Mittelwert
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape)) # log-Standardabweichung

        # Rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    # Umgebungszustand -> Mittelwert, log-Standardabweichung
    def forward(self, x):
        x = F.relu(self.fc1(x)) # erste Schicht - ReLU
        x = F.relu(self.fc2(x)) # zweite Schicht - ReLU
        mean = self.fc_mean(x) # Mittelwert
        log_std = self.fc_logstd(x)                                                 # log-Standardabweichung
        log_std = torch.tanh(log_std)                                               # Einschränkung auf [-1, 1]
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)   # Skalierung auf [MIN, MAX]
        return mean, log_std

    # Umgebungszustand -> Aktion, log-Wahrscheinlichkeit, Mittelwert
    def get_action(self, x):
        mean, log_std = self(x) # forward -> Mittelwert, log-Standardabweichung
        std = log_std.exp() # Umwandlung in Standardabweichung
        normal = torch.distributions.Normal(mean, std) # Normalverteilung
        x_t = normal.rsample() # Reparametrisiertes Sampling (?)
        y_t = torch.tanh(x_t) # Einschränkung auf [-1, 1]
        action = y_t * self.action_scale + self.action_bias # Reskalierung auf gültigen Aktionsbereich
        log_prob = normal.log_prob(x_t)                                     # log-Wahrscheinlichkeit
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)  # Jacobian-Anpassung
        log_prob = log_prob.sum(1, keepdim=True)                            # Gesamt-log-Wahrscheinlichkeit
        mean = torch.tanh(mean) * self.action_scale + self.action_bias # Mittelwerte der Aktionsverteilung
        return action, log_prob, mean