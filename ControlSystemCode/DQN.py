import random
import math
import networkx
import torch
import torch.nn as nn
from datetime import datetime
from collections import namedtuple

device ="cpu"

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class DqnAgent:
    def __init__(
        self,
    ):
        self = self


    def select_action(self, state, steps_done, invalid_action):
        return True

    def learn(self):
        return True