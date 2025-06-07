#!/usr/bin/env python3

class ModalityConfig:
    def __init__(self, delta_indices, modality_keys, shapes_list=None):
        self.delta_indices = delta_indices
        self.modality_keys = modality_keys
        self.shapes_list = shapes_list