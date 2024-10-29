"""
tests/test_vla.py

This script demonstrates how to use the dummy components to create a dummy SkillVLA model.

Usage:
>>> python tests/test_vla.py
"""

import random

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from skillvla.components.action_head.base_action_head import ActionHead
from skillvla.components.lang_encoder.base_lang_encoder import LanguageEncoder
from skillvla.components.llm_backbone.base_llm_backbone import LLMBackbone
from skillvla.components.obs_encoder.base_obs_encoder import ObservationEncoder
from skillvla.components.skill_selector.base_skill_selector import SkillSelector
from skillvla.models.skillvla import SkillVLA


class DummyActionHead(ActionHead):
    def __init__(self, action_head_id, embedding_dim: int = 512, output_dim: int = 7):
        super().__init__(action_head_id)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.mlp(embedding)


class DummyLanguageEncoder(LanguageEncoder):
    def __init__(self, lang_encoder_id: str, embedding_dim: int = 512):
        super().__init__(lang_encoder_id)
        self.embedding_dim = embedding_dim

    def forward(self, text: str):
        words = text.split()
        num_words = len(words)
        embeddings = torch.randn(num_words, self.embedding_dim)

        return embeddings


class DummyLLMBackbone(LLMBackbone):
    def __init__(self, llm_id: str, input_dim: int = 512, hidden_dim: int = 1024):
        super().__init__(llm_id)
        self.llm = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, embeddings: torch.Tensor):
        return self.llm(embeddings)[-1]


class DummyObservationEncoder(ObservationEncoder):
    def __init__(self, obs_encoder_id: str, input_dim: int = 512, image_shape: tuple = (224, 224)):
        super().__init__(obs_encoder_id)
        self.input_dim = input_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # Output: [16, 112, 112]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Output: [32, 56, 56]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: [64, 28, 28]
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, input_dim),
            nn.ReLU(),
        )

        # Image preprocessing transformations
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_shape),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def forward(self, image: Image):
        image_tensor = self.transform(image).unsqueeze(0)
        embedding = self.encoder(image_tensor)
        return embedding.squeeze()


class DummySkillSelector(SkillSelector):
    def __init__(self, skill_selector_id: str, num_skills: int = 50, embedding_dim: int = 512):
        super().__init__(skill_selector_id)
        self.skill_library = [torch.randn(embedding_dim) for _ in range(num_skills)]

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        # Select a random skill embedding from the skill library
        random_skill = random.choice(self.skill_library)
        return random_skill


def get_dummy_vla():
    dummy_vla = SkillVLA(
        model_id="dummy_vla",
        obs_encoder=DummyObservationEncoder("dummy_obs_encoder"),
        lang_encoder=DummyLanguageEncoder("dummy_lang_encoder"),
        llm_backbone=DummyLLMBackbone("dummy_llm_backbone"),
        action_head=DummyActionHead("dummy_action_head"),
        skill_selector=DummySkillSelector("dummy_skill_selector"),
    )
    return dummy_vla


if __name__ == "__main__":
    # Create a dummy VLA model
    dummy_vla = get_dummy_vla()
    print(dummy_vla)

    # Initialize each dummy component
    lang_encoder = DummyLanguageEncoder("dummy_lang_encoder")
    llm_backbone = DummyLLMBackbone("dummy_llm_backbone")
    obs_encoder = DummyObservationEncoder("dummy_obs_encoder")
    action_head = DummyActionHead("dummy_action_head")

    # Sample inputs
    text = "This is a dummy text."
    image = Image.new("RGB", (224, 224), color="blue")

    # Step 1: Encode the text
    text_embeddings = lang_encoder.forward(text)  # Shape: (num_words, embedding_dim)
    # Step 2: Encode the image
    image_embedding = obs_encoder.forward(image)  # Shape: (embedding_dim,)
    # Step 3: Process the inputs through the LLM backbone
    next_token_embedding = llm_backbone.forward(torch.cat([text_embeddings, image_embedding.unsqueeze(0)], dim=0))  # Shape: (embedding_dim,)
    # Step 4: Predict the action
    action_logits = action_head.forward(next_token_embedding)  # Shape: (output_dim,)
    print(action_logits)
