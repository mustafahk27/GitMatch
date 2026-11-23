"""
Two-Tower Neural Network for repository recommendation.

User tower encodes the user's interaction history and language/topic preferences.
Repo tower encodes repository text embeddings and metadata.
Affinity score = dot product of both towers' output embeddings.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class UserTower(nn.Module):
    def __init__(self, num_users: int, num_languages: int, num_topics: int, embed_dim: int = 64, out_dim: int = 128):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.lang_embed = nn.Embedding(num_languages + 1, embed_dim, padding_idx=0)
        self.topic_embed = nn.Embedding(num_topics + 1, embed_dim, padding_idx=0)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, out_dim),
        )

    def forward(self, user_ids: torch.Tensor, lang_ids: torch.Tensor, topic_ids: torch.Tensor) -> torch.Tensor:
        u = self.user_embed(user_ids)
        l = self.lang_embed(lang_ids).mean(dim=1)
        t = self.topic_embed(topic_ids).mean(dim=1)
        return F.normalize(self.fc(torch.cat([u, l, t], dim=-1)), dim=-1)


class RepoTower(nn.Module):
    def __init__(self, text_embed_dim: int = 384, num_languages: int = 100, out_dim: int = 128):
        super().__init__()
        self.lang_embed = nn.Embedding(num_languages + 1, 32, padding_idx=0)
        self.fc = nn.Sequential(
            nn.Linear(text_embed_dim + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, out_dim),
        )

    def forward(self, text_embeddings: torch.Tensor, lang_ids: torch.Tensor) -> torch.Tensor:
        l = self.lang_embed(lang_ids)
        x = torch.cat([text_embeddings, l], dim=-1)
        return F.normalize(self.fc(x), dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, num_users: int, num_languages: int, num_topics: int, text_embed_dim: int = 384, out_dim: int = 128):
        super().__init__()
        self.user_tower = UserTower(num_users, num_languages, num_topics, out_dim=out_dim)
        self.repo_tower = RepoTower(text_embed_dim, num_languages, out_dim=out_dim)

    def forward(self, user_ids, user_lang_ids, user_topic_ids, repo_text_embs, repo_lang_ids) -> torch.Tensor:
        user_emb = self.user_tower(user_ids, user_lang_ids, user_topic_ids)
        repo_emb = self.repo_tower(repo_text_embs, repo_lang_ids)
        return (user_emb * repo_emb).sum(dim=-1)

    def encode_user(self, user_ids, user_lang_ids, user_topic_ids) -> torch.Tensor:
        return self.user_tower(user_ids, user_lang_ids, user_topic_ids)

    def encode_repo(self, repo_text_embs, repo_lang_ids) -> torch.Tensor:
        return self.repo_tower(repo_text_embs, repo_lang_ids)


class InteractionDataset(Dataset):
    def __init__(self, positives: list[dict], all_repos: list[dict], num_negatives: int = 4):
        self.positives = positives
        self.all_repos = all_repos
        self.num_negatives = num_negatives
        self.repo_ids = list(range(len(all_repos)))

    def __len__(self):
        return len(self.positives) * (1 + self.num_negatives)

    def __getitem__(self, idx):
        pos_idx = idx // (1 + self.num_negatives)
        is_positive = (idx % (1 + self.num_negatives)) == 0
        pos = self.positives[pos_idx]
        label = 1.0 if is_positive else 0.0
        repo = pos["repo"] if is_positive else self.all_repos[np.random.randint(len(self.all_repos))]
        return {
            "user_id": pos["user_id"],
            "user_lang_ids": torch.tensor(pos["user_lang_ids"], dtype=torch.long),
            "user_topic_ids": torch.tensor(pos["user_topic_ids"], dtype=torch.long),
            "repo_text_emb": torch.tensor(repo["text_emb"], dtype=torch.float),
            "repo_lang_id": torch.tensor(repo["lang_id"], dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float),
        }


def train(model: TwoTowerModel, dataset: InteractionDataset, epochs: int = 10, lr: float = 1e-3, device: str = "cpu"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=0)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            scores = model(
                batch["user_id"].to(device),
                batch["user_lang_ids"].to(device),
                batch["user_topic_ids"].to(device),
                batch["repo_text_emb"].to(device),
                batch["repo_lang_id"].to(device),
            )
            loss = loss_fn(scores, batch["label"].to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs} — loss: {total_loss / len(loader):.4f}")

    return model
