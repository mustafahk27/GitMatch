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
    def __init__(self, num_users: int, num_languages: int, num_topics: int, embed_dim: int = 64, out_dim: int = 128, repo_emb_dim: int = 384):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.lang_embed = nn.Embedding(num_languages + 1, embed_dim, padding_idx=0)
        self.topic_embed = nn.Embedding(num_topics + 1, embed_dim, padding_idx=0)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 3 + repo_emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, out_dim),
        )

    def forward(self, user_ids: torch.Tensor, lang_ids: torch.Tensor, topic_ids: torch.Tensor, mean_repo_emb: torch.Tensor) -> torch.Tensor:
        u = self.user_embed(user_ids)
        l = self.lang_embed(lang_ids).mean(dim=1)
        t = self.topic_embed(topic_ids).mean(dim=1)
        return F.normalize(self.fc(torch.cat([u, l, t, mean_repo_emb], dim=-1)), dim=-1)


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

    def forward(self, user_ids, user_lang_ids, user_topic_ids, mean_repo_emb, repo_text_embs, repo_lang_ids) -> torch.Tensor:
        user_emb = self.user_tower(user_ids, user_lang_ids, user_topic_ids, mean_repo_emb)
        repo_emb = self.repo_tower(repo_text_embs, repo_lang_ids)
        return (user_emb * repo_emb).sum(dim=-1)

    def encode_user(self, user_ids, user_lang_ids, user_topic_ids, mean_repo_emb) -> torch.Tensor:
        return self.user_tower(user_ids, user_lang_ids, user_topic_ids, mean_repo_emb)

    def encode_repo(self, repo_text_embs, repo_lang_ids) -> torch.Tensor:
        return self.repo_tower(repo_text_embs, repo_lang_ids)


class BPRDataset(Dataset):
    """Bayesian Personalized Ranking dataset — returns (user, pos_repo, neg_repo) triplets."""

    def __init__(self, positives: list[dict], all_repos: list[dict]):
        self.positives = positives
        self.all_repos = all_repos
        self.n_repos = len(all_repos)

    def __len__(self):
        return len(self.positives)

    def __getitem__(self, idx):
        pos = self.positives[idx]
        neg = self.all_repos[np.random.randint(self.n_repos)]
        return {
            "user_id": pos["user_id"],
            "user_lang_ids": torch.tensor(pos["user_lang_ids"], dtype=torch.long),
            "user_topic_ids": torch.tensor(pos["user_topic_ids"], dtype=torch.long),
            "mean_repo_emb": torch.tensor(pos["mean_repo_emb"], dtype=torch.float),
            "pos_text_emb": torch.tensor(pos["repo"]["text_emb"], dtype=torch.float),
            "pos_lang_id": torch.tensor(pos["repo"]["lang_id"], dtype=torch.long),
            "neg_text_emb": torch.tensor(neg["text_emb"], dtype=torch.float),
            "neg_lang_id": torch.tensor(neg["lang_id"], dtype=torch.long),
        }


def train(model: TwoTowerModel, dataset: BPRDataset, epochs: int = 10, lr: float = 1e-3, device: str = "cpu"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=0)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            u_emb = model.encode_user(
                batch["user_id"].to(device),
                batch["user_lang_ids"].to(device),
                batch["user_topic_ids"].to(device),
                batch["mean_repo_emb"].to(device),
            )
            pos_emb = model.encode_repo(batch["pos_text_emb"].to(device), batch["pos_lang_id"].to(device))
            neg_emb = model.encode_repo(batch["neg_text_emb"].to(device), batch["neg_lang_id"].to(device))
            pos_score = (u_emb * pos_emb).sum(dim=-1)
            neg_score = (u_emb * neg_emb).sum(dim=-1)
            loss = -F.logsigmoid(pos_score - neg_score).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs} — loss: {total_loss / len(loader):.4f}")

    return model
