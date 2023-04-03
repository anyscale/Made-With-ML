# madewithml/models.py
import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from madewithml import utils


class FinetunedLLM(nn.Module):
    def __init__(self, llm, dropout_p, embedding_dim, num_classes):
        super(FinetunedLLM, self).__init__()
        self.llm = llm
        self.dropout = torch.nn.Dropout(dropout_p)
        self.fc1 = torch.nn.Linear(embedding_dim, num_classes)

    def forward(self, inputs):
        ids, masks = inputs
        seq, pool = self.llm(input_ids=ids, attention_mask=masks)
        z = self.dropout(pool)
        z = self.fc1(z)
        return z

    def predict_proba(self, inputs):
        self.eval()
        y_probs = []
        with torch.inference_mode():
            z = self(inputs)
            y_probs = F.softmax(z).cpu().numpy()
        return y_probs

    def predict(self, inputs):
        self.eval()
        with torch.inference_mode():
            z = self(inputs)
        return torch.argmax(z, dim=1).cpu().numpy()


def load_model(run_id):
    """Load model using a specific run ID."""
    llm = BertModel.from_pretrained("allenai/scibert_scivocab_uncased", return_dict=False)
    run_info = mlflow.get_run(run_id=run_id).to_dictionary()
    params = utils.get_best_params(artifact_uri=run_info["info"]["artifact_uri"])
    best_checkpoint = utils.get_best_checkpoint(artifact_uri=run_info["info"]["artifact_uri"])
    model = FinetunedLLM(
        llm=llm, dropout_p=params["train_loop_config"]["dropout_p"],
        embedding_dim=llm.embedding_dim, num_classes=num_classes)
    model = TorchCheckpoint.from_checkpoint(best_checkpoint).get_model(model)
    return model