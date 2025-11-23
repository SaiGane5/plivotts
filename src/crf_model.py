import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
try:
    from torchcrf import CRF
except Exception:
    from TorchCRF import CRF


class TokenClassificationWithCRF(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        # torchcrf / TorchCRF APIs vary; avoid passing batch_first for compatibility
        try:
            self.crf = CRF(num_labels, batch_first=True)
        except TypeError:
            self.crf = CRF(num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Get token-level representations
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (batch, seq_len, hidden)
        emissions = self.classifier(sequence_output)  # (batch, seq_len, num_labels)

        mask = attention_mask.bool() if attention_mask is not None else None

        if labels is not None:
            # labels expected shape (batch, seq_len) with -100 for ignored tokens
            # CRF requires labels in [0, num_labels-1]; mask out -100
            label_mask = labels != -100
            # For CRF, we need labels where ignored positions have arbitrary value; we'll set to 0
            lab = labels.clone()
            lab[lab == -100] = 0
            # compute negative log likelihood
            # CRF APIs differ across versions: try calling with reduction first, else call and reduce manually
            try:
                log_likelihood = self.crf(emissions, lab, mask=label_mask, reduction="mean")
                loss = -log_likelihood
            except TypeError:
                # fallback: get per-batch log likelihoods and average
                log_likelihood = self.crf(emissions, lab, mask=label_mask)
                if isinstance(log_likelihood, torch.Tensor):
                    loss = -log_likelihood.mean()
                else:
                    # if not tensor, coerce
                    loss = -torch.tensor(log_likelihood, dtype=emissions.dtype, device=emissions.device).mean()
            # For compatibility with HuggingFace style, return a simple object
            out = type("obj", (), {})()
            out.loss = loss
            out.logits = emissions
            return out
        else:
            # decode best paths - CRF libraries differ in API names
            best_paths = None
            # Try common method names in order
            try:
                best_paths = self.crf.decode(emissions, mask=mask)
            except Exception:
                try:
                    # some CRF implementations expose `viterbi_tags`
                    vt = self.crf.viterbi_tags(emissions, mask=mask)
                    # viterbi_tags might return a list of (tags, score)
                    if isinstance(vt, list) and len(vt) > 0 and isinstance(vt[0], tuple):
                        best_paths = [tags for tags, score in vt]
                    else:
                        best_paths = vt
                except Exception:
                    # final fallback: use argmax over emissions
                    best_paths = emissions.argmax(dim=-1).cpu().tolist()

            out = type("obj", (), {})()
            out.predictions = best_paths
            out.logits = emissions
            return out

    def save_pretrained(self, save_directory: str):
        """Save model weights and config in a HuggingFace-compatible layout.

        This writes `pytorch_model.bin` and `config.json` to `save_directory`.
        """
        os.makedirs(save_directory, exist_ok=True)
        # save state dict
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        # save config
        try:
            # AutoConfig supports save_pretrained
            self.config.save_pretrained(save_directory)
        except Exception:
            # fallback: write json via config.to_json_string if available
            try:
                with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
                    f.write(self.config.to_json_string())
            except Exception:
                pass

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, num_labels: int = None, **kwargs):
        """Instantiate the CRF model from a pretrained backbone or a local directory.

        If `pretrained_model_name_or_path` is a directory containing a saved `pytorch_model.bin`,
        this will load the weights; otherwise the backbone is loaded from the model name.
        """
        # load config first
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

        # If a saved checkpoint exists, try to infer num_labels from the checkpoint
        bin_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        state_dict = None
        if os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
            if num_labels is None:
                # common keys to check
                if "classifier.weight" in state_dict:
                    num_labels = state_dict["classifier.weight"].shape[0]
                else:
                    # try to find any key that looks like classifier.weight
                    for k in state_dict.keys():
                        if k.endswith("classifier.weight"):
                            num_labels = state_dict[k].shape[0]
                            break
                    # try CRF params
                    if num_labels is None:
                        for k in state_dict.keys():
                            if k.endswith("crf.trans_matrix") or k.endswith("crf.transitions"):
                                num_labels = state_dict[k].shape[0]
                                break

        # fallback to config if still unknown
        if num_labels is None:
            num_labels = getattr(config, "num_labels", None)

        if num_labels is None:
            raise ValueError("num_labels could not be inferred; please provide num_labels")

        # instantiate model (this will load the backbone weights from the path or HF hub)
        model = cls(pretrained_model_name_or_path, num_labels=num_labels)

        # load state dict if we have one
        if state_dict is not None:
            try:
                model.load_state_dict(state_dict)
            except Exception:
                # partial load: allow strict=False to ignore param shape mismatches
                model.load_state_dict(state_dict, strict=False)

        return model
