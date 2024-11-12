# train.py
from abc import ABC

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanogpt_uniprot.data.tokenizers import Tokenizer


class BaseModelAbstract(ABC):
    def shared_step(self, batch): ...


class BaseModel(L.LightningModule, BaseModelAbstract):
    def __init__(self, network: nn.Module, loss_fn: nn.Module, lr: float):
        super().__init__()
        self.network = network
        self.loss_fn = loss_fn
        self.lr = lr
        self.save_hyperparameters()

        self.validation_outputs = None

    def shared_step(self, batch) -> torch.Tensor:
        # Send the batch through the network and calculate the loss
        # The Trainer will run .backward(), optimizer.step(), .zero_grad(), etc. for you
        # Assume y_hat are B x C x <Dimensions>
        x, y, mask = batch
        y_hat = self.network(x)
        loss = self.loss_fn(y_hat, y)
        return loss, y_hat

    def training_step(self, batch) -> torch.Tensor:
        loss, y_hat = self.shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": loss, "batch": batch, "y_hat": y_hat}

    def validation_step(self, batch) -> torch.Tensor:
        loss, y_hat = self.shared_step(batch)
        # detach the loss from the graph
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        self.validation_outputs = batch, y_hat

        return {"loss": loss, "batch": batch, "y_hat": y_hat}

    def configure_optimizers(self):
        return torch.optim.Adam(self.network.parameters(), lr=self.lr)


class Seq2SeqModel(BaseModel):
    # assumes the inputs and outputs are batches of sequences

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        block_size: int,
        temperature: float = 1.0,
        top_k: int = None,
    ) -> torch.Tensor:
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size

            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            # forward the model to get the logits for the index in the sequence
            logits = self.network(idx_cond)

            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def on_validation_epoch_end(self):
        # get the tokenizer from the dataloader
        tokenizer: Tokenizer = self.trainer.val_dataloaders.dataset.tokenizer
        block_size = self.trainer.val_dataloaders.dataset.block_size

        # save out example inputs, outputs, and targets
        batch, y_hat = self.validation_outputs

        # detokenize the inputs, outputs, and targets
        x, y = batch
        x_np = x.detach().cpu().numpy()[0]
        x_str = tokenizer.decode(x_np)

        y_np = y.detach().cpu().numpy()[0]
        y_str = tokenizer.decode(y_np)

        y_hat_np = torch.argmax(y_hat[0], -1).detach().cpu().numpy()
        y_hat_str = tokenizer.decode(y_hat_np)

        y_hat_gen = self.generate(x[[0]], max_new_tokens=len(y), block_size=block_size)
        y_hat_gen_np = y_hat_gen.detach().cpu().numpy()[0]
        y_hat_gen_str = tokenizer.decode(y_hat_gen_np)

        # input block
        input_block = f"#####\nInput\n#####\n: {x_str}\n\n"
        # target block
        target_block = f"#####\nTarget\n#####\n: {y_str}\n\n"
        # output block
        output_block = f"#####\nOutput\n#####\n: {y_hat_str}\n\n"
        # generated block
        generated_block = f"#####\nGenerated\n#####\n: {y_hat_gen_str}\n\n"

        out_text = "\n\n".join(
            [input_block, target_block, output_block, generated_block]
        )
        # save out as text files with iteration number in the filename
        with open(f"{self.logger.save_dir}/sample_{self.current_epoch}.txt", "w") as f:
            f.write(out_text)
