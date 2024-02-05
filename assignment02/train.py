import os
import torch
from tqdm import tqdm


from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq

from contextlib import nullcontext

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, SequentialSampler


from lora_model import LoraModelForCasualLM
from utils.common import download_from_driver
from prepare_data import create_datasets

import warnings
warnings.filterwarnings('ignore')
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True


class Trainer:
    def __init__(self,
                 model,
                 tokenizer,
                 gpu_id: int,
                 is_ddp_training: bool = True,
                 output_dir: str = 'checkpoints/',
                 num_epochs: int = 10,
                 max_length: int = 128,
                 batch_size: int = 8,
                 mixed_precision_dtype=None,
                 gradient_accumulation_steps: int = 16):
        """
        Initialize the Trainer class.

        Args:
            model: Pretrained model object.
            tokenizer: Tokenizer object for text processing.
            num_epochs: Number of training epochs.
            max_length: Maximum sequence length.
            batch_size: Training batch size.
            gpu_id: GPU ID for training.
        """

        self.num_epochs = num_epochs
        self.max_length = max_length
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.is_ddp_training = is_ddp_training

        self.gpu_id = gpu_id
        self.model = model.to(f"cuda:{self.gpu_id}")
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.mixed_precision_dtype = mixed_precision_dtype
        self.ctx = None
        self.gradscaler = None

        # set mixed precision context
        self.set_mixed_precision_context(mixed_precision_dtype)

    def set_mixed_precision_context(self, mixed_precision_dtype):
        
        # TODO: Setup mixed precision training context

        if mixed_precision_dtype is None:
            
            # If 'mixed_precision_dtype' is None, use 'nullcontext',
            self.ctx = None

        else:
        
            # TODO Otherwise, use 'torch.amp.autocast' context with the specified dtype, and initialize GradScaler if mixed_precision_dtype is float16.
            self.ctx = None
            self.gradscaler = None

    def _set_ddp_training(self):

        # TODO: Initialize the DistributedDataParallel wrapper for the model.
        # You would need to pass the model and specify the device IDs
        # and output device for the data parallelism.

        ### YOUR CODE HERE ###

        self.model = None

    def _run_batch(self, batch):
        """
        Run a single training batch.

        Args:
            batch: Batch data.

        Returns:
            Loss value for the batch.
        """

        with self.ctx:
            outputs = self.model(**batch)
            loss = outputs.loss / self.gradient_accumulation_steps  # Normalize loss
        loss_val = loss.item()

        # TODO: If 'mixed_precision_dtype' is torch.float16, you have to modify the backward using the gradscaler.
        if self.mixed_precision_dtype == torch.float16:

            ### YOUR CODE HERE ###
            
            pass
        else:
            loss.backward()

        return loss_val

    def _run_epoch(self, train_dataloader, epoch):
        """
        Run a single training epoch.

        Args:
            train_loader: Training data loader.
            epoch: Current epoch number.

        Returns:
            Total loss value for the epoch.
        """

        epoch_loss = 0
        self.model.train()

        if _is_master_process():
            train_progress_bar = tqdm(
                train_dataloader, desc=f"Epoch {epoch + 1} [Training]", position=0, leave=False)
        else:
            train_progress_bar = train_dataloader

        # Add counter for gradient accumulation
        steps = 0
        self.optimizer.zero_grad()  # Reset gradients at the beginning of each epoch
        for step, batch in enumerate(train_progress_bar):
            steps += 1
            batch = {key: value.to(self.gpu_id)
                     for key, value in batch.items()}
            loss = self._run_batch(batch)
            epoch_loss += loss

            # Perform optimizer step and reset gradients after accumulating enough gradients
            if steps % self.gradient_accumulation_steps == 0:

                # If 'mixed_precision_dtype' is torch.float16, you have to modify the gradient update step using the gradscaler.
                if self.mixed_precision_dtype == torch.float16:

                    ### YOUR CODE HERE ###
                    
                    # TODO: optimizer step

                    # TODO: update scaler factor

                    pass
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

                torch.cuda.empty_cache()
        epoch_loss /= (len(train_dataloader) /
                       self.gradient_accumulation_steps)
        return epoch_loss

    def _save_checkpoint(self, epoch):
        path_dir = f"{self.output_dir}/epoch_{epoch}"

        # check path_dir exited
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)

        # save checkpoints
        if self.is_ddp_training and _is_master_process():
            self.model.module.save_pretrained(f'epoch_{epoch}_checkpoint')
        else:
            self.model.save_pretrained(f'epoch_{epoch}_checkpoint')

        print("Done saved at", f'epoch_{epoch}_checkpoint')

    def prepare_dataloader(self, train_dataset, eval_dataset):

        # TODO: Prepare the training DataLoader. Initialize 'DataLoader' with 'train_dataset'
        # and the appropriate 'batch_size'.
        # Depending on whether the training is distributed (is_ddp_training),
        # use 'DistributedSampler' for 'sampler' argument, else use 'None'.
        # Use 'DataCollatorForSeq2Seq' for 'collate_fn', passing 'tokenizer', padding settings and pad_to_multiple_of to 8, and return_tensors="pt"
        # Also add drop_last to True.

        ### YOUR CODE HERE ###

        data_trainloader = None

        # TODO: Prepare the evaluation DataLoader. Initialize 'DataLoader' with 'eval_dataset',
        # the appropriate 'batch_size', and 'SequentialSampler' for 'sampler'.
        # Use 'DataCollatorForSeq2Seq' for 'collate_fn', passing 'tokenizer', padding settings and pad_to_multiple_of to 8, and return_tensors="pt".
        # Also add drop_last to True.

        ### YOUR CODE HERE ###

        data_testloader = None

        return data_trainloader, data_testloader

    def _eval(self, eval_dataloader, epoch: int):
        avg_loss = 0
        model.eval()
        if _is_master_process():
            eval_progress_bar = tqdm(
                eval_dataloader, desc=f"Epoch {epoch + 1} [Evaluation]", position=0, leave=False)
        else:
            eval_progress_bar = eval_dataloader


        for batch in eval_progress_bar:
            with self.ctx:
                with torch.no_grad():
                    if not self.is_ddp_training:
                        outputs = self.model(**batch.to(self.gpu_id))
                    else:
                        outputs = self.model(**batch)
            avg_loss += outputs.loss.item()
        avg_loss = avg_loss/(len(eval_dataloader))
        return avg_loss

    def run(self, data_path: str, size_valid_set: int = 0.25, seed: int = 123):
        """
        Run the training process.

        Returns:
            None
        """
        train_dataset, eval_dataset = create_datasets(
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            data_path=data_path,
            size_valid_set=size_valid_set,
            seed=seed
        )

        train_dataloader, eval_dataloader = self.prepare_dataloader(
            train_dataset, eval_dataset)

        if self.is_ddp_training:
            self._set_ddp_training()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate)

        for epoch in range(self.num_epochs):

            if self.is_ddp_training:
                train_dataloader.sampler.set_epoch(epoch)

            train_loss = self._run_epoch(train_dataloader, epoch)
            if self.is_ddp_training:
                dist.barrier() 
            if _is_master_process() or (epoch == self.num_epochs - 1):
                eval_loss = self._eval(
                    eval_dataloader=eval_dataloader, epoch=epoch)

                print(
                    f"epoch = {epoch+1} | avg_train_loss = {train_loss} | eval_loss = {eval_loss}")
            
            if _is_master_process():
                self._save_checkpoint(epoch=epoch+1)

def load_tokenizer_from_pretrained_model(model_path):

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    architecture = config.architectures[0]
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, device_map={"": torch.device(f"cuda:{0}")})
    tokenizer.pad_token = tokenizer.eos_token
    if _is_master_process():
        print('Completed to load config & tokenizer')

    if "Llama" in architecture:
        if _is_master_process():
            print("Setting EOS, BOS, UNK, and PAD tokens for LLama tokenizer")
        tokenizer.add_special_tokens(
            {
                "eos_token": "</s>",
                "bos_token": "</s>",
                "unk_token": "</s>",
            }
        )
        tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )

    return tokenizer


def _is_master_process():
    ddp_rank = int(os.environ['RANK'])
    return ddp_rank == 0


def load_pretrained_model(local_rank, model_path: str = ""):
    # TODO: Load a pretrained AutoModelForCausalLM from the 'model_path'.
    # Make sure to set 'device_map' to '{"": torch.device(f"cuda:{local_rank}")}' for DDP training
    # and trust_remote_code=True.

    ### YOUR CODE HERE ###

    model = None 

    # TODO: Create a LoraConfig with the parameters: 
    # r=4, lora_alpha=8, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=['lm_head.linear', 'transformer.embd.wte'].
    # We will then use the config to initialize a LoraModelForCasualLM with the loaded model.

    ### YOUR CODE HERE ###

    lora_config = None 

    # TODO: Create LoRA model

    model = None  # Apply current model to Lora Model

    if _is_master_process():
        model.print_trainable_parameters()

    return model


if __name__ == "__main__":
    OUTPUT_DIR = "checkpoints/"

    backend = "nccl"
    model_path = 'TheBloke/phi-2-GPTQ'
    if os.environ.get("DEBUG"):
        data_path = "test_data.json"
    else:
        data_path = 'alpaca_data.json'

    size_valid_set = 0.15
    max_length = 128
    num_epochs = 3
    batch_size = 2
    gradient_accumulation_steps = 8

    learning_rate = 3e-4
    lr_scheduler_type = 'cosine'
    num_warmup_steps = 100
    weight_decay = 0.06

    seed = 0
    log_freq = 1
    eval_freq = 150

    distributed_strategy = "ddp" if os.environ.get("ON_DDP") else "no"

    if distributed_strategy == "ddp":

        # TODO: Initialize the process group for distributed data parallelism with nccl backend.
        # After that, you should set the 'local_rank' from the environment variable 'LOCAL_RANK'.

        # Initialize the process group

        ### YOUR CODE HERE ###
        
        pass
    else:
        os.environ['RANK'] = '0'
        local_rank = 0

    # Prepare model
    model = load_pretrained_model(local_rank, model_path=model_path)
    
    # Get tokenizer
    tokenizer = load_tokenizer_from_pretrained_model(model_path=model_path)

    # prepare trainer
    trainer = Trainer(
        model=model,
        num_epochs=num_epochs,
        max_length=max_length,
        batch_size=batch_size,
        gpu_id=local_rank,
        
        mixed_precision_dtype=torch.float16 if os.environ.get("ON_MP") else None,
        
        tokenizer=tokenizer,
        output_dir=OUTPUT_DIR,
        is_ddp_training=True if distributed_strategy == "ddp" else False,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    # set ddp for wraping model
    # execute trainer
    trainer.run(
        data_path=data_path,
        size_valid_set=size_valid_set,
        seed=seed
    )

    if distributed_strategy == "ddp":
        destroy_process_group()