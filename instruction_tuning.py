import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

class GraphInstructionTuning:
    """Graph instruction tuning for zero-shot transfer"""
    def __init__(
        self,
        base_model_name: str = "meta-llama/Llama-2-7b",
        graph_embedding_dim: int = 768,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load base LLM
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name)
        
        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj"]
        )
        self.model = get_peft_model(self.model, peft_config)
        
        # Projector for graph embeddings
        self.graph_projector = nn.Linear(graph_embedding_dim, self.model.config.hidden_size)
        
    def prepare_prompt(
        self,
        graph_embedding: torch.Tensor,
        instruction: str,
        task_type: str,
        label_space: Optional[List[str]] = None
    ) -> str:
        """
        Prepare prompt for instruction tuning
        
        Args:
            graph_embedding: Graph embedding from pre-trained model
            instruction: Natural language instruction
            task_type: Type of task (node/edge/graph)
            label_space: List of possible labels for classification
            
        Returns:
            Formatted prompt string
        """
        # Project graph embedding to LLM's embedding space
        projected_emb = self.graph_projector(graph_embedding)
        
        # Format prompt based on task type
        if task_type == "node":
            prompt = f"Given the node embedding: {projected_emb}\n"
        elif task_type == "edge":
            prompt = f"Given the edge embedding: {projected_emb}\n"
        else:  # graph
            prompt = f"Given the graph embedding: {projected_emb}\n"
            
        prompt += f"Instruction: {instruction}\n"
        
        if label_space:
            prompt += f"Possible labels: {', '.join(label_space)}\n"
            
        prompt += "Answer:"
        return prompt
        
    def train_step(
        self,
        graph_embedding: torch.Tensor,
        instruction: str,
        target: str,
        task_type: str,
        label_space: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Perform one training step
        
        Args:
            graph_embedding: Graph embedding from pre-trained model
            instruction: Natural language instruction
            target: Target label/answer
            task_type: Type of task
            label_space: List of possible labels
            
        Returns:
            Tuple of (loss, metrics)
        """
        # Prepare input
        prompt = self.prepare_prompt(graph_embedding, instruction, task_type, label_space)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        labels = self.tokenizer(target, return_tensors="pt").to(self.device)
        
        # Forward pass
        outputs = self.model(**inputs, labels=labels["input_ids"])
        loss = outputs.loss
        
        # Calculate metrics
        metrics = {
            "loss": loss.item(),
            "perplexity": torch.exp(loss).item()
        }
        
        return loss, metrics
        
    def predict(
        self,
        graph_embedding: torch.Tensor,
        instruction: str,
        task_type: str,
        label_space: Optional[List[str]] = None,
        max_length: int = 100
    ) -> str:
        """
        Make prediction using instruction-tuned model
        
        Args:
            graph_embedding: Graph embedding from pre-trained model
            instruction: Natural language instruction
            task_type: Type of task
            label_space: List of possible labels
            max_length: Maximum length of generated text
            
        Returns:
            Generated prediction
        """
        # Prepare input
        prompt = self.prepare_prompt(graph_embedding, instruction, task_type, label_space)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate prediction
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
            
        prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return prediction.split("Answer:")[-1].strip() 