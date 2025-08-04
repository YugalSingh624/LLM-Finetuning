# Complete Guide to LLM Fine-Tuning: From Fundamentals to Advanced Techniques

## 📚 Table of Contents
1. [Fundamentals of LLM Fine-Tuning](#fundamentals)
2. [Mathematical Foundations](#mathematics)
3. [Fine-Tuning Methodologies](#methodologies)
4. [Parameter-Efficient Fine-Tuning (PEFT)](#peft)
5. [Advanced Techniques](#advanced)
6. [Training Strategies](#training)
7. [Evaluation and Benchmarking](#evaluation)
8. [Production Considerations](#production)
9. [Recent Research and Future Directions](#research)
10. [Practical Implementation](#implementation)

---

## 📖 1. Fundamentals of LLM Fine-Tuning {#fundamentals}

### What is LLM Fine-Tuning?

Fine-tuning is the process of taking a pre-trained language model and adapting it to a specific task, domain, or dataset. It's a form of **transfer learning** where we leverage the knowledge already captured during pre-training and specialize it for our needs.

### Why Fine-Tune Instead of Training from Scratch?

1. **Cost Efficiency**: Pre-training costs millions of dollars; fine-tuning costs hundreds
2. **Data Efficiency**: Requires much less data (thousands vs trillions of tokens)
3. **Time Efficiency**: Hours/days instead of months
4. **Knowledge Transfer**: Leverages existing linguistic and world knowledge
5. **Resource Accessibility**: Possible on consumer hardware with techniques like LoRA

### Types of Fine-Tuning

#### **1. Supervised Fine-Tuning (SFT)**
- Train on input-output pairs
- Most common approach for task-specific adaptation
- Examples: Question-answering, text classification, instruction following

#### **2. Reinforcement Learning from Human Feedback (RLHF)**
- Uses human preferences to align model behavior
- Two-stage process: reward model training + PPO optimization
- Critical for safety and alignment (ChatGPT, Claude)

#### **3. Constitutional AI**
- Uses a set of principles/constitution to guide behavior
- Self-supervised approach to alignment
- Reduces need for extensive human feedback

#### **4. Multi-Task Fine-Tuning**
- Train on multiple tasks simultaneously
- Improves generalization and reduces catastrophic forgetting
- Examples: T5, FLAN, InstructGPT

---

## 🔢 2. Mathematical Foundations {#mathematics}

### Pre-Training Objective

Most LLMs are trained with **autoregressive language modeling**:

```
L_pretrain = -∑(i=1 to n) log P(x_i | x_1, x_2, ..., x_(i-1); θ)
```

Where:
- `x_i` is the i-th token in the sequence
- `θ` represents model parameters
- The model learns to predict the next token given previous context

### Fine-Tuning Objective

For supervised fine-tuning, we modify the objective:

```
L_finetune = -∑(i=1 to m) log P(y_i | x_i; θ_fine)
```

Where:
- `(x_i, y_i)` are input-output pairs in the fine-tuning dataset
- `θ_fine` are the updated parameters after fine-tuning
- `m` is the number of training examples

### Gradient Computation

During fine-tuning, we compute gradients:

```
∇θ L_finetune = ∑(i=1 to m) ∇θ log P(y_i | x_i; θ)
```

The parameters are updated using optimizers like Adam or AdamW:

```
θ_(t+1) = θ_t - α * ∇θ L_finetune
```

Where `α` is the learning rate.

---

## 🛠️ 3. Fine-Tuning Methodologies {#methodologies}

### Full Fine-Tuning

**Concept**: Update all model parameters during training.

**Advantages**:
- Maximum flexibility and adaptation capability
- Can dramatically change model behavior
- Best performance for domain-specific tasks

**Disadvantages**:
- Requires enormous GPU memory (100GB+ for 70B models)
- Risk of catastrophic forgetting
- Expensive and time-consuming
- Need large datasets to avoid overfitting

**When to Use**:
- You have massive computational resources
- Large, high-quality dataset available
- Domain is very different from pre-training data
- Maximum performance is critical

### Instruction Tuning

**Concept**: Fine-tune on instruction-following datasets where inputs are natural language instructions.

**Dataset Format**:
```json
{
  "instruction": "Translate the following English text to French:",
  "input": "Hello, how are you?",
  "output": "Bonjour, comment allez-vous?"
}
```

**Popular Datasets**:
- **Alpaca**: 52K instruction-following examples
- **FLAN**: Multi-task instruction tuning dataset
- **OpenAssistant**: Human-generated conversations
- **ShareGPT**: Conversations with ChatGPT

**Training Process**:
1. Format data with special tokens (e.g., `### Instruction:`, `### Response:`)
2. Train model to generate appropriate responses
3. Use teacher forcing during training
4. Evaluate on held-out instruction-following tasks

### Domain Adaptation

**Concept**: Adapt a general-purpose model to a specific domain (medical, legal, scientific).

**Strategies**:
1. **Continue Pre-training**: Further pre-train on domain-specific text
2. **Domain-Specific SFT**: Fine-tune on domain instruction data
3. **Hybrid Approach**: Combine both strategies

**Example Domains**:
- **Medical**: PubMed articles, clinical notes, medical Q&A
- **Legal**: Legal documents, case law, legal reasoning
- **Code**: Programming repositories, code documentation
- **Scientific**: Research papers, technical documentation

---

## ⚡ 4. Parameter-Efficient Fine-Tuning (PEFT) {#peft}

### Why PEFT?

Traditional fine-tuning is resource-intensive. PEFT methods solve this by:
- Reducing trainable parameters by 99%+
- Maintaining competitive performance
- Enabling fine-tuning on consumer hardware
- Allowing multiple task-specific adapters

### LoRA (Low-Rank Adaptation)

**Core Idea**: Decompose weight updates into low-rank matrices.

**Mathematical Formulation**:
```
h = W₀x + ΔWx = W₀x + BAx
```

Where:
- `W₀`: Frozen pre-trained weights (d × k)
- `A`: Trainable matrix (d × r)
- `B`: Trainable matrix (r × k)
- `r`: Rank (r << min(d,k))

**Key Parameters**:
- **Rank (r)**: Controls adaptation capacity (typically 8, 16, 32, 64)
- **Alpha**: Scaling factor (often equals rank)
- **Target Modules**: Which layers to adapt (attention, MLP, embeddings)
- **Dropout**: Regularization for LoRA layers

**Implementation**:
```python
# LoRA layer implementation
class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank, alpha, dropout=0.1):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(original_layer.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, original_layer.out_features))
        self.dropout = nn.Dropout(dropout)
        
        # Initialize A with normal distribution, B with zeros
        nn.init.normal_(self.lora_A, std=0.02)
        
    def forward(self, x):
        original_output = self.original_layer(x)
        lora_output = self.dropout(x) @ self.lora_A @ self.lora_B
        return original_output + (self.alpha / self.rank) * lora_output
```

**Advantages**:
- 99%+ parameter reduction
- Fast training and inference
- Multiple adapters can be stored and swapped
- No increase in inference latency when merged

**Disadvantages**:
- Limited expressiveness compared to full fine-tuning
- Rank selection requires experimentation
- May not work well for very different domains

### QLoRA (Quantized LoRA)

**Innovation**: Combines 4-bit quantization with LoRA for even greater efficiency.

**Key Components**:
1. **4-bit Normal Float (NF4)**: Optimized quantization for normally distributed weights
2. **Double Quantization**: Quantize the quantization constants
3. **Paged Optimizers**: Handle memory spikes during training

**Memory Savings**:
- Base model: 4-bit quantized (4x reduction)
- Gradients: Only for LoRA parameters
- Optimizer states: Only for trainable parameters
- Total: ~16x memory reduction vs full fine-tuning

**Performance**: Matches 16-bit LoRA performance with much less memory.

### AdaLoRA (Adaptive LoRA)

**Innovation**: Dynamically allocates rank to different modules based on importance.

**Key Features**:
- **Importance Scoring**: Uses gradient-based metrics to score parameter importance
- **Rank Pruning**: Removes less important rank dimensions
- **Rank Growing**: Adds rank to important modules
- **Orthogonal Regularization**: Maintains orthogonality of low-rank matrices

**Algorithm**:
1. Start with uniform rank allocation
2. Compute importance scores during training
3. Prune low-importance ranks
4. Redistribute rank budget to important modules
5. Add orthogonal regularization loss

### Other PEFT Methods

#### **Prefix Tuning**
- Add trainable prefix tokens to each layer
- Prefix vectors are prepended to key and value matrices
- Good for generation tasks

#### **P-Tuning v2**
- Learnable prompt tokens for each layer
- More flexible than prefix tuning
- Effective for understanding tasks

#### **IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)**
- Element-wise scaling of activations
- Even fewer parameters than LoRA
- Good for maintaining pre-trained knowledge

#### **Adapters**
- Small bottleneck layers inserted between transformer blocks
- Sequential processing (increases inference time)
- Early PEFT method, less popular now

---

## 🧠 5. Advanced Techniques {#advanced}

### Multi-Task Learning

**Concept**: Train on multiple tasks simultaneously to improve generalization.

**Benefits**:
- Reduces catastrophic forgetting
- Improves zero-shot performance on new tasks
- Better parameter utilization
- Robustness to distribution shift

**Challenges**:
- Task interference (negative transfer)
- Balancing task weights
- Scalability to many tasks
- Dataset formatting consistency

**Strategies**:
1. **Task Mixing**: Random sampling from different task datasets
2. **Curriculum Learning**: Order tasks by difficulty or relatedness
3. **Task Weighting**: Adjust loss weights for different tasks
4. **Task-Specific Heads**: Different output layers for different tasks

### Mixture of Experts (MoE) Fine-Tuning

**Concept**: Add expert modules that specialize in different aspects of the task.

**Architecture**:
```
Output = ∑(i=1 to n) G(x)ᵢ * Eᵢ(x)
```

Where:
- `G(x)`: Gating network (decides which experts to use)
- `Eᵢ(x)`: i-th expert network
- `n`: Number of experts

**Benefits**:
- Increased model capacity without proportional compute increase
- Specialization for different input types
- Better scaling properties

### Retrieval-Augmented Generation (RAG) Fine-Tuning

**Concept**: Combine retrieval with generation for knowledge-intensive tasks.

**Components**:
1. **Retriever**: Finds relevant documents for a query
2. **Generator**: Produces output conditioned on query and retrieved docs
3. **Knowledge Base**: External corpus of documents

**Training**:
- Can fine-tune retriever, generator, or both
- Joint training often works better than separate training
- Requires careful balancing of retrieval and generation losses

### Constitutional AI Fine-Tuning

**Process**:
1. **Initial SFT**: Train on human-written examples
2. **AI Feedback**: Model critiques its own outputs using principles
3. **Revision**: Model revises outputs based on critiques
4. **Final Training**: Train on revised, improved outputs

**Benefits**:
- Scalable approach to alignment
- Reduces need for human feedback
- Improves consistency in behavior
- Better at following complex instructions

### Meta-Learning for Few-Shot Fine-Tuning

**Concept**: Train models to quickly adapt to new tasks with minimal examples.

**Methods**:
1. **MAML (Model-Agnostic Meta-Learning)**: Learn initialization that fine-tunes quickly
2. **Reptile**: Simplified version of MAML
3. **In-Context Learning**: Use examples as context rather than gradients

**Applications**:
- Rapid adaptation to new domains
- Personalization with minimal user data
- Few-shot learning for specialized tasks

---

## 🎯 6. Training Strategies {#training}

### Learning Rate Scheduling

**Importance**: Learning rate is the most critical hyperparameter in fine-tuning.

#### **Linear Warmup + Cosine Decay**
```python
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)
```

**Why This Works**:
- Warmup prevents early training instability
- Cosine decay provides smooth convergence
- Widely used in successful models (BERT, GPT, etc.)

#### **Learning Rate Guidelines**:
- **Full Fine-tuning**: 1e-5 to 5e-5 (much lower than pre-training)
- **LoRA**: 1e-4 to 5e-4 (higher because fewer parameters)
- **Instruction Tuning**: 2e-5 to 1e-4
- **Domain Adaptation**: 1e-5 to 3e-5

### Gradient Techniques

#### **Gradient Clipping**
```python
# Prevent gradient explosion
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

#### **Gradient Accumulation**
```python
# Simulate larger batch sizes
loss = loss / accumulation_steps
loss.backward()

if (step + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

#### **Gradient Checkpointing**
- Trade computation for memory
- Recompute activations during backward pass
- Essential for large models on limited hardware

### Memory Optimization

#### **Mixed Precision Training**
```python
# Use automatic mixed precision
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

#### **DeepSpeed ZeRO**
- **ZeRO-1**: Optimizer state partitioning
- **ZeRO-2**: Gradient partitioning
- **ZeRO-3**: Parameter partitioning
- **ZeRO-Infinity**: CPU/NVMe offloading

#### **Activation Checkpointing**
- Store only some activations during forward pass
- Recompute others during backward pass
- Significant memory savings for minimal compute overhead

### Regularization Techniques

#### **Dropout**
- Standard dropout on hidden layers
- LoRA-specific dropout on adapter layers
- Attention dropout for transformer models

#### **Weight Decay**
- L2 regularization on model parameters
- Prevents overfitting on small datasets
- Typical values: 0.01 to 0.1

#### **Early Stopping**
- Monitor validation metrics
- Stop training when performance plateaus
- Prevents overfitting and saves compute

#### **Data Augmentation**
- Paraphrasing for text tasks
- Back-translation for multilingual tasks
- Noise injection for robustness

---

## 📊 7. Evaluation and Benchmarking {#evaluation}

### Standard Benchmarks

#### **General Language Understanding**
- **GLUE**: General Language Understanding Evaluation
- **SuperGLUE**: More challenging version of GLUE
- **MMLU**: Massive Multitask Language Understanding (57 subjects)
- **BIG-bench**: Beyond the Imitation Game benchmark

#### **Reasoning and Knowledge**
- **ARC**: AI2 Reasoning Challenge (science questions)
- **HellaSwag**: Commonsense reasoning
- **PIQA**: Physical Interaction QA
- **WinoGrande**: Winograd schema challenge

#### **Code Generation**
- **HumanEval**: Python programming problems
- **MBPP**: Mostly Basic Python Problems
- **CodeXGLUE**: Code understanding and generation

#### **Mathematical Reasoning**
- **GSM8K**: Grade school math word problems
- **MATH**: Competition-level mathematics
- **DropMath**: Mathematical reasoning with dropout

#### **Instruction Following**
- **Alpaca Eval**: Evaluation on Alpaca test set
- **Self-Instruct**: Self-generated instruction following
- **Vicuna Bench**: Conversational evaluation

### Evaluation Methodologies

#### **Automatic Metrics**
```python
# Accuracy for classification
accuracy = correct_predictions / total_predictions

# BLEU for generation
from nltk.translate.bleu_score import sentence_bleu
bleu_score = sentence_bleu([reference], candidate)

# ROUGE for summarization
from rouge import Rouge
rouge = Rouge()
scores = rouge.get_scores(candidate, reference)

# Perplexity for language modeling
perplexity = torch.exp(loss)
```

#### **Human Evaluation**
- **Likert Scale Ratings**: Rate outputs on 1-5 scale
- **Pairwise Comparisons**: Choose better of two outputs
- **Task-Specific Metrics**: Correctness, helpfulness, safety
- **Inter-Annotator Agreement**: Ensure consistent evaluation

#### **LLM-as-Judge**
```python
# Use strong model to evaluate weaker model outputs
def llm_judge(question, answer_a, answer_b):
    prompt = f"""
    Question: {question}
    Answer A: {answer_a}
    Answer B: {answer_b}
    
    Which answer is better? Consider accuracy, helpfulness, and clarity.
    Choose: A or B
    """
    return llm_model.generate(prompt)
```

### Evaluation Best Practices

#### **Proper Train/Val/Test Splits**
- Never evaluate on training data
- Use held-out validation for hyperparameter tuning
- Report final results on unseen test set only

#### **Statistical Significance**
```python
# Bootstrap confidence intervals
def bootstrap_confidence_interval(scores, n_bootstrap=1000, alpha=0.05):
    bootstrap_scores = []
    n = len(scores)
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=n, replace=True)
        bootstrap_scores.append(np.mean(sample))
    
    lower = np.percentile(bootstrap_scores, 100 * alpha / 2)
    upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))
    return lower, upper
```

#### **Multiple Runs**
- Run experiments with different random seeds
- Report mean and standard deviation
- Ensure reproducibility with fixed seeds

#### **Fair Comparison**
- Use same evaluation prompts/formats
- Control for model size and training data
- Report computational costs (FLOPs, GPU hours)

---

## 🏭 8. Production Considerations {#production}

### Model Serving

#### **Inference Optimization**
- **Quantization**: INT8, INT4 for faster inference
- **Pruning**: Remove less important parameters
- **Knowledge Distillation**: Train smaller student model
- **Dynamic Batching**: Batch requests for efficiency

#### **Serving Frameworks**
```python
# vLLM for high-throughput serving
from vllm import LLM, SamplingParams

llm = LLM(model="your-fine-tuned-model")
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(prompts, sampling_params)

# TensorRT-LLM for NVIDIA GPUs
# Optimized kernels for transformer inference

# Text Generation Inference (TGI) by Hugging Face
# Docker-based serving with auto-batching
```

#### **Model Deployment Patterns**
1. **Single Model Serving**: Deploy one fine-tuned model
2. **Multi-Model Serving**: Multiple task-specific models
3. **Adapter Swapping**: Base model + swappable LoRA adapters
4. **Ensemble Methods**: Combine multiple model predictions

### Monitoring and Maintenance

#### **Performance Monitoring**
```python
# Track key metrics in production
class ModelMonitor:
    def __init__(self):
        self.metrics = {
            'latency': [],
            'throughput': [],
            'error_rate': [],
            'quality_score': []
        }
    
    def log_request(self, latency, quality_score, error=False):
        self.metrics['latency'].append(latency)
        self.metrics['error_rate'].append(1 if error else 0)
        self.metrics['quality_score'].append(quality_score)
    
    def get_stats(self):
        return {
            'avg_latency': np.mean(self.metrics['latency']),
            'error_rate': np.mean(self.metrics['error_rate']),
            'avg_quality': np.mean(self.metrics['quality_score'])
        }
```

#### **Model Drift Detection**
- Monitor input distribution changes
- Track output quality over time
- Set up alerts for performance degradation
- Implement automatic retraining triggers

#### **A/B Testing**
```python
# Split traffic between models
def route_request(user_id, models):
    if hash(user_id) % 100 < 50:  # 50% traffic
        return models['model_a']
    else:
        return models['model_b']
```

### Safety and Alignment

#### **Content Filtering**
```python
# Input filtering
def filter_harmful_input(text):
    harmful_patterns = [
        r'how to make.*bomb',
        r'illegal.*drugs',
        # Add more patterns
    ]
    
    for pattern in harmful_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

# Output filtering
def filter_harmful_output(text):
    # Use classification model for toxicity detection
    toxicity_score = toxicity_classifier(text)
    return toxicity_score > 0.8
```

#### **Bias Detection**
- Test on diverse demographic groups
- Monitor for unfair treatment
- Use bias evaluation datasets
- Implement fairness constraints

#### **Robustness Testing**
- Adversarial examples
- Out-of-distribution inputs
- Edge cases and corner cases
- Stress testing with high load

### Cost Optimization

#### **Compute Efficiency**
- **Model Compression**: Reduce model size without losing performance
- **Caching**: Store frequent query results
- **Load Balancing**: Distribute requests across replicas
- **Auto-scaling**: Scale resources based on demand

#### **Cost Monitoring**
```python
# Track inference costs
class CostTracker:
    def __init__(self, cost_per_token=0.0001):
        self.cost_per_token = cost_per_token
        self.total_tokens = 0
        self.total_requests = 0
    
    def log_request(self, input_tokens, output_tokens):
        self.total_tokens += input_tokens + output_tokens
        self.total_requests += 1
    
    def get_costs(self):
        return {
            'total_cost': self.total_tokens * self.cost_per_token,
            'cost_per_request': (self.total_tokens * self.cost_per_token) / self.total_requests,
            'average_tokens_per_request': self.total_tokens / self.total_requests
        }
```

---

## 🔬 9. Recent Research and Future Directions {#research}

### Scaling Laws for Fine-Tuning

Recent research has identified scaling laws specific to fine-tuning:

#### **Data Scaling**
- Performance improves with more fine-tuning data, but with diminishing returns
- Quality matters more than quantity for instruction tuning
- Optimal dataset size depends on task complexity and model size

#### **Parameter Scaling**
- Larger models generally fine-tune better
- But smaller models can match larger ones on specific tasks with good fine-tuning
- Parameter-efficient methods scale better than full fine-tuning

#### **Compute Scaling**
- More training compute generally helps, but returns diminish quickly
- Early stopping often optimal for generalization
- Compute better spent on data quality than training longer

### Emerging Techniques

#### **In-Context Learning Enhancement**
- Fine-tune models to be better few-shot learners
- Meta-learning approaches for rapid adaptation
- Instruction tuning improves in-context learning ability

#### **Tool-Using Models**
```python
# Fine-tune models to use external tools
def tool_augmented_generation(query, tools):
    # Model decides which tool to use
    tool_choice = model.select_tool(query, available_tools=tools)
    
    # Execute tool and get result
    tool_result = tools[tool_choice].execute(query)
    
    # Generate final response using tool result
    response = model.generate(query, tool_result=tool_result)
    return response
```

#### **Multimodal Fine-Tuning**
- Extend text models to handle images, audio, video
- Vision-language models (CLIP, DALLE-2, GPT-4V)
- Cross-modal transfer learning

#### **Federated Fine-Tuning**
- Fine-tune models across distributed devices
- Privacy-preserving techniques
- Challenges: heterogeneous data, communication costs

### Research Frontiers

#### **Theoretical Understanding**
- Why does fine-tuning work so well?
- What knowledge is transferred vs. learned anew?
- Optimal fine-tuning strategies for different scenarios

#### **Efficiency Improvements**
- Even more parameter-efficient methods
- Faster training algorithms
- Better hardware utilization

#### **Alignment and Safety**
- Scalable oversight techniques
- Constitutional AI improvements
- Robustness to adversarial attacks

#### **Personalization**
- User-specific fine-tuning
- Continual learning without forgetting
- Privacy-preserving personalization

---

## 💻 10. Practical Implementation {#implementation}

### Complete Fine-Tuning Pipeline

#### **Step 1: Environment Setup**
```bash
# Install required packages
pip install torch transformers datasets accelerate peft
pip install bitsandbytes  # For quantization
pip install deepspeed     # For large model training
pip install wandb         # For experiment tracking
```

#### **Step 2: Data Preparation**
```python
import json
from datasets import Dataset

# Load and format data
def format_instruction_data(examples):
    prompts = []
    for instruction, input_text, output in zip(
        examples['instruction'], examples['input'], examples['output']
    ):
        prompt = f"### Instruction:\n{instruction}\n"
        if input_text:
            prompt += f"### Input:\n{input_text}\n"
        prompt += f"### Response:\n{output}"
        prompts.append(prompt)
    return {"text": prompts}

# Create dataset
dataset = Dataset.from_json("alpaca_data.json")
dataset = dataset.map(format_instruction_data, batched=True)
```

#### **Step 3: Model and Tokenizer Setup**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import torch

# Load model and tokenizer
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Add padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                   "gate_proj", "up_proj", "down_proj"]
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

#### **Step 4: Training Setup**
```python
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    evaluation_strategy="steps",
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to="wandb",
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)
```

#### **Step 5: Training and Evaluation**
```python
# Train the model
trainer.train()

# Save the model
model.save_pretrained("./fine-tuned-model")
tokenizer.save_pretrained("./fine-tuned-model")

# Evaluation
def evaluate_model(model, tokenizer, test_prompts):
    model.eval()
    results = []
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append(response)
    
    return results
```

### Advanced Training Configurations

#### **Multi-GPU Training with DeepSpeed**
```python
# deepspeed_config.json
{
    "train_batch_size": 32,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 2,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu"
        }
    }
}

# Launch training
# deepspeed --num_gpus=4 train.py --deepspeed deepspeed_config.json
```

#### **Gradient Checkpointing for Memory Efficiency**
```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Custom checkpointing function
def custom_checkpoint_function(forward_function, *args):
    # Define which layers to checkpoint
    return torch.utils.checkpoint.checkpoint(forward_function, *args, use_reentrant=False)
```

#### **Custom Learning Rate Scheduling**
```python
from transformers import get_scheduler

# Create custom scheduler
scheduler = get_scheduler(
    name="cosine_with_restarts",
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=10000,
    num_cycles=2
)

# Use in training loop
for step, batch in enumerate(dataloader):
    # Training step
    loss = model(**batch).loss
    loss.backward()
    
    # Update learning rate
    scheduler.step()
    optimizer.step()
    optimizer.zero_grad()
```

### Debugging and Troubleshooting

#### **Common Issues and Solutions**

**1. Out of Memory Errors**
```python
# Solutions
- Reduce batch size
- Enable gradient checkpointing
- Use gradient accumulation
- Apply quantization (4-bit/8-bit)
- Use CPU offloading with DeepSpeed

# Memory monitoring
def monitor_memory():
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

**2. Training Instability**
```python
# Solutions
- Lower learning rate
- Add gradient clipping
- Use warmup steps
- Check data quality
- Reduce LoRA rank

# Gradient monitoring
def monitor_gradients(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    print(f"Gradient norm: {total_norm}")
```

**3. Poor Performance**
```python
# Debugging checklist
- Verify data formatting
- Check loss curves
- Validate evaluation setup
- Try different hyperparameters
- Ensure sufficient training data
- Test with smaller model first

# Loss analysis
def analyze_loss(losses):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.show()
    
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Loss reduction: {(losses[0] - losses[-1]) / losses[0] * 100:.2f}%")
```

This comprehensive guide covers the entire landscape of LLM fine-tuning, from theoretical foundations to practical implementation. The field is rapidly evolving, with new techniques and improvements being published regularly. The key is to understand the fundamentals deeply while staying current with the latest developments.
