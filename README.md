# Plan-and-Rerank Framework for Reasoning-Intensive Retrieval

## setup
- For inference,
```
pip install -r requirements.txt
```

- For training,
```
cd shortcut_reranker/train
pip install -r requirements.txt
```



## Training
- Train data generation
```
bash bash/train_sft.sh
```

## Graph construction

- Graph construction
```
bash bash/construct_graph.sh
```

## Inference

- Inference (e.g. BRIGHT)
```
bash bash/rank_bright.sh
```

## Evaluation
- Evaluation (e.g. BRIGHT)
```
bash bash/eval_bright.sh
```