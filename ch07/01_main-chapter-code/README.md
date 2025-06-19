# 7장: 지시를 따르도록 미세 튜닝하기

### 예제 코드

- [ch07.ipynb](ch07.ipynb)에는 이 장에 포함된 모든 코드가 담겨 있습니다.
- [previous_chapters.py](previous_chapters.py)는 이전 장에서 만들고 훈련한 GPT 모델과 여러 유틸리 함수를 담고 있는 파이썬 모듈입니다.
- [gpt_download.py](gpt_download.py)는 사전 훈련된 GPT 모델의 가중치를 다운로드하기 위함 유틸리티 함수를 담고 있습니다.
- [exercise-solutions.ipynb](exercise-solutions.ipynb)는 이 장의 연습문제 솔루션을 담고 있습니다.


### Optional Code

- [load-finetuned-model.ipynb](load-finetuned-model.ipynb) is a standalone Jupyter notebook to load the instruction finetuned model we created in this chapter

- [gpt_instruction_finetuning.py](gpt_instruction_finetuning.py) is a standalone Python script to instruction finetune the model as described in the main chapter (think of it as a chapter summary focused on the finetuning parts)

Usage:

```bash
python gpt_instruction_finetuning.py
```

```
matplotlib version: 3.9.0
tiktoken version: 0.7.0
torch version: 2.3.1
tqdm version: 4.66.4
tensorflow version: 2.16.1
--------------------------------------------------
Training set length: 935
Validation set length: 55
Test set length: 110
--------------------------------------------------
Device: cpu
--------------------------------------------------
File already exists and is up-to-date: gpt2/355M/checkpoint
File already exists and is up-to-date: gpt2/355M/encoder.json
File already exists and is up-to-date: gpt2/355M/hparams.json
File already exists and is up-to-date: gpt2/355M/model.ckpt.data-00000-of-00001
File already exists and is up-to-date: gpt2/355M/model.ckpt.index
File already exists and is up-to-date: gpt2/355M/model.ckpt.meta
File already exists and is up-to-date: gpt2/355M/vocab.bpe
Loaded model: gpt2-medium (355M)
--------------------------------------------------
Initial losses
   Training loss: 3.839039182662964
   Validation loss: 3.7619192123413088
Ep 1 (Step 000000): Train loss 2.611, Val loss 2.668
Ep 1 (Step 000005): Train loss 1.161, Val loss 1.131
Ep 1 (Step 000010): Train loss 0.939, Val loss 0.973
...
Training completed in 15.66 minutes.
Plot saved as loss-plot-standalone.pdf
--------------------------------------------------
Generating responses
100%|█████████████████████████████████████████████████████████| 110/110 [06:57<00:00,  3.80s/it]
Responses saved as instruction-data-with-response-standalone.json
Model saved as gpt2-medium355M-sft-standalone.pth
```

- [ollama_evaluate.py](ollama_evaluate.py) is a standalone Python script to evaluate the responses of the finetuned model as described in the main chapter (think of it as a chapter summary focused on the evaluation parts)

Usage:

```bash
python ollama_evaluate.py --file_path instruction-data-with-response-standalone.json
```

```
Ollama running: True
Scoring entries: 100%|███████████████████████████████████████| 110/110 [01:08<00:00,  1.62it/s]
Number of scores: 110 of 110
Average score: 51.75
```

- [exercise_experiments.py](exercise_experiments.py) is an optional scropt that implements the exercise solutions; for more details see [exercise-solutions.ipynb](exercise-solutions.ipynb)
