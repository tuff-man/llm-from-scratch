# 5장: 레이블이 없는 데이터를 활용한 사전 훈련

### 예제 코드

- [ch05.ipynb](ch05.ipynb)에는 이 장에 포함된 모든 코드가 담겨 있습니다.
- [previous_chapters.py](previous_chapters.py)는 이전 장에서 만든 `MultiHeadAttention`과 `GPTModel` 클래스를 담고 있는 파이썬 모듈입니다. GPT 모델을 사전 훈련하기 위해 [ch05.ipynb](ch05.ipynb)에서 임포트합니다.
- [gpt_download.py](gpt_download.py)는 사전 훈련된 GPT 모델의 가중치를 다운로드하기 위함 유틸리티 함수를 담고 있습니다.
- [exercise-solutions.ipynb](exercise-solutions.ipynb)는 이 장의 연습문제 솔루션을 담고 있습니다.

### Optional Code

- [gpt_train.py](gpt_train.py) is a standalone Python script file with the code that we implemented in [ch05.ipynb](ch05.ipynb) to train the GPT model (you can think of it as a code file summarizing this chapter)
- [gpt_generate.py](gpt_generate.py) is a standalone Python script file with the code that we implemented in [ch05.ipynb](ch05.ipynb) to load and use the pretrained model weights from OpenAI

