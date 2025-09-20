# 밑바닥부터 만들면서 배우는 LLM (길벗, 2025)

<밑바닥부터 만들면서 배우는 LLM>(길벗, 2025) 책의 공식 코드 저장소로 GPT와 유사한 LLM의 개발, 사전 훈련, 미세 튜닝하기 위한 코드를 포함하고 있습니다.

<br>

<a href="https://tensorflow.blog/llm-from-scratch/"><img src="https://tensorflow.blog/wp-content/uploads/2025/09/ebb091ebb094eb8ba5llm_ebb3b8ecb185_ec959eeba9b4.jpg" width="350px"></a>

<br>

<밑바닥부터 만들면서 배우는 LLM>에서는 대규모 언어 모델(LLM)이 내부적으로 어떻게 작동하는지 밑바닥부터 단계별로 직접 코딩하면서 배우고 이해합니다. 이 책에서는 명확한 설명, 다이어그램, 예제를 통해 여러분만의 LLM을 만드는 과정을 안내합니다.

이 책에서는 교육적인 목적으로 작지만 완전한 기능을 갖춘 모델을 만듭니다. 이런 모델을 훈련하고 개발하는 방법은 ChatGPT와 같은 대규모 파운데이터션 모델을 만드는 데 사용된 방법과 동일합니다. 또한, 이 책에는 사전 훈련된 모델의 가중치를 불러와 미세 튜닝하는 코드도 포함되어 있습니다.

* [공식 코드 저장소](https://github.com/rickiepark/llm-from-scratch)
* [에러타 페이지](https://tensorflow.blog/llm-from-scratch/)
* [교보문고](https://product.kyobobook.co.kr/detail/S000217570241), [Yes24](https://www.yes24.com/product/goods/154099735), [알라딘](https://www.aladin.co.kr/shop/wproduct.aspx?ItemId=372272431)

<br>

이 저장소를 다운로드하려면 [Download ZIP](https://github.com/rickiepark/llm-from-scratch/archive/refs/heads/main.zip) 버튼을 클릭하거나 터미널에서 다음 명령을 실행하세요:

```bash
git clone --depth 1 https://github.com/rickiepark/llm-from-scratch.git
```

<br>

# 목차

이 `README.md` 파일은 마크다운(Markdown)(`.md`) 파일입니다. 저장소를 다운로드해서 이 파일을 로컬 컴퓨터에서 보려면 마크다운 에디터나 뷰어를 사용하세요.

[깃허브](https://github.com/rickiepark/llm-from-scratch)는 마크다운 파일을 자동으로 렌더링해 줍니다.

<br>

> **팁:**
> 로컬 컴퓨터나 클라우드에서 파이썬 환경을 구성하는 가이드를 찾고 있다면 [setup](setup) 디렉토리 안의 [README.md](setup/README.md) 파일을 참고하세요.

<br>

| 제목                                              | 메인 코드                                                                                                    | 모든 코드 + 부록      |
|------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| [Setup recommendations](setup)                             | -                                                                                                                               | -                             |
| 1장: 대규모 언어 모델 이해하기                  | 코드 없음                                                                                                                         | -                             |
| 2장: 텍스트 데이터 다루기                               | - [ch02.ipynb](ch02/01_main-chapter-code/ch02.ipynb)<br/>- [dataloader.ipynb](ch02/01_main-chapter-code/dataloader.ipynb) (요약)<br/>- [exercise-solutions.ipynb](ch02/01_main-chapter-code/exercise-solutions.ipynb)               | [./ch02](./ch02)            |
| 3장: 어텐션 메커니즘 구현하기                          | - [ch03.ipynb](ch03/01_main-chapter-code/ch03.ipynb)<br/>- [multihead-attention.ipynb](ch03/01_main-chapter-code/multihead-attention.ipynb) (요약) <br/>- [exercise-solutions.ipynb](ch03/01_main-chapter-code/exercise-solutions.ipynb)| [./ch03](./ch03)             |
| 4장: 밑바닥부터 GPT 모델 구현하기                | - [ch04.ipynb](ch04/01_main-chapter-code/ch04.ipynb)<br/>- [gpt.py](ch04/01_main-chapter-code/gpt.py) (요약)<br/>- [exercise-solutions.ipynb](ch04/01_main-chapter-code/exercise-solutions.ipynb) | [./ch04](./ch04)           |
| 5장: 레이블이 없는 데이터를 활용한 사전 훈련                        | - [ch05.ipynb](ch05/01_main-chapter-code/ch05.ipynb)<br/>- [gpt_train.py](ch05/01_main-chapter-code/gpt_train.py) (요약) <br/>- [gpt_generate.py](ch05/01_main-chapter-code/gpt_generate.py) (요약) <br/>- [exercise-solutions.ipynb](ch05/01_main-chapter-code/exercise-solutions.ipynb) | [./ch05](./ch05)              |
| 6장: 분류를 위해 미세 튜닝하기                   | - [ch06.ipynb](ch06/01_main-chapter-code/ch06.ipynb)  <br/>- [gpt_class_finetune.py](ch06/01_main-chapter-code/gpt_class_finetune.py)  <br/>- [exercise-solutions.ipynb](ch06/01_main-chapter-code/exercise-solutions.ipynb) | [./ch06](./ch06)              |
| 7장: 지시를 따르도록 미세 튜닝하기                    | - [ch07.ipynb](ch07/01_main-chapter-code/ch07.ipynb)<br/>- [gpt_instruction_finetuning.py](ch07/01_main-chapter-code/gpt_instruction_finetuning.py) (요약)<br/>- [ollama_evaluate.py](ch07/01_main-chapter-code/ollama_evaluate.py) (요약)<br/>- [exercise-solutions.ipynb](ch07/01_main-chapter-code/exercise-solutions.ipynb) | [./ch07](./ch07)  |
| 부록 A: 파이토치 소개                        | - [code-part1.ipynb](appendix-A/01_main-chapter-code/code-part1.ipynb)<br/>- [code-part2.ipynb](appendix-A/01_main-chapter-code/code-part2.ipynb)<br/>- [DDP-script.py](appendix-A/01_main-chapter-code/DDP-script.py)<br/>- [exercise-solutions.ipynb](appendix-A/01_main-chapter-code/exercise-solutions.ipynb) | [./appendix-A](./appendix-A) |
| 부록 B: 참고 및 더 읽을 거리                 | 코드 없음                                                                                                                         | -                             |
| 부록 C: 연습문제 해답                             | 코드 없음                                                                                                                         | -                             |
| 부록 D: 훈련 루프에 부가 기능 추가하기 | - [appendix-D.ipynb](appendix-D/01_main-chapter-code/appendix-D.ipynb)                                                          | [./appendix-D](./appendix-D)  |
| 부록 E: LoRA를 사용한 파라미터 효율적인 미세 튜닝       | - [appendix-E.ipynb](appendix-E/01_main-chapter-code/appendix-E.ipynb)                                                          | [./appendix-E](./appendix-E) |

<br>
&nbsp;

아래 그림이 이 책에서 다룰 내용을 요약해서 보여줍니다.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/mental-model.jpg" width="650px">

<br>
&nbsp;

## 하드웨어 요구사항

이 책의 주요 장에 포함된 코드는 합리적인 시간 내에 일반 노트북에서 실행될 수 있도록 설계되었으며, 전용 하드웨어가 필요하지 않습니다. 이런 방식을 채택함으로써 더 많은 독자가 쉽게 내용을 따라갈 수 있습니다. 또한, 이 저장소의 코드는 GPU가 사용 가능한 경우 이를 자동으로 활용합니다. (추가 권장 사항은 [설정](setup/README.md) 문서를 참조하세요.)

&nbsp;
## 보너스 자료

관심있는 독자를 위해 몇몇 폴더에 추가 자료가 담겨 있습니다:

- **설정**
  - [Python Setup Tips](setup/01_optional-python-setup-preferences)
  - [Installing Python Packages and Libraries Used In This Book](setup/02_installing-python-libraries)
  - [Docker Environment Setup Guide](setup/03_optional-docker-environment)
- **2장: 텍스트 데이터 다루기**
  - [Byte Pair Encoding (BPE) Tokenizer From Scratch](ch02/05_bpe-from-scratch/bpe-from-scratch.ipynb)
  - [Comparing Various Byte Pair Encoding (BPE) Implementations](ch02/02_bonus_bytepair-encoder)
  - [Understanding the Difference Between Embedding Layers and Linear Layers](ch02/03_bonus_embedding-vs-matmul)
  - [Dataloader Intuition with Simple Numbers](ch02/04_bonus_dataloader-intuition)
- **3장: 어텐션 메커니즘 구현하기**
  - [Comparing Efficient Multi-Head Attention Implementations](ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb)
  - [Understanding PyTorch Buffers](ch03/03_understanding-buffers/understanding-buffers.ipynb)
- **4장: 밑바닥부터 GPT 모델 구현하기**
  - [FLOPS Analysis](ch04/02_performance-analysis/flops-analysis.ipynb)
- **5장: 레이블이 없는 데이터를 활용한 사전 훈련**
  - [Alternative Weight Loading Methods](ch05/02_alternative_weight_loading/)
  - [Pretraining GPT on the Project Gutenberg Dataset](ch05/03_bonus_pretraining_on_gutenberg)
  - [Adding Bells and Whistles to the Training Loop](ch05/04_learning_rate_schedulers)
  - [Optimizing Hyperparameters for Pretraining](ch05/05_bonus_hparam_tuning)
  - [Building a User Interface to Interact With the Pretrained LLM](ch05/06_user_interface)
  - [Converting GPT to Llama](ch05/07_gpt_to_llama)
  - [Llama 3.2 From Scratch](ch05/07_gpt_to_llama/standalone-llama32.ipynb)
  - [Memory-efficient Model Weight Loading](ch05/08_memory_efficient_weight_loading/memory-efficient-state-dict.ipynb)
  - [Extending the Tiktoken BPE Tokenizer with New Tokens](ch05/09_extending-tokenizers/extend-tiktoken.ipynb)
  - [PyTorch Performance Tips for Faster LLM Training](ch05/10_llm-training-speed)
- **6장: 분류를 위해 미세 튜닝하기**
  - [Additional experiments finetuning different layers and using larger models](ch06/02_bonus_additional-experiments)
  - [Finetuning different models on 50k IMDB movie review dataset](ch06/03_bonus_imdb-classification)
  - [Building a User Interface to Interact With the GPT-based Spam Classifier](ch06/04_user_interface)
- **7장: 지시를 따르도록 미세 튜닝하기**
  - [Dataset Utilities for Finding Near Duplicates and Creating Passive Voice Entries](ch07/02_dataset-utilities)
  - [Evaluating Instruction Responses Using the OpenAI API and Ollama](ch07/03_model-evaluation)
  - [Generating a Dataset for Instruction Finetuning](ch07/05_dataset-generation/llama3-ollama.ipynb)
  - [Improving a Dataset for Instruction Finetuning](ch07/05_dataset-generation/reflection-gpt4.ipynb)
  - [Generating a Preference Dataset with Llama 3.1 70B and Ollama](ch07/04_preference-tuning-with-dpo/create-preference-data-ollama.ipynb)
  - [Direct Preference Optimization (DPO) for LLM Alignment](ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb)
  - [Building a User Interface to Interact With the Instruction Finetuned GPT Model](ch07/06_user_interface)
