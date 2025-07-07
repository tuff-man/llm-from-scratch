# 부록 A: 파이토치 소개

### 예제 코드

- [code-part1.ipynb](code-part1.ipynb)는 A.1절부터 A.8절까지 코드가 포함되어 있습니다.
- [code-part2.ipynb](code-part2.ipynb)는 A.9절의 GPU 코드가 담겨 있습니다.
- [DDP-script.py](DDP-script.py)는 다중 GPU 사용 방법을 보여줍니다(주피터 노트북은 단일 GPU만 지원하기 때문에 스크립트로 작성되었습니다). `python DDP-script.py`와 같이 실행할 수 있습니다. 2개 이상의 GPU가 있다면 `CUDA_VISIBLE_DEVIVES=0,1 python DDP-script.py`와 같이 실행하세요.
- [exercise-solutions.ipynb](exercise-solutions.ipynb)는 이장의 연습문제 솔루션을 담고 있습니다.

### Optional Code

- [DDP-script-torchrun.py](DDP-script-torchrun.py) is an optional version of the `DDP-script.py` script that runs via the PyTorch `torchrun` command instead of spawning and managing multiple processes ourselves via `multiprocessing.spawn`. The `torchrun` command has the advantage of automatically handling distributed initialization, including multi-node coordination, which slightly simplifies the setup process. You can use this script via `torchrun --nproc_per_node=2 DDP-script-torchrun.py`
