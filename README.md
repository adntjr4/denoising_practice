# Denoising practice

My custom repository for multiple Denoising task.

- inlcude supervised training
- include self-supervised training

- include various dataset
    - SIDD
    - DND

## How to use

## Add custom code

## Results

## Containing functions

---

### development schedule (inverse time line)

- Generative noise model training.
    - [ ] revise codes for GAN (e.g. loss, trainer, config, etc)
    - [ ] C2N reproduction
    
- basic modules for deep-learning.
    - [ ] Add test code
    - [ ] Add summary printing module
    - [ ] Add tensor board output
    - [x] Use "start_epoch" and "interval_epoch"
    - [x] Add validation image out
    - [x] Add training process for self-supervised manner


### Notepad

- DBSN을 그냥 SIDD_val에서 학습.
- RBSN이 얼마나 바뀌는지 확인
- laplacian으로 가정하면 얼마나 더 좋아지는지 확인
- 더 좋아졌을 경우 DND에서 test하여 결과 확인.


- DBSN의 장점은 4방향으로 퍼진다는 것이다.
- EBSN의 장점은 3x3conv를 사용하여 connection이 많고 skip connection이 있는 network를 구성할 수 있다는 것이다.
- 이 둘의 장점을 합칠 수 있을 것이다.
- receptive field가 좁아져도 성능이 늘어나는가?
    - 1x1 conv만 있어도 괜찮은 가를 실험 or paper survey
- near pixel attention을 적용할 방법을 생각.
- training scheme을 Laine or DBSN처럼 하는 방법을 생각.
