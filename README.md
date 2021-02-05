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

- receptive field가 좁아져도 성능이 늘어나는가?
    - 1x1 conv만 있어도 괜찮은 가를 실험 or paper survey
- near pixel attention을 적용할 방법을 생각.
- training scheme을 Laine or DBSN처럼 하는 방법을 생각.
