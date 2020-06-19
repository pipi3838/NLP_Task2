# NLP Final Task2

## Usage
### train cause and effect classifier separately

***1. specify cause/effect in the config file***

```python=
dataset:
    name: 'Task2Dataset'
    kwargs:
        data_dir: '/nfs/nas-5.1/wbcheng/nlp_task2/data/'
        max_length: 512
        cause_or_effect: 'cause'
``` 
cause_or_effect = cause / effect

***2. select the pretrain model config file***
```python=
net:
    name: 'pretrainedNet'
    kwargs:
        model_type: 'bert'
        trained_path: 'bert-base-uncased'
        num_labels: 2
```
>1. bert-base-uncased
>2. bert-base-cased
>3. bert-large-uncased-whole-word-masking-finetuned-squad
>4. .....

***3. train model!***
> python -m src.main configs/train/task2.yaml 

