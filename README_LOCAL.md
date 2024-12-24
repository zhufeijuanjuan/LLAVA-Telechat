## 1. 第1阶段预训练
```angular2html
bash scripts/archived/pretrain.sh
```

## 2. 预训练之后进行验证测试CLI

```angular2html
CUDA_VISIBLE_DEVICES=0 python llava/serve/cli.py --model-path "/data/LLaVA-NeXT/checkpoints/llava-telechat2-7B-pretrain/" --model-base "/data/LLaVA-NeXT/pretrain_model/telechat2-7B" --image-file "/data/LLaVA-NeXT/docs/ov_chat_images/example2_dog.jpg"
```
- model-path为预训练保存的mm-projector权重路径，model-base为预训练模型路径，image-file为需要预测的图片路径。


## 3. 第二阶段视觉指令微调
```angular2html
bash scripts/archived/finetune_lora.sh
```
