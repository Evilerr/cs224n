import random
import argparse

import dataset
import models
import trainer
import utils

import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

random.seed(0)

#function 对应 finetune，表示执行什么任务      可选pretrain：预训练. finetune：微调. evaluate：评估
#variant 对应 vanilla，表示模型类型            可选vanilla：普通位置编码  rope：旋转位置编码
#pretrain_corpus_path：预训练数据路径          对应 wiki.txt

#--reading_params_path：读取已有模型参数
#--writing_params_path：模型训练完保存到哪里
#--finetune_corpus_path：微调数据路径
#--eval_corpus_path：评估数据路径
#--outputs_path：预测结果保存路径
#--pretrain_lr：预训练学习率，默认 0.006
#--finetune_lr：微调学习率，默认 0.0006
#--tb_expt_name：TensorBoard 实验名称

#把整段 argparse 理解成 run.py 的“启动菜单”：用户在终端选择功能，程序根据选项执行对应代码。
argp = argparse.ArgumentParser()
argp.add_argument('function', help="Choose pretrain, finetune, or evaluate")
argp.add_argument('variant', help="Choose vanilla or rope")
argp.add_argument('pretrain_corpus_path', default=None)
argp.add_argument('--reading_params_path',default=None)
argp.add_argument('--writing_params_path',default=None)
argp.add_argument('--finetune_corpus_path', default=None)
argp.add_argument('--eval_corpus_path', default=None)
argp.add_argument('--outputs_path', default=None)
argp.add_argument('--pretrain_lr', default=6e-3, type=float)
argp.add_argument('--finetune_lr', default=6e-4, type=float)
argp.add_argument('--tb_expt_name', help='debug string for tb log.',
                  default='run')
args = argp.parse_args()

device = 'cpu'
if torch.cuda.is_available():
    device = torch.cuda.current_device()
elif torch.backends.mps.is_available() and args.variant == 'vanilla':
    device = 'mps'

# TensorBoard training log
writer = SummaryWriter(log_dir='expt/%s/%s_%s_pt_lr_%f_ft_lr_%f' % (
    args.function,
    args.tb_expt_name,
    args.variant,
    args.pretrain_lr,
    args.finetune_lr))

# Keep the block size 128
# Why is the pretraining corpus always required (even if we're not pretraining?)
# It's because we're using it as a hack to always have the same vocabulary
# (that is, the same mapping from character to integer, and we build the
# vocab from the pretraining corpus.)

#block_size = 128：每条文本序列包含128个字符 token。

block_size = 128
text = open(args.pretrain_corpus_path, encoding='utf-8').read()
pretrain_dataset = dataset.CharCorruptionDataset(text, block_size)

# We don't suggest you change these hyperparameters, as they're known to work.
# use them for both the vanilla and the RoPE models
#创建模型的“配置说明书” mconf，还没有真正创建模型。
mconf = models.GPTConfig(
    pretrain_dataset.vocab_size,
    pretrain_dataset.block_size,
    n_layer=4,
    n_head=8,
    n_embd=256)

# define models.
# note: models should moved to device defined on lines 30-34.

model = None
if args.variant == 'vanilla':
    # TODO: [part c] Make some model here
    ### YOUR CODE HERE ###
    model = models.GPT(mconf).to(device) #to(device)是说把模型放到指定设备上运行
    ### END YOUR CODE ###
elif args.variant == 'rope':
    # TODO: [part g] Make some other model here
    # set mconf.rope parameter
    ### YOUR CODE HERE ###
    mconf.rope = True
    model = models.GPT(mconf).to(device)
    ### END YOUR CODE ###
else:
    raise ValueError("Unknown model variant")

print('Model on device: ', next(model.parameters()).device)

# Perform pretraining, finetuning, or evaluation
if args.function == 'pretrain':
    assert args.writing_params_path is not None
    # TODO [part f]:
    # - Given:
    #     1. A corpus specified in args.pretrain_corpus_path
    #     2. An output path args.writing_params_path for the model parameters
    # - Goals:
    #     1. Pretrain the model on this corpus
    #     2. Save the resulting model in args.writing_params_path

    # - Make sure to use the following hyperparameters for pretraining:
    # Hyperparameters for pretraining:
    # max_epochs=650
    # batch_size=128
    # learning_rate=args.pretrain_lr
    # lr_decay=True
    # warmup_tokens=512*20
    # final_tokens=650*len(pretrain_dataset)*block_size
    # num_workers=4
    # writer=writer

    ### YOUR CODE HERE ###
    
    pretrain_config = trainer.TrainerConfig(
        max_epochs = 650,
        batch_size = 128,
        learning_rate = args.pretrain_lr,
        lr_decay = True,
        warmup_tokens = 512*20,
        final_tokens = 650*len(pretrain_dataset)*block_size,
        num_workers = 4,     #DataLoader 用4个后台进程准备数据，训练时取 batch 会更快
        writer = writer,     #
        #checkpoint settings
        ckpt_path=args.writing_params_path
        )

    pretrainer = trainer.Trainer(model,pretrain_dataset,None,pretrain_config)

    pretrainer.train()
    ### END YOUR CODE ###
elif args.function == 'finetune':
    assert args.writing_params_path is not None
    assert args.finetune_corpus_path is not None
    # TODO [part d] [part f]:
    # - Given:
    #     1. A finetuning corpus specified in args.finetune_corpus_path
    #     2. A path args.reading_params_path containing pretrained model
    #         parameters, or None if finetuning without a pretrained model
    #     3. An output path args.writing_params_path for the model parameters
    # - Goals:
    #     1. If args.reading_params_path is specified, load these parameters
    #         into the model
    #     2. Finetune the model on this corpus
    #     3. Save the resulting model in args.writing_params_path
    # - Make sure to use the following hyperparameters:
    #     [part d] Hyperparameters for finetuning WITHOUT a pretrained model:
    #         max_epochs=75
    #         batch_size=256
    #         learning_rate=args.finetune_lr
    #         lr_decay=True
    #         warmup_tokens=512*20
    #         final_tokens=200*len(pretrain_dataset)*block_size
    #         num_workers=4
    #         writer=writer
    #     [part f] Hyperparameters for finetuning WITH a pretrained model:
    #         max_epochs=10
    #         batch_size=256
    #         learning_rate=args.finetune_lr
    #         lr_decay=True
    #         warmup_tokens=512*20
    #         final_tokens=200*len(pretrain_dataset)*block_size
    #         num_workers=4
    #         writer=writer
    #     You can use the args.reading_params_path flag to switch between the
    #     number of epochs for each case.

    ### YOUR CODE HERE ###
    finetune_text = open(args.finetune_corpus_path, encoding='utf-8').read()
    finetune_dataset = dataset.NameDataset(pretrain_dataset,finetune_text)

    if args.reading_params_path is not None:#有预训练模型

        #加载预训练模型
        model.load_state_dict(torch.load(args.reading_params_path))

        finetune_config = trainer.TrainerConfig(
            max_epochs = 10,
            batch_size = 256,
            learning_rate = args.finetune_lr,
            lr_decay = True,
            warmup_tokens = 512*20,
            final_tokens = 200*len(pretrain_dataset)*block_size,
            num_workers = 4,
            writer = writer,
            ckpt_path=args.writing_params_path
        )

        finetune_trainer = trainer.Trainer(model,finetune_dataset,None, finetune_config)
    else:
        finetune_config = trainer.TrainerConfig(
            max_epochs = 75,
            batch_size = 256,
            learning_rate = args.finetune_lr,
            lr_decay = True,
            warmup_tokens = 512*20,
            final_tokens = 200*len(pretrain_dataset)*block_size,
            num_workers = 4,
            writer = writer,
            ckpt_path=args.writing_params_path
        )
        finetune_trainer = trainer.Trainer(model,finetune_dataset,None, finetune_config)

    finetune_trainer.train()
    ### END YOUR CODE ###
elif args.function == 'evaluate':
    assert args.outputs_path is not None
    assert args.reading_params_path is not None
    assert args.eval_corpus_path is not None
    model.load_state_dict(torch.load(args.reading_params_path))
    correct = 0
    total = 0
    with open(args.outputs_path, 'w', encoding='utf-8') as fout:
        predictions = []
        for line in tqdm(open(args.eval_corpus_path, encoding='utf-8')):
            x = line.split('\t')[0]
            x = x + '⁇'
            x = torch.tensor([pretrain_dataset.stoi[s] for s in x],
                             dtype=torch.long)[None,...].to(device)
            pred = utils.sample(model, x, 32, sample=False)[0]
            completion = ''.join([pretrain_dataset.itos[int(i)] for i in pred])
            pred = completion.split('⁇')[1]
            predictions.append(pred)
            fout.write(pred + '\n')
        total, correct = utils.evaluate_places(args.eval_corpus_path, predictions)
    if total > 0:
        print(f'Correct: {correct} out of {total}: {correct/total*100}%')
    else:
        print(f'Predictions written to {args.outputs_path}; no targets provided')
