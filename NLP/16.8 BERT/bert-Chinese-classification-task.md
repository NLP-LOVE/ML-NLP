# bert-Chinese-classification-task
bert中文分类实践

在run_classifier_word.py中添加NewsProcessor，即新闻的预处理读入部分 \
在main方法中添加news类型数据处理label \
 processors = { \
        "cola": ColaProcessor,\
        "mnli": MnliProcessor,\
        "mrpc": MrpcProcessor,\
        "news": NewsProcessor,\
    }
    
download_glue_data.py 提供glue_data下面其他的bert论文公测glue数据下载

data目录下是news数据的样例

export GLUE_DIR=/search/odin/bert/extract_code/glue_data \
export BERT_BASE_DIR=/search/odin/bert/chinese_L-12_H-768_A-12/ \
export BERT_PYTORCH_DIR=/search/odin/bert/chinese_L-12_H-768_A-12/

python run_classifier_word.py \
  --task_name NEWS \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/NewsAll/ \
  --vocab_file $BERT_BASE_DIR/vocab.txt \
  --bert_config_file $BERT_BASE_DIR/bert_config.json \
  --init_checkpoint $BERT_PYTORCH_DIR/pytorch_model.bin \
  --max_seq_length 256 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir ./newsAll_output/ \
  --local_rank 3
  
  中文分类任务实践

实验中对中文34个topic进行实践（包括：时政，娱乐，体育等），在对run_classifier.py代码中的预处理环节需要加入NewsProcessor模块，及类似于MrpcProcessor，但是需要对中文的编码进行适当修改，训练数据与测试数据按照4:1进行切割，数据量约80万，单卡GPU资源，训练时间18小时，acc为92.8%

eval_accuracy = 0.9281581998809113

eval_loss = 0.2222444740207354

global_step = 59826

loss = 0.14488934577978746
