### Benchmark / Bert
The reference sample finetuning script is adopted from the example provided by [transformers](https://github.com/huggingface/transformers).  

### Run
Since we adopted the `accelerator` for training, you need to configure the `accelerator` before running the scripts.

### Notice
Different from convs, we don't need to simulate performance of INT8 now. You can direct finetune the model under the `export QSYNC_SIMU=0`.

### Dynamic Batching
When run dynamic batching, be careful that a ratio should be altered according to the batchsize to ensure each gradient contributes same to the overall weight. e.g. when 18. 6 is applied to different devices, you should 1.5times the gradient obtained from the 16 but half the gradient contribution obtained from 8. which means you should set ratio to be 1.5 and 0.5, respectively.

### Trails
We provides several sample trails under the folder of each model, including the case for 1card and 32card. The learning setup for different cards can be different, as we adjust learning rate and epoch numbers to make the finetuning task can be run with a large batchsize on 32 card, with nelgible accuracy drop. Experiment shows this modification siginifcantly accelerates the finetuning of these tasks. (we may produce a paper regarding large batchsize finetuning? XD)

### Use of low-precision kernel
Pls refer to the training script `acc_squad.py` and `acc_swag.py`. You can change the corresponding layer to its low-precision kernel execution format.