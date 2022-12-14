# DB Sampler
Since the open-source official implementation for [Semi-Dynamic Load Balancing: Efficient Distributed Learning in Non-Dedicated Environments](https://jhc.sjtu.edu.cn/~chen-chen/papers/lbbsp-socc20.pdf) is wrong and biased(which directly slice the data, interupt the training), and the paper mentioned to use sampler. This is a self-implemented version for the Dyanmic Batching by modifying the distributed sampler and DB sampler. 
- We don't implement the runtime version as in the non-dedicated environment, the inherent computational difference and memory difference introduced the biggest gap among devices. 


## Notice
Since we don't focus too much on the implementation of DB, it will produce a wrong verbose in the terminal (tqdm), but the batchsize will be altered correctly according to the `dy_bs`. Dont' worry :].


