# Codes for "Unknown Support Prototype Set for Open Set Recognition"
Codes for "Unknown Support Prototype Set for Open Set Recognition". The corresponding paper can be found in \[[link](https://link.springer.com/article/10.1007/s11263-025-02384-9\)]. <br>
* Highlights：
1. In contrast with classical prototype-based methods which constructs prototypes for unknown classes, we construct prototypes for unknowns without any available samples.<br>
2. We construct unknown prototypes in semantic space and then map them to deep feature space.
* Implimentation：<br>
This repository provides a pytorch-version implementation of USUP. The scripts was tested on CIFAR-10-10. Here are the running commands:
1. cd ./backbone
2. python3 run.py --dataset cifar-10-10 --split_idx 3  --data_root /root/data
3. cd ..
4. python3 cgan.py --dataset cifar10 --split 3
5. python3 usps.py --dataset cifar10 --split 3


* Notes：<br>
We found sensitive hyperparameters include: training epoch of cgan, adversarial training round and epoch per round.
 
If you have any problems, feel free to contact me. Have fun and may it inspire your own idea :-)

