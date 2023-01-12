# Anti-Screenshot-Watermark

Anti-screenshot watermark extraction based on DAE and CNN

WHUCSE Information Hiding Class Assignment (bachelor)

## Based on

+ TERA
+ [UnseenCode](https://github.com/cuihaoleo/UnseenCodeDesktop)
+ [SUNet](https://github.com/FanChiMao/SUNet)
+ [PIMoG](https://github.com/FangHanNUS/PIMoG-An-Effective-Screen-shooting-Noise-Layer-Simulation-for-Deep-Learning-Based-Watermarking-Netw)

## Introductoin

Embedding:
1. **TERAinsert.py**: Embed Watermark
2. **LSDmat.py**: Generate LSD matrix
3. **generate_gif.py**

Denoising:
1. **distortion.py**: Noise Layer
2. **SUNet/**: Denoising network
   + train.py
   + demo.py
   + process_image.py

Extracting:
1. **CNN/**: Extracting network
   + train_CNN.py
   + CNNextract.py
2. **all_possible_combination.py**



**To be well organized**

## Collaborators

+ [@rain152](https://github.com/rain152)
+ [@Mohan Liu](https://github.com/barnard33-233)
+ [@Aphelios](https://github.com/shallowsea53166)
+ [@IvanSG1](https://github.com/IvanSG1)
