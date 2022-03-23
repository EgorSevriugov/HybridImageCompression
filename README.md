# Hybrid Image Compression

**Prerequisites**

Clone [official neural compression githab](https://github.com/facebookresearch/NeuralCompression). Read and check that your system satisfies all requirements, if not follow provided instructions. Move to **Tutorials** directory and copy paste there **compress_image.py** file. 

**Hybrid Compression**

To compress particular image open **Metrics_Example.ipynb**, specify **PATH** to the image and **GPU_ID** (index of GPU to use) and run the following command :

```bash
!CUDA_VISIBLE_DEVICES=GPU_ID python compress_image.py PATH
```
