
### Prerequisites
- Python 3.9
- torch (tested 2.0)
- torchvision

### Installation
Before installing, set the CUDA_HOME path. Make sure to replace your cuda path below.
```
export CUDA_HOME=/usr/local/cuda
```
```sh
git clone https://github.com/IRVLUTD/robokit.git && cd robokit 
pip install -r requirements.txt
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
python setup.py install
```

Note: Check GroundingDINO [installation](https://github.com/IDEA-Research/GroundingDINO?tab=readme-ov-file#hammer_and_wrench-install) for the following error
```sh
NameError: name '_C' is not defined
```
## Acknowledgments

This project is based on the following repositories (license check mandatory):
- [CLIP](https://github.com/openai/CLIP)
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [DepthAnything](https://huggingface.co/docs/transformers/main/en/model_doc/depth_anything#transformers.DepthAnythingForDepthEstimation)
- [FeatUp](https://github.com/mhamilton723/FeatUp)


## License
This project is licensed under the MIT License. However, before using this tool please check the respective works for specific licenses.
