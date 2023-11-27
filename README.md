# TBRNet

<img src="https://github.com/nachifur/TBRNet/blob/main/img/f2.jpg"/>

# 1. Resources

## 1.1 Dataset
* [SRD](https://github.com/Liangqiong/DeShadowNet)
* [ISTD](https://github.com/DeepInsight-PCALab/ST-CGAN)
* [ISTD+DA, SRD+DA](https://github.com/vinthony/ghost-free-shadow-removal)

## 1.2 Results
* Results on SRD: [TBRNet_SRD](https://mailustceducn-my.sharepoint.com/:u:/g/personal/nachifur_mail_ustc_edu_cn/EcjEGvbQLY5FhM_HTB0sGiwBzVGZgLJ0hOzvvLBZ9aHgSg?e=zMopUV)
* Results on SRD+DA: [TBRNet_SRD_DA](https://mailustceducn-my.sharepoint.com/:u:/g/personal/nachifur_mail_ustc_edu_cn/ERZMY7pF1aJEtpGjPx4eMBQBAJF7YRJjaWkiaAFqZh7_xQ?e=Uhkc84)
* Results on ISTD: [TBRNet_ISTD](https://mailustceducn-my.sharepoint.com/:u:/g/personal/nachifur_mail_ustc_edu_cn/Eand2fwdBplBnOi9DGCeyewBTDNTbR4H1gMkXUu6nB186g?e=N9iiX2)
* Results on ISTD+DA: [TBRNet_ISTD_DA](https://mailustceducn-my.sharepoint.com/:u:/g/personal/nachifur_mail_ustc_edu_cn/ETavmK5QwspJiTUCgBuK3wABNR6T3bJBrdSFQE2Urp6Jgw?e=BnC1Pd)
* Results on ISTD+: [TBRNet_ISTD+](https://mailustceducn-my.sharepoint.com/:u:/g/personal/nachifur_mail_ustc_edu_cn/EZssY3AcYnlKud00MPqANX8BYmikxY4WSyCFYAuTQkAXFA?e=i2HZQ8)

*Visual comparison results of penumbra removal results on the SRD dataset* - (Powered by [MulimgViewer](https://github.com/nachifur/MulimgViewer))
<img src="https://github.com/nachifur/TBRNet/blob/main/img/f1.jpg"/>


## 1.3  Model Weight File for Test
* Model on SRD: [TBRNet_SRD.pth](https://mailustceducn-my.sharepoint.com/:u:/g/personal/nachifur_mail_ustc_edu_cn/EfX-fgeSt-RGpq70XAwk8RYBdzB-dbXqi2snIgIbhCjCqg?e=byAji3)
* Model on SRD+DA: [TBRNet_SRD_DA.pth](https://mailustceducn-my.sharepoint.com/:u:/g/personal/nachifur_mail_ustc_edu_cn/ET7oQhFbKNNKtSIz1i6247MBeF20MsWe8uyjqrpH6BkG4Q?e=l1F6mE)
* Model on ISTD: [TBRNet_ISTD.pth](https://mailustceducn-my.sharepoint.com/:u:/g/personal/nachifur_mail_ustc_edu_cn/EW7n1zFH2FpBvjvfNC0mrcMB45n6aUocl81EhOokZDLMeA?e=409Tiu)
* Model on ISTD+DA: [TBRNet_ISTD_DA.pth](https://mailustceducn-my.sharepoint.com/:u:/g/personal/nachifur_mail_ustc_edu_cn/EcNu1-1LnNFCjcvtbvv6JC0BILJh_-VDCn3iVKqv9wcROQ?e=pRAciZ)
* Model on ISTD+: [TBRNet_ISTD+.pth](https://mailustceducn-my.sharepoint.com/:u:/g/personal/nachifur_mail_ustc_edu_cn/EU5yZC_FspZMoh-sJg_YWM4BG5rVLGS-Jl0KK0UkPJ14qw?e=MWLrHj)

## 1.4 Evaluation Code
1. MAE (i.e., RMSE in paper). [Hieu Le](https://openaccess.thecvf.com/content_ICCV_2019/papers/Le_Shadow_Removal_via_Shadow_Image_Decomposition_ICCV_2019_paper.pdf) reuploaded the [evaluation code](https://github.com/cvlab-stonybrook/SID). Before that, everyone can obtain this code from [Jifeng Wang](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Stacked_Conditional_Generative_CVPR_2018_paper.pdf) via email. Currently, the MATLAB code is used in most state-of-the-art works for shadow removal.

2. PSNR+SSIM. [zhu_aaai2022](https://github.com/zhuyr97/AAAI2022_Unfolding_Network_Shadow_Removal)

# 2. Environments
**ubuntu18.04+cuda10.2+pytorch1.7.1**
1. create environments
```
conda env create -f install.yaml
```
2. activate environments
```
conda activate TBRNet
```

# 3. Data Processing
For example, generate the dataset list of ISTD:
1. Download:
   * ISTD and SRD
   * [USR shadowfree images](https://github.com/xw-hu/Mask-ShadowGAN)
   * [Syn. Shadow](https://github.com/vinthony/ghost-free-shadow-removal)
   * [SRD shadow mask](https://github.com/vinthony/ghost-free-shadow-removal) (or try our [re-uploaded link](https://mailustceducn-my.sharepoint.com/:u:/g/personal/nachifur_mail_ustc_edu_cn/Efi0tTOuIeJDj_s0FhYh2tUBbJkkGtjQY_PXIBgjanr-dg?e=0HZTnx))
   * train_B_ISTD:
   ```
   cp -r ISTD_Dataset_arg/train_B ISTD_Dataset_arg/train_B_ISTD
   cp -r ISTD_Dataset_arg/train_B SRD_Dataset_arg/train_B_ISTD
   ```
   * [VGG19](https://download.pytorch.org/models/vgg19-dcbb9e9d.pth)
   ```
   cp vgg19-dcbb9e9d.pth ISTD_Dataset_arg/
   cp vgg19-dcbb9e9d.pth SRD_Dataset_arg/
   ```
2. The data folders should be:
    ```
    ISTD_Dataset_arg
        * train
            - train_A # ISTD shadow image
            - train_B # ISTD shadow mask
            - train_C # ISTD shadowfree image
            - shadow_free # USR shadowfree images
            - synC # Syn. shadow
            - train_B_ISTD # ISTD shadow mask
        * test
            - test_A # ISTD shadow image
            - test_B # ISTD shadow mask
            - test_C # ISTD shadowfree image
        * vgg19-dcbb9e9d.pth

    SRD_Dataset_arg
        * train #  renaming the original `Train` folder in `SRD`.
            - train_A # SRD shadow image, renaming the original `shadow` folder in `SRD`.
            - train_B # SRD shadow mask
            - train_C # SRD shadowfree image, renaming the original `shadow_free` folder in `SRD`.
            - shadow_free # USR shadowfree images
            - synC # Syn. shadow
            - train_B_ISTD # ISTD shadow mask
        * test #  renaming the original `test_data` folder in `SRD`.
            - train_A # SRD shadow image, renaming the original `shadow` folder in `SRD`.
            - train_B # SRD shadow mask
            - train_C # SRD shadowfree image, renaming the original `shadow_free` folder in `SRD`.
        * vgg19-dcbb9e9d.pth 
    ```
3. Edit `generate_flist_istd.py`: (Replace path)

```
ISTD_path = "/Your_data_storage_path/ISTD_Dataset_arg"
```
4. Generate Datasets List. (Already contains ISTD+DA.)
```
conda activate TBRNet
cd script/
python generate_flist_istd.py
```
5. Edit `config_ISTD.yml`: (Replace path)
```
DATA_ROOT: /Your_data_storage_path/ISTD_Dataset_arg
```

# 4. Training+Test+Evaluation
## 4.1 Training+Test+Evaluation
For example, training+test+evaluation on ISTD dataset.
```
cp config/config_ISTD.yml config.yml 
cp config/run_ISTD.py run.py
conda activate TBRNet
python run.py
```
## 4.2 Only Test and Evaluation
For example, test+evaluation on ISTD dataset.
1. Download weight file(`TBRNet_ISTD.pth`) to `pre_train_model/ISTD`
2. Copy file
```
cp config/config_ISTD.yml config.yml 
cp config/run_ISTD.py run.py
mkdir -p checkpoints/ISTD/
cp config.yml checkpoints/ISTD/config.yml
cp pre_train_model/ISTD/TBRNet_ISTD.pth  checkpoints/ISTD/ShadowRemoval.pth
```

3. Edit `run.py`. Comment the training code.

```
    # # pre_train (no data augmentation)
    # MODE = 0
    # print('\nmode-'+str(MODE)+': start pre_training(data augmentation)...\n')
    # for i in range(1):
    #     skip_train = init_config(checkpoints_path, MODE=MODE,
    #                             EVAL_INTERVAL_EPOCH=1, EPOCH=[90,i])
    #     if not skip_train:
    #         main(MODE, config_path)
    # src_path = Path('./pre_train_model') / \
    #     config["SUBJECT_WORD"]/(config["MODEL_NAME"]+'_pre_da.pth')
    # copypth(dest_path, src_path)

    # # train
    # MODE = 2
    # print('\nmode-'+str(MODE)+': start training...\n')
    # for i in range(1):
    #     skip_train = init_config(checkpoints_path, MODE=MODE,
    #                             EVAL_INTERVAL_EPOCH=0.1, EPOCH=[60,i])
    #     if not skip_train:
    #         main(MODE, config_path)
    # src_path = Path('./pre_train_model') / \
    #     config["SUBJECT_WORD"]/(config["MODEL_NAME"]+'_final.pth')
    # copypth(dest_path, src_path)
```
4. Run

```
conda activate TBRNet
python run.py
```
## 4.3 Show Results
After evaluation, execute the following code to display the final RMSE.
```
python show_eval_result.py
```
Output:
```
running rmse-shadow: xxx, rmse-non-shadow: xxx, rmse-all: xxx # ISRD
```
This is the evaluation result of python+pytorch, which is only used during training. To get the evaluation results in the paper, you need to run the [matlab code](#14-evaluation-code).

# 5. Acknowledgements
Part of the code is based upon:
* https://github.com/nachifur/LLPC
* https://github.com/vinthony/ghost-free-shadow-removal
* https://github.com/knazeri/edge-connect

# 6. Citation
If you find our work useful in your research, please consider citing:
```
@ARTICLE{liu2023shadow,
  author={Liu, Jiawei and Wang, Qiang and Fan, Huijie and Tian, Jiandong and Tang, Yandong},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={A Shadow Imaging Bilinear Model and Three-Branch Residual Network for Shadow Removal}, 
  year={2023},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TNNLS.2023.3290078}
}
```
# 7. Contact
Please contact Jiawei Liu if there is any question (liujiawei18@mails.ucas.ac.cn).
