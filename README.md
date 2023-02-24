# IQA-RC
Optimizing Dnn Based Quality Assessment Metric for Image Compression: A Novel Rate Control Method
## 1. Environment
    Python 3.7
    PyTorch = 1.5.1
    opencv-python

## 2. Dataset
The four datasets, Kodak, Tecnick, Kadid-10k and TID2013, applied in this project are open source and can be downloaded according to references of the paper.

## 3. Pre-process
1. Preprocess the original file to get the bit rate distribution of the original images(saliencybits.txt) set under different QPs and the encoded image(*.bin). 
Under preprocess folder, modify the *.cfg configuration file，with InputFile/FramesToBeEncoded/BitstreamFile/QP settings, then run:
    ```
    .\TAppEncoder.exe -c .\BasketballPass.cfg -c .\encoder_intra_main.cfg
    ```
2. Extract the image files generated under different QPs from the. bin file. Run:
    ```
    ffmpeg -i *.bin %d.png
    ```
## 4. Fitting
1. Before getting the visualization map, for instance, if you want to fine-tune the IQA model of NIMA, run the following command in `./VSD model/NIMA/`.
    ```
    python train_nima.py
    ```
2. Run the following command for getting the visualization maps and weighted MSE of images of different QPs.
    ```
    python wmse_cam_qp_nima.py
    ```
3. Use MATLAB to .

## 5. Process after fitting
1. After getting the fitted visualization map, change the map resolution to 448 * 448 through the 448.sh file under afterfitting folder.
    ```
    sh 448.sh
    ```
2. Run read49.py under the directory to generate the saliency information file corresponding to the map under each QPs. Generate saliency.txt corresponding to different qps in each fitting folder.
    ```
    python read49.py
    ```
4）Visualization map encode
1. Modify the *.cfg configuration file with InputFile/FramesToBeEncoded/BitstreamFile/QP settings.
2. Under vismapencoder folder, run:
    ```
    .\TAppEncoder.exe -c .\BasketballPass.cfg -c .\encoder_intra_main.cfg --SaliencyTxt=saliency.txt --SaliencyBitsTxt=saliencybits.txt
    ```
to get the re-encoded image with visualisation map.
