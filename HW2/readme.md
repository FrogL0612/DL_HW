HW2_Q1 : 2層NN 
參考Q1.py, test正確率 : 96.67%

HW2_Q2 : 
Q2_m1 -> autoencoder結構並加入regularization &dropout
Q2_m2 -> autoencoder結構
Q2_m3 -> DNN結構

HW2_Q3 : 使用MNIST中dataset
Q3_CNN -> LetNet-5，使用原始dataset訓練並測試
          1.原始dataset正確率 : 99.72%
          2.原始dataset加上30% salt &pepper noise正確率 : 90.47%
          
          3.將原始dataset加上30% salt &pepper noise後訓練
          測試加上30% salt &pepper noise的test set正確率
          正確率 : 99.195%
          
Q3_4_Denoise : 嘗試auto-encoder結構將加入30% salt &pepper noise的dataset還原
          將經過auto-encoder結構處理的data透過Q3-1的LetNet-5
          其正確率 : 97%
