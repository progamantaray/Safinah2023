# Program Pengolahan Citra Gamantaray UGM 2021

Program ini digunakan untuk deteksi bola merah dan hijau menggunakan algoritma YOLOv4 pada kapal Safinah One

## Cara Pakai
### Conda (Recommended)

```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov4-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov4-gpu
```

### Pip
```bash
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt
```

## Download Weights dan Names
File weights hasil training dapat di unduh di drive Gamantaray UGM dalam folder yolov4/backup
Untuk versi yang digunakan pilih yolov4-obj_best.weights
Jika versi tersebut tidak ada, hubungi sesepuh programmer terdekat

File names dapat di download di drive Gamantaray UGM dalam folder yolov4
Untuk versi yang digunakan pilih obj.names

## Memasang Weights dan Names
Ubah nama file weights yang sudah di unduh menjadi ball.weights
Pindahkan file weights yang sudah di unduh pada safinah_vison/data

Ubah nama file names yang sudah di unduh menjadi ball.names
Pindahkan file names yang sudah di unduh pada safinah_vison/data/classes

## Konversi ke Tensorflow
Untuk menjalankan YOLOv4 dengan TensorFlow, kita perlu mengkonversi .weights ke .pb
```bash
# yolov4 biasa
python save_model.py --weights ./data/ball.weights --output ./checkpoints/ball-416 --input_size 416 --model yolov4 
```

## Menjalankan Tensorflow
<strong>Akan ada kemungkinan error saat pertama clone karena direktori atau file tidak ada, Jika error tersebut terjadi maka silakan dibuat sendiri direktorinya</strong>

```bash
# Foto
python detect.py --weights ./checkpoints/ball-416 --size 416 --model yolov4 --images ./data/images/bola.jpg

# Video
python detect_video.py --weights ./checkpoints/ball-416 --size 416 --model yolov4 --video ./data/video/kapal1.mp4 --output ./detections/kapal1.avi --info --count

# Webcam
python detect_video.py --weights ./checkpoints/ball-416 --size 416 --model yolov4 --video 0 --output ./detections/kamerakapal1.avi --info --count
```

## Command Lainnya

```bash
save_model.py:
  --weights: path to weights file
    (default: './data/yolov4.weights')
  --output: path to output
    (default: './checkpoints/yolov4-416')
  --[no]tiny: yolov4 or yolov4-tiny
    (default: 'False')
  --input_size: define input size of export model
    (default: 416)
  --framework: what framework to use (tf, trt, tflite)
    (default: tf)
  --model: yolov3 or yolov4
    (default: yolov4)

detect.py:
  --images: path to input images as a string with images separated by ","
    (default: './data/images/kite.jpg')
  --output: path to output folder
    (default: './detections/')
  --[no]tiny: yolov4 or yolov4-tiny
    (default: 'False')
  --weights: path to weights file
    (default: './checkpoints/yolov4-416')
  --framework: what framework to use (tf, trt, tflite)
    (default: tf)
  --model: yolov3 or yolov4
    (default: yolov4)
  --size: resize images to
    (default: 416)
  --iou: iou threshold
    (default: 0.45)
  --score: confidence threshold
    (default: 0.25)
  --count: count objects within images
    (default: False)
  --dont_show: dont show image output
    (default: False)
  --info: print info on detections
    (default: False)
  --crop: crop detections and save as new images
    (default: False)
    
detect_video.py:
  --video: path to input video (use 0 for webcam)
    (default: './data/video/video.mp4')
  --output: path to output video (remember to set right codec for given format. e.g. XVID for .avi)
    (default: None)
  --output_format: codec used in VideoWriter when saving video to file
    (default: 'XVID)
  --[no]tiny: yolov4 or yolov4-tiny
    (default: 'false')
  --weights: path to weights file
    (default: './checkpoints/yolov4-416')
  --framework: what framework to use (tf, trt, tflite)
    (default: tf)
  --model: yolov3 or yolov4
    (default: yolov4)
  --size: resize images to
    (default: 416)
  --iou: iou threshold
    (default: 0.45)
  --score: confidence threshold
    (default: 0.25)
  --count: count objects within video
    (default: False)
  --dont_show: dont show video output
    (default: False)
  --info: print info on detections
    (default: False)
  --crop: crop detections and save as new images
    (default: False)
```

### Referensi 

   Terima kasih kepada AI Guy yang sudah membuat backbone untuk program ini:
  * [yolov4-custom-functions](https://github.com/theAIGuysCode/yolov4-custom-functions)
