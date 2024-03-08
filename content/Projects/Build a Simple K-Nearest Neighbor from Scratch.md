---
title: Build a Simple K-Nearest Neighbor from Scratch
draft: true
tags:
  - MachineLearning
date: 2024-09-08
---
K-Nearest Neighbor, atau KNN, adalah salah satu algoritma supervised machine learning yang sangat populer dan seringkali digunakan untuk menyelesaikan kasus-kasus klasifikasi maupun regresi. Cara kerja dari KNN ini sebenarnya cukup sederhana, yaitu dengan mengidentifikasi "K" atau jumlah tetangga terdekat dari data yang sudah ada kemudian menggunakan sampel tersebut untuk menentukan prediksi nilai regresi ataupun klasifikasi dari data baru. Cara kerja dan metode dari KNN adalah non-parametrik karena algoritma ini tidak membuat asumsi apapun dari distribusi data yang ada, sederhananya KNN menentukan nilai atau kelas dari sebuah data dengan melihat data yang ada di sekitar data tersebut. Meskipun sederhana dan sudah diperkenalkan sejak 1950-an, algoritma ini masih jadi salah satu pilihan yang bisa diandalkan untuk membangun kasus-kasus machine learning, terutama pada dataset yang ukurannya kecil hingga menengan dan sudah cukup bersih.

Untuk dapat memahami lebih dalam apa itu KNN dan bagaimana cara kerjanya, di dalam artikel ini kita akan menulis KNN dari awal menggunakan Python. Pada artikel berikutnya, kita akan coba mengimplementasikan algoritma yang sudah kita tulis ini ke dalam sebuah use-case.

Kita akan menulis tiga source code, yaitu `base.py`, `classification.py`, dan `regression.py`, dan kemudian kita inisialisasi di dalam `__init__.py`. Direktori dari project ini akan terlihat kurang-lebih seperti berikut:
```
ml_from_scratch/ 
│ 
├── neighbors/ 
│ ├── __init__.py 
│ ├── base.py 
│ ├── classification.py 
│ └── regression.py 
│ 
└── try_ml_algo_notebooks/ 
│└── try_knn.ipynb
```
