# 🧪 TensorBoard on AWS EC2 실습

이 프로젝트는 AWS EC2에서 TensorFlow 모델을 학습하고, TensorBoard를 통해 로그를 시각화하는 실습입니다.
Ubuntu + 가상환경 + TensorBoard 구성을 통해, 클라우드 기반 시험 환경을 실제로 구조해보는 것이 목표입니다.

---

## 📌 실습 목적

* TensorBoard의 기본 사용법을 실습
* TensorFlow 모델 학습 로그를 수집하고 웹 UI로 시각화
* EC2 인스턴스에서 원격 시험 환경 구성하기

---

## ⚙️ 실습 환경

| 항목      | 내용                          |
| ------- | --------------------------- |
| 클라우드    | AWS EC2                     |
| 인스턴스 타입 | `t3.medium`                 |
| OS      | Ubuntu 22.04                |
| Python  | 3.10+                       |
| 가상환경    | Python venv 사용              |
| 주우 패키지  | `tensorflow`, `tensorboard` |

---

## 🧼 실습 전체 과정

### 1. EC2 인스턴스 생성

* Ubuntu 22.04 기반
* 인번드 규칙: TCP 포트 **22 (SSH)**, **6006 (TensorBoard)** 허용

### 2. 서버 접속 & 가상환경 구성

```bash
ssh -i your_key.pem ubuntu@<EC2 Public DNS>

sudo apt update
sudo apt install python3-venv -y
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install tensorflow tensorboard tensorboard-plugin-profile
```

---

### 3. 코드 작성: `train_tfboard.py`

```python
import tensorflow as tf
import numpy as np

(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

log_dir = "./logdir"
tensorboard_cb = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    profile_batch='1,2'
)

model.fit(x_train, y_train, epochs=3, callbacks=[tensorboard_cb])
```

---

### 4. tensorflow파일 실행

```bash
python3 train_tfboard.py
```

---

### 5. TensorBoard 실행

```bash
tensorboard --bind_all --logdir=./logdir --port=6006
```

브라우저에서 접속:

```
http://<EC2 Public DNS>:6006
```

---

## 🔍 주요 확인 포인트

| 탭          | 내용                                     |
| ---------- | -------------------------------------- |
| Scalars    | 손실 및 정확도 그래프                           |
| Graphs     | 모델 구조 시각화                              |
| Profile    | 실행 시간, 메모리 등 성능 분석                     |
| Histograms | 레이어별 weight 변화 (histogram\_freq=1일 때만) |

---

## 🗾 명령어 요약

```bash
# 가상환경 생성 및 실행
python3 -m venv venv
source venv/bin/activate

# 패키지 설치
pip install tensorflow tensorboard

# 파일 실행
python3 train_tfboard.py

# TensorBoard 실행
tensorboard --bind_all --logdir=./logdir --port=6006
```

---

## 📚 참고 자료
https://ddps.cloud/wiki/2022-09-22-Wiki-[kor]-How-to-use-tensorboard-in-ec2
## 🏁 추가 확장 아이디어

* Docker를 활용한 TensorFlow + TensorBoard 환경 구성
* GPU 인스턴스(g4dn)에서 성능 비교
* 시험 로그를 S3에 저장하고 다른 인스턴스에서 공유
