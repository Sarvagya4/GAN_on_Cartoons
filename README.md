# 🎨 GAN on Cartoons

Generative Adversarial Networks (GANs) for creating and understanding **cartoon-style images**!  
This project explores how **deep learning** can be used to generate, train, and visualize cartoon images using a **GAN-based architecture**.

---

## 🧠 Project Overview

This repository implements a **Generative Adversarial Network (GAN)** that learns to produce cartoon-like images from noise.  
GANs consist of two neural networks — a **Generator** 🪄 and a **Discriminator** 🕵️ — that compete against each other:

- **Generator:** Creates fake cartoon images that look as real as possible.  
- **Discriminator:** Distinguishes between real and generated images.  

Through training, both networks improve until the generator produces images that are almost indistinguishable from the real cartoon dataset.

---

## 📚 Features

✨ Implemented from scratch using **TensorFlow/Keras**  
🖼️ Trains on a **Cartoon Faces Dataset** (or any similar image dataset)  
⚙️ Customizable model architecture and hyperparameters  
📈 Visualization of **training progress** and **generated samples**  
💾 Easy to reproduce and extend for your own cartoon datasets  

---

## 🧩 Project Structure

```
GAN_on_Cartoons/
├── data/                   # Dataset folder (cartoon images)
├── models/                 # Model definitions (Generator, Discriminator)
├── notebooks/              # Jupyter notebooks for exploration
├── results/                # Generated images and training results
├── train.py                # Main training script
├── utils.py                # Helper functions
└── README.md               # Project documentation
```

---

## 🚀 Getting Started

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Sarvagya4/GAN_on_Cartoons.git
cd GAN_on_Cartoons
```

### 2️⃣ Install dependencies
Make sure you have **Python 3.8+** and install the required libraries:
```bash
pip install -r requirements.txt
```

### 3️⃣ Prepare the dataset
Download or place your cartoon images inside the `data/` folder.  
You can use datasets like [Cartoon Faces Dataset](https://www.kaggle.com/datasets) or your own collection.

### 4️⃣ Train the GAN
```bash
python train.py
```

### 5️⃣ View Results
Generated images and training progress will be saved in the `results/` directory.

---

## 📊 Results & Visualization

During training, the model gradually improves its ability to generate high-quality cartoon faces.  
Sample outputs include:

| Epoch | Generated Image |
|-------|------------------|
| 10    | ![sample1](results/sample_epoch_10.png) |
| 50    | ![sample2](results/sample_epoch_50.png) |
| 100   | ![sample3](results/sample_epoch_100.png) |

> The more epochs you train, the more expressive and realistic your cartoon generations become!

---

## 🧰 Technologies Used

- 🐍 **Python 3.8+**  
- 🧠 **TensorFlow / Keras**  
- 📊 **NumPy**, **Matplotlib**, **Pandas**  
- 🖼️ **OpenCV / PIL** for image preprocessing  

---

## 💡 Future Improvements

- ✅ Conditional GANs for style-based cartoon generation  
- 🎭 Transfer learning using pre-trained models  
- 🌈 Interactive demo with Streamlit or Gradio  
- 🔥 Optimization for faster convergence  

---

## 🤝 Contributing

Contributions are welcome!  
If you'd like to enhance this project:
1. Fork the repo  
2. Create a new branch (`feature/my-feature`)  
3. Commit your changes  
4. Push the branch and open a **Pull Request**

---

## 🌟 Acknowledgements

- Inspired by **Ian Goodfellow’s** original GAN paper.  
- Datasets sourced from **Kaggle** and **Cartoon Faces Dataset**.  
- Thanks to the **TensorFlow** and **Keras** communities for their amazing tools.

