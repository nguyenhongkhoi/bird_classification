# 🦜 Bird Sound Classification (BirdCLEF 2024)

Dự án nghiên cứu và thử nghiệm phân loại tiếng hót của các loài chim dựa trên tập dữ liệu thi đấu **BirdCLEF 2024** từ Kaggle. Pipeline bao gồm các bước: tải dữ liệu, trích xuất đặc trưng âm thanh **Log-Mel Spectrogram**, huấn luyện và so sánh hiệu năng giữa các mô hình Machine Learning cổ điển và Deep Learning (Random Forest, KNN, Scikit-learn MLP, PyTorch MLP, và PyTorch CRNN).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nguyenhongkhoi/bird_classification/blob/main/bird_classification.ipynb)

---

## 📌 Bảng so sánh phương thức chạy

| Tiêu chí | 🚀 Chạy trên Google Colab (**Khuyến khích**) | 💻 Chạy trên máy cá nhân (**Không khuyến khích**) |
| :--- | :--- | :--- |
| **Dung lượng ổ cứng** | Dùng hoàn toàn trên ổ cứng ảo của Google (~100GB tạm thời) | Tốn **> 20GB** dung lượng SSD/HDD của máy |
| **Tốc độ tải dữ liệu** | Siêu nhanh (~1 - 2 phút) nhờ băng thông đám mây | Lâu (từ 15 phút đến hàng giờ tùy mạng) |
| **Tài nguyên tính toán** | Có sẵn **GPU miễn phí (T4 GPU)** | Phụ thuộc cấu hình máy (chạy CPU sẽ rất lâu) |
| **Cài đặt môi trường** | Môi trường Linux chuẩn hóa, có sẵn `unzip`, `librosa`, PyTorch | Phải cài Python, virtualenv, CUDA, FFmpeg đọc audio |

---

## 🚀 Cách 1: Hướng dẫn chạy trên Google Colab (Khuyến khích)

Đây là phương pháp **tối ưu và nhanh nhất** để chạy thử nghiệm và huấn luyện mô hình.

### Bước 1: Mở Notebook trên Colab
- Nhấp vào huy hiệu **[Open In Colab](https://colab.research.google.com/github/nguyenhongkhoi/bird_classification/blob/main/bird_classification.ipynb)** ở trên.
- Hoặc truy cập [Google Colab](https://colab.research.google.com/), chọn tab **Upload** và tải lên file `bird_classification.ipynb`.

### Bước 2: Bật Tăng tốc phần cứng (GPU)
1. Trên thanh menu Colab, chọn **Runtime** -> **Change runtime type** (Thời lượng -> Thay đổi loại thời lượng).
2. Tại mục **Hardware accelerator** (Bộ tăng tốc phần cứng), chọn **T4 GPU**.
3. Bấm **Save** (Lưu).

### Bước 3: Chuẩn bị tài khoản Kaggle
Tập dữ liệu thuộc cuộc thi **BirdCLEF 2024**. Để tải được dữ liệu:
1. Đăng nhập vào Kaggle và truy cập trang cuộc thi: [BirdCLEF 2024 Competition](https://www.kaggle.com/competitions/birdclef-2024).
2. Nhấn **"Understand and Accept"** / **"Join Competition"** để đồng ý với điều khoản cuộc thi (bắt buộc, nếu chưa chấp thuận thì API sẽ báo lỗi 403 Forbidden).
3. Trong notebook đã có sẵn cấu hình token Kaggle tại Cell 1. Nếu bạn muốn sử dụng token Kaggle của riêng mình:
   - Vào Kaggle Settings -> mục **API** -> nhấn **Create New Token** (hoặc copy giá trị API Token).
   - Thay thế giá trị `os.environ["KAGGLE_API_TOKEN"]` tại Cell 1 bằng token của bạn.

### Bước 4: Chạy Notebook
- Chọn **Runtime** -> **Run all** (hoặc nhấn tổ hợp phím `Ctrl + F9`) để chạy lần lượt từ trên xuống dưới.
- Cell 1 sẽ tự động tải file zip, chỉ giải nén thư mục `train_audio/` và tự động xóa file `.zip` để tiết kiệm bộ nhớ đệm của Colab.
- Các cell tiếp theo sẽ trích xuất đặc trưng Mel Spectrogram và huấn luyện các mô hình.

---

## ⚠️ Cách 2: Hướng dẫn chạy trên máy cá nhân (Local) (Không khuyến khích)

> **CẢNH BÁO:** 
> - Bộ dữ liệu `birdclef-2024.zip` có dung lượng nén khoảng **15 - 20 GB**. Sau khi giải nén, dung lượng có thể chiếm thêm hàng chục GB trên ổ cứng.
> - Quá trình trích xuất âm thanh từ hàng nghìn file `.ogg` và huấn luyện mô hình BiLSTM / CRNN trên CPU sẽ rất chậm nếu máy không có card đồ họa rời NVIDIA (hỗ trợ CUDA).
> - Chỉ nên làm theo cách này nếu bạn có đường truyền mạng mạnh, ổ cứng còn trống tối thiểu **40GB** và cấu hình máy đủ khỏe.

### Bước 1: Chuẩn bị môi trường & Clone mã nguồn
Clone dự án về máy:
```bash
git clone https://github.com/nguyenhongkhoi/bird_classification.git
cd bird_classification
```

Tạo và kích hoạt môi trường ảo (khuyến nghị Python 3.10 - 3.11):
- **Trên Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- **Trên Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Bước 2: Cài đặt các thư viện cần thiết
Cài đặt từ file `requirements.txt`:
```bash
pip install -r requirements.txt
```

> **Lưu ý về PyTorch & GPU:** 
> Nếu máy bạn có card đồ họa NVIDIA, hãy cài đặt phiên bản PyTorch hỗ trợ CUDA theo hướng dẫn chính thức tại [pytorch.org](https://pytorch.org/get-started/locally/) để tận dụng GPU thay vì chạy bằng CPU.
> 
> **Lưu ý về Audio Backend:**
> Thư viện `librosa` cần backend để đọc các định dạng âm thanh `.ogg`. Trên Windows, `soundfile` thường đã đi kèm thư viện nén; trên Linux nếu gặp lỗi đọc audio, bạn hãy cài thêm:
> ```bash
> sudo apt-get update && sudo apt-get install -y ffmpeg libsndfile1
> ```

### Bước 3: Cấu hình Kaggle API trên máy
1. Truy cập [Kaggle](https://www.kaggle.com/) -> Profile -> **Settings** -> nhấn **Create New Token** để tải về file `kaggle.json`.
2. Đặt file `kaggle.json` vào thư mục người dùng:
   - **Windows:** `C:\Users\<Tên_User>\.kaggle\kaggle.json`
   - **Linux / macOS:** `~/.kaggle/kaggle.json` (sau đó chạy `chmod 600 ~/.kaggle/kaggle.json`)
3. Đảm bảo đã bấm chấp thuận điều khoản tại cuộc thi [BirdCLEF 2024](https://www.kaggle.com/competitions/birdclef-2024).

### Bước 4: Lưu ý về Cell 1 khi chạy trên Windows
Trong Cell 1 của notebook có sử dụng các lệnh shell Linux như `!unzip` và `!rm`:
- Nếu bạn chạy trên **Windows**, các lệnh `!unzip` và `!rm` có thể báo lỗi `command not found`.
- Bạn có thể tải và giải nén thủ công bằng lệnh terminal:
  ```powershell
  kaggle competitions download -c birdclef-2024
  # Dùng phần mềm 7-Zip hoặc PowerShell để giải nén thư mục train_audio vào folder dataset/train_audio:
  Expand-Archive -Path birdclef-2024.zip -DestinationPath dataset
  # Sau đó xóa file zip để tiết kiệm dung lượng:
  Remove-Item birdclef-2024.zip
  ```
  Hoặc thay thế nội dung Cell 1 bằng script giải nén bằng Python (`zipfile`).

### Bước 5: Khởi động Jupyter Notebook
Khởi chạy Jupyter Notebook hoặc mở trực tiếp trên VS Code:
```bash
jupyter notebook bird_classification.ipynb
```
Chạy tuần tự các cell từ dữ liệu đến các bước huấn luyện.

---

## 🧠 Các mô hình được thử nghiệm

1. **Random Forest (Scikit-Learn)**: Mô hình học máy dựa trên rừng cây quyết định, hoạt động trên vector đặc trưng Spectrogram phẳng (Flattened).
2. **K-Nearest Neighbors - KNN (Scikit-Learn)**: Phân loại theo lân cận gần nhất ($k=5$).
3. **Multi-Layer Perceptron (Scikit-Learn MLP)**: Mạng neural truyền thẳng đa tầng cơ bản.
4. **PyTorch MLP**: Mạng nơ-ron sâu với các kỹ thuật Batch Normalization, Dropout và AdamW optimizer.
5. **PyTorch CRNN (CNN 2D + BiLSTM)**: Kiến trúc nâng cao kết hợp mạng tích chập 2D (trích xuất đặc trưng không gian phổ Mel) và mạng hồi quy 2 chiều Bidirectional LSTM (nắm bắt diễn biến tần số theo trục thời gian).

---

## 📁 Cấu trúc thư mục

```text
bird_classification/
├── bird_classification.ipynb   # Jupyter Notebook chứa toàn bộ pipeline
├── requirements.txt            # Danh sách các thư viện cần thiết
├── README.md                   # Tài liệu hướng dẫn sử dụng
└── dataset/                    # Thư mục chứa dữ liệu sau khi giải nén (tự động tạo)
    └── train_audio/            # Thư mục âm thanh gốc theo từng loài chim
```