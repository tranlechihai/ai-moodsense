# 😊 AI MoodSense – Student Facial Emotion Classification Web App

AI MoodSense là một ứng dụng web xây dựng bằng Streamlit, dùng để nhận diện cảm xúc khuôn mặt từ ảnh hoặc webcam bằng cách sử dụng mô hình CNN được học tại khóa lập trình Computer Science tại GearGen. 

Các cảm xúc được ánh xạ vào ba nhóm:
- **Positive (Tích cực)**: Happy, Surprise  
- **Neutral (Trung lập)**: Neutral  
- **Negative (Tiêu cực)**: Sad, Angry, Fear, Disgust

## ✨ Những gì có trong phiên bản hiện tại
- Nhận diện một hoặc nhiều khuôn mặt, hiển thị bounding box cho từng khuôn mặt.
- Thẻ kết quả cho từng khuôn mặt, hiển thị class cảm xúc + độ tin cậy, kèm biểu đồ xác suất 3 lớp.
- Xuất file CSV kết quả theo từng khuôn mặt (cả khi upload ảnh và dùng webcam).
- Debug mode để xem raw emotion scores từ DeepFace.
- Tùy chọn auto-detect khuôn mặt; nếu tắt, hệ thống sẽ phân tích toàn bộ frame.

## 📁 Cấu trúc dự án
```
ai-sample/
├── app.py                 # Streamlit app
├── requirements.txt       # Dependencies (Streamlit, DeepFace, OpenCV, etc.)
├── README.md
├── data/
│   ├── logo_geargen.png   # Page logo 
│   └── logo_geargen.ico   # Favicon 
├── notebooks/
│   └── train_model.ipynb  # Optional legacy training notebook
└── src/
    ├── data_processing.py # FER2013 
    └── model_utils.py     # DeepFace wrapper and mapping logic
```

## 🚀 Cài đặt (chạy local)
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
streamlit run app.py
```
The app runs at `http://localhost:8501`.


## 🖥️ Cách sử dụng ứng dụng
1. Mở ứng dụng và chọn **Upload Image** hoặc **Webcam Capture**.
2. Bật **Detect multiple faces** nếu ảnh có nhiều người.
3. (Không bắt buộc) Bật **Debug mode** để xem raw output từ DeepFace.
4. Tải kết quả dự đoán theo từng khuôn mặt bằng các nút **CSV**.

## ❓ FAQ / Tips
- Không phát hiện được khuôn mặt? → Tắt auto-detect và thử lại, hoặc dùng ảnh rõ nét, nhìn thẳng.
- Surprise bị nhận nhầm thành Fear? → Bật “Detect multiple faces”, đảm bảo ánh sáng tốt;

## 📝 Ghi chú
- Ứng dụng sử dụng DeepFace pre-trained
- Notebook train cũ để tham khảo và thử nghiệm.

