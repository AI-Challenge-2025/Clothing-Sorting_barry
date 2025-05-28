# 👕 ชุดข้อมูลสำหรับการจำแนกประเภทเสื้อผ้า (Clothing Classification Dataset)

โปรเจกต์นี้เป็นชุดข้อมูลภาพสำหรับการ **จำแนกประเภทของเสื้อผ้า** เหมาะสำหรับนำไปใช้ในการฝึกโมเดล Machine Learning หรือ Deep Learning ด้านการรู้จำภาพ เช่น การจำแนกประเภทเสื้อผ้าในระบบอีคอมเมิร์ซ หรือแอปพลิเคชันแฟชั่น

## 📊 รายละเอียดของชุดข้อมูล

ในชุดข้อมูลนี้ประกอบด้วยประเภทของเสื้อผ้าและจำนวนรูปภาพในแต่ละคลาสดังนี้:

| ประเภทเสื้อผ้า    | จำนวนภาพ |
|--------------------|-----------|
| Tshirt             | 757       |
| Jacket             | 198       |
| Long Dress         | 146       |
| Long Skirt         | 115       |
| Midi Dress         | 132       |
| Midi Skirt         | 219       |
| Pants              | 495       |
| Shirt              | 148       |
| Short              | 130       |
| Short Dress        | 162       |
| Short Skirt        | 193       |
| Skirt              | 0         |
| Sweater            | 615       |

📌 **จำนวนภาพทั้งหมด**: 3,340 ภาพ  
📌 **จำนวนคลาสทั้งหมด**: 13 คลาส (ไม่นับ Skirt ที่มี 0 ภาพ)

> ⚠️ หมายเหตุ: คลาส `Skirt` ยังไม่มีข้อมูลภาพในเวอร์ชันปัจจุบัน

## 🌐 แหล่งที่มาของข้อมูล

ข้อมูลชุดนี้นำมาจาก Roboflow โดยสามารถเข้าถึงได้ที่:

👉 [https://app.roboflow.com/yhjyhjkukul/main-fashion-vysqe/overview](https://app.roboflow.com/yhjyhjkukul/main-fashion-vysqe/overview)

## 🧠 ตัวอย่างการใช้งาน

สามารถนำชุดข้อมูลนี้ไปใช้กับงานต่าง ๆ ได้ เช่น:

- ฝึกโมเดลสำหรับการจำแนกประเภทเสื้อผ้า
- ประยุกต์ใช้ในระบบรู้จำภาพ (Image Recognition)
- สร้างต้นแบบแอปพลิเคชันแฟชั่น

## 🚀 เริ่มต้นใช้งาน

1. เริ่มจากการ Run ทางด้านของ Python เพื่อรันในส่วนของ main.py
   ```bash
   python .\main.py

2. แล้วจากนั้นเพื่อตัวของ Server เพื่อใช้ในคำสั่ง Run server.js
   ```bash
   node .\server.js
3. จากนั้นเปิดหน้า Web แล้วเปิดด้วย ip เครื่องของตัวเอง หรือจะเป็น Localhost ก็ได้
   ```bash
   http://127.0.0.1:8004/static/index.html
<h2 align="center">🎯 ตัวอย่างการทำงานของระบบจำแนกเสื้อผ้าอัตโนมัติด้วย AI</h2>

<p align="center">
  <img src="https://raw.githubusercontent.com/AI-Challenge-2025/Clothing-Sorting_barry/main/Image/messageImage_1748454386903.jpg" width="45%" style="margin: 5px;" />
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/AI-Challenge-2025/Clothing-Sorting_barry/main/Image/messageImage_1748454584649.jpg" width="45%" style="margin: 5px;" />
</p>


- จากนั้นบนหน้าเว็ปจะ Detect เสื้อผ่านวิดิโอที่เราเปิดอยู่พร้อมมีกรอบวัดความแม่นยำ
- แสดงค่าความแม่นยำจาก วิดิโอจากใต้ภาพ
- ปุ่ม Capture ในการถ่ายภาพขณะที่ Detect อยู่
- option เสริม สามารถเปิดปิด Dark/Light Mode
- 3.1. เปิดในส่วนอีกหน้าหนึ่งผ่าน Hamburger Menu (☰) หรือ เปิดจากอันนี้
   ```bash
   http://127.0.0.1:8004/static/index2.html
<p align="center">
  <img src="https://raw.githubusercontent.com/AI-Challenge-2025/Clothing-Sorting_barry/main/Image/messageImage_1748454909884.jpg" width="45%" style="margin: 5px;" />
</p>
   - จะแสดงในส่วนของการเพิ่มรูป หรือ อัปโหลดรูป เพื่อทำนาย
   - แสดงค่าความแม่นยำจาก รูปจากใต้ภาพ
   - option เสริม สามารถเปิดปิด Dark/Light Mode




