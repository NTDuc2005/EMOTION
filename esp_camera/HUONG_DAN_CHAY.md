# 🚀 HƯỚNG DẪN CHẠY ESP32 CAMERA APP

## 📋 BƯỚC 1: CHUẨN BỊ ESP32

### 1.1 Chuẩn Bị Phần Cứng
- ESP32-CAM (AI-Thinker)
- USB to Serial converter (FTDI / CP2102)
- Dây USB
- Jumper wires

### 1.2 Kết Nối Phần Cứng
```
ESP32-CAM          USB Serial Converter
─────────────────────────────────────
GND        ────────────  GND
U0T (TX)   ────────────  RX
U0R (RX)   ────────────  TX
5V         ────────────  5V
```

### 1.3 Cài Arduino IDE
1. Tải: https://www.arduino.cc/en/software
2. Cài đặt

---

## 💻 BƯỚC 2: CẤU HÌNH ARDUINO IDE

### 2.1 Thêm Board Manager
1. **File** → **Preferences**
2. Tìm: **Additional Boards Manager URLs**
3. Thêm URL:
   ```
   https://dl.espressif.com/dl/package_esp32_index.json
   ```
4. **OK**

### 2.2 Cài Board ESP32
1. **Tools** → **Board** → **Boards Manager**
2. Tìm: **esp32**
3. Cài: **ESP32 by Espressif Systems**

### 2.3 Cài Thư Viện Camera
1. **Sketch** → **Include Library** → **Manage Libraries**
2. Tìm: **esp32-camera**
3. Cài đặt

### 2.4 Cấu Hình Board
```
Tools → Board              : ESP32 Wrover Module
Tools → Upload Speed       : 115200
Tools → Flash Frequency    : 80 MHz
Tools → Flash Mode         : DIO
Tools → Partition Scheme   : Huge APP (3MB No OTA)
Tools → Port               : COM3 (hoặc COM của bạn)
```

---

## 📝 BƯỚC 3: UPLOAD CODE ESP32

### 3.1 Sửa WiFi
1. Mở: `D:\meetingmind-main\esp_camera\ESP32_Camera_Code.ino`
2. Tìm dòng ~20-21:
   ```cpp
   const char* ssid = "YOUR_SSID";        // ← Thay tên WiFi
   const char* password = "YOUR_PASSWORD"; // ← Thay mật khẩu
   ```
3. Ví dụ:
   ```cpp
   const char* ssid = "My_WiFi";
   const char* password = "12345678";
   ```

### 3.2 Upload Code
1. Copy toàn bộ code từ `ESP32_Camera_Code.ino`
2. Paste vào Arduino IDE
3. **Sketch** → **Upload** (Ctrl+U)
4. Chờ upload xong (khoảng 30-60 giây)

### 3.3 Kiểm Tra Kết Nối
1. **Tools** → **Serial Monitor** (Ctrl+Shift+M)
2. Chọn baud rate: **115200**
3. Xem output (chờ khoảng 5 giây):
   ```
   WiFi connected. IP address: 192.168.1.100
   HTTP server started
   Stream URL: http://192.168.1.100:8080/stream
   ```
4. **Ghi nhớ IP: `192.168.1.100`** ← Dùng cho app

---

## 📱 BƯỚC 4: CHẠY FLUTTER APP

### 4.1 Mở Terminal PowerShell
```powershell
# Mở PowerShell (Windows + R → powershell → Enter)
```

### 4.2 Vào Thư Mục App
```powershell
cd D:\meetingmind-main\esp_camera
```

### 4.3 Cài Dependencies
```powershell
flutter pub get
```

Chờ khoảng 2-3 phút tải xong.

### 4.4 Chạy App
```powershell
flutter run
```

Hoặc nếu có nhiều thiết bị:
```powershell
flutter devices                    # Liệt kê các thiết bị
flutter run -d <device_id>       # Chọn thiết bị
```

---

## 🎮 BƯỚC 5: SỬ DỤNG APP

### 5.1 Khi App Mở
1. Màn hình hiển thị: **"Cài đặt kết nối"**
2. Ô nhập: **Địa chỉ IP của ESP32**
3. Nhập: **`192.168.1.100`** (từ bước 3.3)

### 5.2 Kết Nối
1. Nhấn nút: **"Kết nối"**
2. Chờ 3-5 giây
3. Nếu thành công → SnackBar xanh: **"Kết nối ESP32 thành công!"**

### 5.3 Xem Camera
- Camera stream sẽ hiển thị
- Nếu phát hiện khuôn mặt → Hiển thị emotion (vui, buồn, v.v.)
- Gửi thông báo đến điện thoại

### 5.4 Ngắt Kết Nối
- Nhấn nút: **"Ngắt kết nối"**

---

## ⚠️ GỠ RỠ SỰ CỐ

### ❌ Vấn đề: Arduino IDE không nhận COM
**Giải pháp:**
```
1. Kiểm tra driver USB (Device Manager)
2. Cài đặt driver FTDI hoặc CH340
3. Plug out → Plug in lại
4. Thử COM khác
```

### ❌ Vấn đề: Upload không thành công
**Giải pháp:**
```
1. Chọn lại board: ESP32 Wrover Module
2. Chọn baud: 115200
3. Giữ nút GND trong 5 giây
4. Thử upload lại
```

### ❌ Vấn đề: ESP32 không kết nối WiFi
**Giải pháp:**
```
1. Kiểm tra SSID + Password có chính xác không
2. WiFi phải hỗ trợ 2.4GHz (không 5GHz)
3. Mở Serial Monitor xem lỗi
4. Reset ESP32 (nhấn nút RESET)
```

### ❌ Vấn đề: App không kết nối ESP32
**Giải pháp:**
```
1. Kiểm tra IP trong Serial Monitor (step 3.3)
2. Đảm bảo điện thoại cùng mạng WiFi với ESP32
3. Kiểm tra firewall cho phép port 8080
4. Thử ping từ PowerShell:
   ping 192.168.1.100
5. Nếu timeout → Kiểm tra kết nối WiFi
```

### ❌ Vấn đề: Không nhận được thông báo
**Giải pháp:**
```
1. Kiểm tra: Settings → Apps → Notifications
2. Cho phép thông báo cho app
3. Không để điện thoại ở chế độ im lặng
```

---

## 📊 KIỂM TRA NHANH

| Bước | Kiểm Tra | ✅ Đúng |
|------|---------|--------|
| 1 | Phần cứng kết nối | LED ESP32 sáng |
| 2 | Board cài | Arduino IDE nhận board ESP32 |
| 3 | WiFi đúng | Serial Monitor in IP |
| 4 | App chạy | Flutter app mở trên điện thoại |
| 5 | Kết nối | App kết nối → SnackBar xanh |

---

## 📞 HỖ TRỢ NHANH

**Nếu bị lỗi, kiểm tra:**

1. **Serial Monitor** (Arduino):
   ```
   Ctrl+Shift+M → Baud: 115200
   ```

2. **Flutter Logs** (Terminal):
   ```
   flutter logs
   ```

3. **IP ESP32:**
   - Mở Serial Monitor
   - Reset ESP32
   - Tìm dòng: `IP address: xxx.xxx.xxx.xxx`

4. **Test kết nối:**
   ```powershell
   ping 192.168.1.100
   ```

---

## ✅ HOÀN TẤT!

Khi mọi thứ hoạt động:
- ✅ Serial Monitor in: `HTTP server started`
- ✅ App hiển thị camera stream
- ✅ Phát hiện khuôn mặt và gửi thông báo

**Chúc mừng bạn! 🎉**

---

**Liên hệ nếu cần giúp đỡ!**
