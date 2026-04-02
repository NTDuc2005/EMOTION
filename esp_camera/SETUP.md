# 📱 ESP32 Camera App - Hướng Dẫn Cài Đặt

## 🎯 Tính Năng
- ✅ Xem camera từ ESP32 qua WiFi
- ✅ Nhận dạng khuôn mặt in real-time
- ✅ Phát hiện biểu cảm (vui, buồn, tức giận, ngạc nhiên)
- ✅ Gửi thông báo khi phát hiện khuôn mặt
- ✅ Điều khiển camera từ ứng dụng

---

## 🔧 PHẦN I: CẤU HÌNH ESP32

### 1. Chuẩn Bị Phần Cứng
- 1x ESP32-CAM (AI-Thinker)
- 1x USB to Serial converter
- Dây USB và jumper wires

### 2. Upload Code Arduino
1. **Mở Arduino IDE**
2. **Cài đặt Board:**
   - File → Preferences → Additional Boards Manager URLs
   - Thêm: `https://dl.espressif.com/dl/package_esp32_index.json`
   - Tools → Board Manager → Tìm "esp32" → Cài

3. **Cài đặt thư viện:**
   - Sketch → Include Library → Manage Libraries
   - Tìm "esp32-camera" → Cài đặt

4. **Sửa Code:**
   ```cpp
   const char* ssid = "TEN_WIFI_CUA_BAN";
   const char* password = "MAT_KHAU_WIFI";
   ```

5. **Upload:** Nhấn Upload (Ctrl+U)

---

## 📱 PHẦN II: CẤU HÌNH FLUTTER APP

### 1. Cài Đặt Dependencies
```bash
cd esp_camera
flutter pub get
```

### 2. Cấu Hình Android
**File: android/app/src/AndroidManifest.xml**
```xml
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
```

### 3. Cấu Hình iOS
**File: ios/Runner/Info.plist**
```xml
<key>NSCameraUsageDescription</key>
<string>Ứng dụng cần quyền truy cập camera</string>
<key>NSLocalNetworkUsageDescription</key>
<string>Ứng dụng cần kết nối với ESP32</string>
```

### 4. Cập Nhật IP ESP32
**File: lib/camera_module.dart - Dòng ~25**
```dart
static const String defaultIPAddress = 'XXX.XXX.XXX.XXX'; // IP của ESP32
```

---

## 🚀 CHẠY APP

```bash
flutter run
```

---

## 📁 Cấu Trúc Thư Mục
```
esp_camera/
├── pubspec.yaml              # Dependencies
├── lib/
│   ├── main.dart            # Entry point
│   └── camera_module.dart   # Tất cả code (Config, Services, Provider, Screen)
├── android/
│   └── app/src/AndroidManifest.xml
├── ios/
│   └── Runner/Info.plist
└── ESP32_Camera_Code.ino    # Code Arduino cho ESP32
```

---

## 🔗 API Endpoints ESP32

| Endpoint | Mô Tả |
|----------|-------|
| `/capture` | Lấy ảnh snapshot |
| `/stream` | Xem stream MJPEG |
| `/status` | Kiểm tra trạng thái |
| `/cmd?command=brightness&value=1` | Điều chỉnh độ sáng |

---

## ⚠️ Khắc Phục Sự Cố

**Không kết nối được ESP32:**
- Kiểm tra IP có chính xác không
- Đảm bảo cùng mạng WiFi
- Kiểm tra firewall cho phép port 8080

**Không nhận được thông báo:**
- Kiểm tra quyền thông báo trên điện thoại
- Chắc chắn `enableFaceDetectionNotification = true`

**Face detection không hoạt động:**
- Đảm bảo đủ ánh sáng
- Khuôn mặt phải đủ lớn trong ảnh

---

## 📞 Hỗ Trợ

Mở terminal và chạy:
```bash
flutter logs
```

Để xem logcat của app.

---

**Chúc bạn thành công! 🎉**
