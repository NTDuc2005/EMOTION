#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"

// ================= CAMERA MODEL =================
#define CAMERA_MODEL_AI_THINKER
#include "camera_pins.h"

// ================= WIFI =================
const char* ssid = "VanTam";
const char* password = "0937534540";

// ================= VC02 UART =================
HardwareSerial VC02(2);   // UART2
String vc02Buffer = "";


// ================= CAMERA SERVER =================
void startCameraServer();

// ================= SETUP =================
void setup() {
  Serial.begin(115200);
  Serial.println("\nESP32-CAM START");

  // ===== VC02 UART =====
  VC02.begin(9600, SERIAL_8N1, 15, 14); // RX, TX
  Serial.println("VC02 UART READY");

  // ===== Camera config =====
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;

  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // 🔥 QUAN TRỌNG: MƯỢT NHẤT
  config.frame_size   = FRAMESIZE_QVGA;   // 320x240
  config.jpeg_quality = 15;
  config.fb_count     = 2;
  config.grab_mode    = CAMERA_GRAB_LATEST;
  config.fb_location  = CAMERA_FB_IN_PSRAM;

  // ===== Init camera =====
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return;
  }

  sensor_t * s = esp_camera_sensor_get();
  s->set_framesize(s, FRAMESIZE_QVGA);
  s->set_quality(s, 15);

  Serial.println("Camera init OK");

  // ===== WiFi =====
  WiFi.begin(ssid, password);
  WiFi.setSleep(false);

  Serial.print("WiFi connecting");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi connected");
  Serial.print("Stream URL: http://");
  Serial.print(WiFi.localIP());
  Serial.println(":81/stream");

  startCameraServer();
}

// ================= LOOP =================
void loop() {
  // Đọc dữ liệu VC02
  while (VC02.available()) {
    char c = VC02.read();
    if (c == '\n') {
      handleVC02(vc02Buffer);
      vc02Buffer = "";
    } else {
      vc02Buffer += c;
    }
  }
}

// ================= VC02 HANDLER =================
void handleVC02(String data) {
  data.trim();
  if (data.length() == 0) return;

  // In JSON để Python đọc
  if (data == "PERSON") {
    Serial.println("{\"object\":\"person\",\"confidence\":0.95}");
  }
  else if (data == "NONE") {
    Serial.println("{\"object\":\"none\",\"confidence\":0.0}");
  }
  else {
    Serial.print("{\"vc02\":\"");
    Serial.print(data);
    Serial.println("\"}");
  }
}