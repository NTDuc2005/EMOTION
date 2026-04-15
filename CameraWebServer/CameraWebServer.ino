#include "esp_camera.h"
#include <WiFi.h>
#include <ESP32Servo.h>
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

// ================= SERVO PAN / TILT =================
#define SERVO_PAN_PIN 12
#define SERVO_TILT_PIN 13

Servo servoPan;
Servo servoTilt;
int panAngle = 90;
int tiltAngle = 90;

void setPan(int angle) {
  panAngle = constrain(angle, 0, 180);
  servoPan.write(panAngle);
}

void setTilt(int angle) {
  tiltAngle = constrain(angle, 0, 180);
  servoTilt.write(tiltAngle);
}

// ================= CAMERA SERVER =================
void startCameraServer();

// ================= CAMERA PROFILE FOR FACE RECOGNITION =================
static const framesize_t FACE_STREAM_SIZE = FRAMESIZE_VGA;   // 640x480
static const framesize_t FACE_FALLBACK_SIZE = FRAMESIZE_CIF; // 400x296
static const int FACE_JPEG_QUALITY = 10;                     // lower = better image

void applyFaceRecognitionProfile(sensor_t *s) {
  if (!s) return;
  s->set_framesize(s, FACE_STREAM_SIZE);
  s->set_quality(s, FACE_JPEG_QUALITY);
  s->set_brightness(s, 1);
  s->set_contrast(s, 1);
  s->set_saturation(s, 0);
  s->set_special_effect(s, 0);
  s->set_whitebal(s, 1);
  s->set_awb_gain(s, 1);
  s->set_wb_mode(s, 0);
  s->set_exposure_ctrl(s, 1);
  s->set_aec2(s, 1);
  s->set_ae_level(s, 0);
  s->set_gain_ctrl(s, 1);
  s->set_agc_gain(s, 0);
  s->set_gainceiling(s, (gainceiling_t)2);
  s->set_bpc(s, 1);
  s->set_wpc(s, 1);
  s->set_raw_gma(s, 1);
  s->set_lenc(s, 1);
  s->set_hmirror(s, 0);
  s->set_vflip(s, 0);
  s->set_dcw(s, 1);
}

// ================= SETUP =================
void setup() {
  Serial.begin(115200);
  Serial.println("\nESP32-CAM START");

  // ===== Servo setup =====
  servoPan.setPeriodHertz(50);
  servoTilt.setPeriodHertz(50);
  servoPan.attach(SERVO_PAN_PIN, 500, 2500);
  servoTilt.attach(SERVO_TILT_PIN, 500, 2500);
  setPan(panAngle);
  setTilt(tiltAngle);

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
  config.frame_size   = FACE_STREAM_SIZE;
  config.jpeg_quality = FACE_JPEG_QUALITY;
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
  if (!psramFound()) {
    s->set_framesize(s, FACE_FALLBACK_SIZE);
    s->set_quality(s, 12);
  }
  applyFaceRecognitionProfile(s);
  if (!psramFound()) {
    s->set_framesize(s, FACE_FALLBACK_SIZE);
  }

  Serial.println("Camera init OK");
  Serial.println("Face-recognition profile enabled");

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
  Serial.println("Control URL: /control?var=pan&val=90 or /control?var=tilt&val=90");

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