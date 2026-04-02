#include <esp32-camera.h>
#include <WiFi.h>
#include <WebServer.h>

// ===== CẤU HÌNH CAMERA =====
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// ===== CẤU HÌNH WiFi =====
const char* ssid = "YOUR_SSID";
const char* password = "YOUR_PASSWORD";

WebServer server(8080);

void initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sda = SIOD_GPIO_NUM;
  config.pin_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_VGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  if (esp_camera_init(&config) != ESP_OK) {
    Serial.println("Camera init failed");
    return;
  }

  sensor_t * s = esp_camera_sensor_get();
  if (s->id.PID == OV2640_PID) {
    s->set_framesize(s, FRAMESIZE_VGA);
  }

  Serial.println("Camera initialized successfully");
}

void initWiFi() {
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("");
    Serial.print("WiFi connected. IP address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("Failed to connect to WiFi");
  }
}

void handleStream() {
  WiFiClient client = server.client();
  String response = "HTTP/1.1 200 OK\r\n";
  response += "Content-Type: multipart/x-mixed-replace; boundary=123456789000000000000987654321\r\n\r\n";
  server.sendContent(response);

  while (client.connected()) {
    camera_fb_t * fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed");
      break;
    }

    String part = "--123456789000000000000987654321\r\n";
    part += "Content-Type: image/jpeg\r\n";
    part += "Content-Length: " + String(fb->len) + "\r\n\r\n";

    server.sendContent(part);
    server.sendContent_P((const uint8_t *)fb->buf, fb->len);
    server.sendContent("\r\n");

    esp_camera_fb_return(fb);

    if (!client.connected()) break;
    delay(30);
  }
}

void handleCapture() {
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    server.send(500, "text/plain", "Camera capture failed");
    return;
  }

  server.sendHeader("Content-Disposition", "inline; filename=capture.jpg");
  server.send_P(200, "image/jpeg", (const uint8_t *)fb->buf, fb->len);
  esp_camera_fb_return(fb);
}

void handleStatus() {
  String json = "{";
  json += "\"ready\":true,";
  json += "\"wifi\":{";
  json += "\"connected\":" + String(WiFi.status() == WL_CONNECTED ? "true" : "false") + ",";
  json += "\"signal\":" + String(WiFi.RSSI()) + ",";
  json += "\"ip\":\"" + WiFi.localIP().toString() + "\"";
  json += "},";
  json += "\"camera\":{";
  json += "\"resolution\":\"VGA\",";
  json += "\"quality\":\"JPEG\"";
  json += "}";
  json += "}";

  server.sendHeader("Content-Type", "application/json");
  server.send(200, "application/json", json);
}

void handleCommand() {
  if (!server.hasArg("command")) {
    server.send(400, "text/plain", "Missing command parameter");
    return;
  }

  String command = server.arg("command");
  sensor_t * s = esp_camera_sensor_get();

  if (command == "brightness") {
    if (server.hasArg("value")) {
      int value = server.arg("value").toInt();
      s->set_brightness(s, value);
    }
  } else if (command == "contrast") {
    if (server.hasArg("value")) {
      int value = server.arg("value").toInt();
      s->set_contrast(s, value);
    }
  } else if (command == "saturation") {
    if (server.hasArg("value")) {
      int value = server.arg("value").toInt();
      s->set_saturation(s, value);
    }
  } else if (command == "vflip") {
    if (server.hasArg("value")) {
      int value = server.arg("value").toInt();
      s->set_vflip(s, value);
    }
  } else if (command == "hflip") {
    if (server.hasArg("value")) {
      int value = server.arg("value").toInt();
      s->set_hmirror(s, value);
    }
  }

  server.send(200, "text/plain", "OK");
}

void setup() {
  Serial.begin(115200);
  delay(2000);
  Serial.println("\n\nESP32 Camera Web Server");
  Serial.println("=======================");

  initCamera();
  initWiFi();

  server.on("/", handleCapture);
  server.on("/capture", handleCapture);
  server.on("/stream", handleStream);
  server.on("/status", handleStatus);
  server.on("/cmd", handleCommand);

  server.begin();
  Serial.println("HTTP server started");
  Serial.print("Stream URL: http://");
  Serial.print(WiFi.localIP());
  Serial.println(":8080/stream");
}

void loop() {
  server.handleClient();
  delay(2);
}
