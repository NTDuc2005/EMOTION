import 'dart:io';
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:connectivity_plus/connectivity_plus.dart';

// ============= CONFIG =============
class ESP32Config {
  static const String defaultIPAddress = '192.168.1.47';
  static const int defaultPort = 81;
  static const String streamPath = '/stream';
  static const int cameraResolutionWidth = 640;
  static const int cameraResolutionHeight = 480;
  static const double faceDetectionConfidenceThreshold = 0.7;
  static const int minFaceDetectionSize = 50;
  static const int notificationCooldownSeconds = 5;
  static const bool enableFaceDetectionNotification = true;
}

// ============= MODELS =============
class FaceDetectionData {
  final String id;
  final DateTime detectedAt;
  final double confidence;
  final String emotion;
  final double emotionConfidence;
  final String imagePath;
  final bool notificationSent;
  final String? notificationMessage;

  FaceDetectionData({
    required this.id,
    required this.detectedAt,
    required this.confidence,
    required this.emotion,
    required this.emotionConfidence,
    required this.imagePath,
    required this.notificationSent,
    this.notificationMessage,
  });

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'detectedAt': detectedAt.toIso8601String(),
      'confidence': confidence,
      'emotion': emotion,
      'emotionConfidence': emotionConfidence,
      'imagePath': imagePath,
      'notificationSent': notificationSent,
      'notificationMessage': notificationMessage,
    };
  }

  factory FaceDetectionData.fromJson(Map<String, dynamic> json) {
    return FaceDetectionData(
      id: json['id'] ?? '',
      detectedAt: DateTime.parse(json['detectedAt'] ?? DateTime.now().toIso8601String()),
      confidence: (json['confidence'] ?? 0.0).toDouble(),
      emotion: json['emotion'] ?? 'neutral',
      emotionConfidence: (json['emotionConfidence'] ?? 0.0).toDouble(),
      imagePath: json['imagePath'] ?? '',
      notificationSent: json['notificationSent'] ?? false,
      notificationMessage: json['notificationMessage'],
    );
  }
}

// ============= SERVICES - ESP32 Camera Service =============
class ESP32CameraService {
  static final ESP32CameraService _instance = ESP32CameraService._internal();
  factory ESP32CameraService() => _instance;
  ESP32CameraService._internal();

  HttpClient? _httpClient;
  String _esp32IP = '192.168.1.100';
  int _esp32Port = 8080;
  bool _isConnected = false;

  String get streamUrl => 'http://$_esp32IP:$_esp32Port${ESP32Config.streamPath}';

  bool get isConnected => _isConnected;
  String get esp32IP => _esp32IP;
  String? get streamUrl => _httpClient != null && _isConnected ? 'http://$_esp32IP:$_esp32Port${ESP32Config.streamPath}' : null;

  Future<bool> connectToESP32({
    required String ipAddress,
    int port = 8080,
  }) async {
    try {
      _esp32IP = ipAddress;
      _esp32Port = port;

      final connectivityResult = await Connectivity().checkConnectivity();
      if (connectivityResult == ConnectivityResult.none) {
        print('No internet connection');
        return false;
      }

      _httpClient = HttpClient();
      _httpClient!.connectionTimeout = const Duration(seconds: 5);

      try {
        final request = await _httpClient!.get(_esp32IP, _esp32Port, ESP32Config.streamPath);
        final response = await request.close();

          if (response.statusCode == 200) {
          _isConnected = true;
          print('Connected to ESP32 at $_esp32IP:$_esp32Port');
          return true;
        }
      } catch (e) {
        print('Failed to connect to ESP32: $e');
        _isConnected = false;
        return false;
      }

      return false;
    } catch (e) {
      print('Error in connectToESP32: $e');
      _isConnected = false;
      return false;
    }
  }

  Future<List<int>?> getSnapshotFromESP32() async {
    try {
      if (!_isConnected) return null;
      final request = await _httpClient!.get(_esp32IP, _esp32Port, '/capture');
      final response = await request.close();
      if (response.statusCode == 200) {
        return await response.expand((chunk) => chunk).toList();
      }
      return null;
    } catch (e) {
      print('Error getting snapshot: $e');
      return null;
    }
  }

  Future<bool> sendCommandToESP32({
    required String command,
    Map<String, String>? parameters,
  }) async {
    try {
      if (!_isConnected) return false;
      String url = '/cmd?command=$command';
      if (parameters != null) {
        parameters.forEach((key, value) {
          url += '&$key=$value';
        });
      }
      final request = await _httpClient!.get(_esp32IP, _esp32Port, url);
      final response = await request.close();
      return response.statusCode == 200;
    } catch (e) {
      print('Error sending command: $e');
      return false;
    }
  }

  void disconnect() {
    _httpClient?.close(force: true);
    _isConnected = false;
  }
}

// ============= SERVICES - Face Detection Service =============
class FaceDetectionService {
  static final FaceDetectionService _instance = FaceDetectionService._internal();
  factory FaceDetectionService() => _instance;
  FaceDetectionService._internal();

  late FaceDetector _faceDetector;
  DateTime? _lastNotificationTime;

  Future<void> initialize() async {
    final options = FaceDetectorOptions(
      enableClassification: true,
      enableLandmarks: true,
      enableContours: true,
      enableTracking: true,
    );
    _faceDetector = FaceDetector(options: options);
  }

  Future<List<Face>> detectFaces(InputImage inputImage) async {
    try {
      final faces = await _faceDetector.processImage(inputImage);
      return faces;
    } catch (e) {
      print('Error detecting faces: $e');
      return [];
    }
  }

  String analyzeEmotion(Face face) {
    final smilingProbability = face.smilingProbability ?? 0.0;
    final leftEyeOpenProbability = face.leftEyeOpenProbability ?? 0.0;
    final rightEyeOpenProbability = face.rightEyeOpenProbability ?? 0.0;

    if (smilingProbability > 0.7) {
      return 'happy';
    } else if (leftEyeOpenProbability < 0.3 && rightEyeOpenProbability < 0.3) {
      return 'sad';
    } else if (smilingProbability < 0.3 && leftEyeOpenProbability > 0.7) {
      return 'surprised';
    } else {
      return 'neutral';
    }
  }

  FaceDetectionData createFaceDetectionData({
    required Face face,
    required String imagePath,
    required bool notificationSent,
  }) {
    final emotion = analyzeEmotion(face);
    final confidence = face.headEulerAngleZ ?? 0.0;

    return FaceDetectionData(
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      detectedAt: DateTime.now(),
      confidence: (confidence / 100).clamp(0.0, 1.0),
      emotion: emotion,
      emotionConfidence: face.smilingProbability ?? 0.5,
      imagePath: imagePath,
      notificationSent: notificationSent,
      notificationMessage: 'Phát hiện khuôn mặt - Cảm xúc: $emotion',
    );
  }

  bool shouldSendNotification() {
    if (!ESP32Config.enableFaceDetectionNotification) return false;
    final now = DateTime.now();
    if (_lastNotificationTime == null) {
      _lastNotificationTime = now;
      return true;
    }
    final difference = now.difference(_lastNotificationTime!).inSeconds;
    if (difference >= ESP32Config.notificationCooldownSeconds) {
      _lastNotificationTime = now;
      return true;
    }
    return false;
  }

  void dispose() {
    _faceDetector.close();
  }
}

// ============= SERVICES - Notification Service =============
class NotificationServiceCamera {
  static final NotificationServiceCamera _instance = NotificationServiceCamera._internal();
  factory NotificationServiceCamera() => _instance;
  NotificationServiceCamera._internal();

  final FlutterLocalNotificationsPlugin _notificationsPlugin = FlutterLocalNotificationsPlugin();

  Future<void> initialize() async {
    const AndroidInitializationSettings androidSettings = AndroidInitializationSettings('@mipmap/ic_launcher');
    const DarwinInitializationSettings iosSettings = DarwinInitializationSettings(
      requestAlertPermission: true,
      requestBadgePermission: true,
      requestSoundPermission: true,
    );

    const InitializationSettings settings = InitializationSettings(
      android: androidSettings,
      iOS: iosSettings,
    );

    await _notificationsPlugin.initialize(settings);

    const AndroidNotificationChannel faceChannel = AndroidNotificationChannel(
      id: 'face_detection_channel',
      name: 'Face Detection',
      description: 'Notifications for face detection events',
      importance: Importance.high,
    );

    await _notificationsPlugin
        .resolvePlatformSpecificImplementation<AndroidFlutterLocalNotificationsPlugin>()
        ?.createNotificationChannel(faceChannel);
  }

  Future<void> showFaceDetectionNotification({
    required String title,
    required String body,
  }) async {
    const AndroidNotificationDetails androidDetails = AndroidNotificationDetails(
      'face_detection_channel',
      'Face Detection',
      importance: Importance.high,
      priority: Priority.high,
      playSound: true,
      enableVibration: true,
      enableLedFlash: true,
      ticker: 'Face Detected',
    );

    const DarwinNotificationDetails iosDetails = DarwinNotificationDetails(
      presentAlert: true,
      presentBadge: true,
      presentSound: true,
    );

    const NotificationDetails notificationDetails = NotificationDetails(
      android: androidDetails,
      iOS: iosDetails,
    );

    await _notificationsPlugin.show(
      DateTime.now().millisecond,
      title,
      body,
      notificationDetails,
    );
  }
}

// ============= PROVIDER =============
class CameraProvider extends ChangeNotifier {
  late ESP32CameraService _cameraService;
  late FaceDetectionService _faceDetectionService;
  late NotificationServiceCamera _notificationService;

  bool _isConnected = false;
  bool _isConnecting = false;
  String _esp32IP = ESP32Config.defaultIPAddress;
  Uint8List? _currentFrame;
  List<FaceDetectionData> _detectedFaces = [];
  String? _errorMessage;

  bool get isConnected => _isConnected;
  bool get isConnecting => _isConnecting;
  String get esp32IP => _esp32IP;
  Uint8List? get currentFrame => _currentFrame;
  List<FaceDetectionData> get detectedFaces => _detectedFaces;
  String? get errorMessage => _errorMessage;

  CameraProvider() {
    _cameraService = ESP32CameraService();
    _faceDetectionService = FaceDetectionService();
    _notificationService = NotificationServiceCamera();
    _initialize();
  }

  Future<void> _initialize() async {
    try {
      await _faceDetectionService.initialize();
      await _notificationService.initialize();
    } catch (e) {
      _errorMessage = 'Lỗi khởi tạo: $e';
      notifyListeners();
    }
  }

  Future<bool> connectToESP32({required String ipAddress}) async {
    _isConnecting = true;
    _errorMessage = null;
    notifyListeners();

    try {
      _esp32IP = ipAddress;
      final success = await _cameraService.connectToESP32(
        ipAddress: ipAddress,
        port: ESP32Config.defaultPort,
      );

      if (success) {
        _streamUrl = _cameraService.streamUrl;
      } else {
        _streamUrl = null;
      }

      _isConnected = success;
      _isConnecting = false;

      if (!success) {
        _errorMessage = 'Không thể kết nối ESP32';
      }

      notifyListeners();
      return success;
    } catch (e) {
      _isConnecting = false;
      _isConnected = false;
      _errorMessage = 'Lỗi kết nối: ${e.toString()}';
      notifyListeners();
      return false;
    }
  }

  void disconnect() {
    _cameraService.disconnect();
    _faceDetectionService.dispose();
    _isConnected = false;
    _currentFrame = null;
    _detectedFaces.clear();
    _errorMessage = null;
    _streamUrl = null;
    notifyListeners();
  }

  Future<void> captureSnapshot() async {
    try {
      final imageData = await _cameraService.getSnapshotFromESP32();
      if (imageData != null) {
        _currentFrame = Uint8List.fromList(imageData);
        notifyListeners();
      }
    } catch (e) {
      _errorMessage = 'Lỗi chụp ảnh: $e';
      notifyListeners();
    }
  }

  void addFaceDetection(FaceDetectionData faceData) {
    _detectedFaces.insert(0, faceData);
    if (_detectedFaces.length > 20) {
      _detectedFaces.removeLast();
    }
    notifyListeners();
  }

  @override
  void dispose() {
    disconnect();
    super.dispose();
  }
}

// ============= SCREEN - Camera Screen =============
class CameraScreen extends StatefulWidget {
  const CameraScreen({Key? key}) : super(key: key);

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  late TextEditingController _ipController;

  @override
  void initState() {
    super.initState();
    _ipController = TextEditingController(text: ESP32Config.defaultIPAddress);
  }

  Future<void> _connectToESP32(CameraProvider provider) async {
    final success = await provider.connectToESP32(ipAddress: _ipController.text);

    if (mounted) {
      if (success) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Kết nối ESP32 thành công!'),
            backgroundColor: Colors.green,
          ),
        );
        _startStreamProcessing(provider);
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(provider.errorMessage ?? 'Không thể kết nối đến ESP32'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  Future<void> _startStreamProcessing(CameraProvider provider) async {
    while (provider.isConnected && mounted) {
      await provider.captureSnapshot();
      await Future.delayed(const Duration(milliseconds: 500));
    }
  }

  @override
  void dispose() {
    _ipController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('ESP32 Camera'),
        elevation: 0,
        centerTitle: true,
      ),
      body: Consumer<CameraProvider>(
        builder: (context, provider, _) {
          return Column(
            children: [
              Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Cài đặt kết nối',
                      style: Theme.of(context).textTheme.titleMedium,
                    ),
                    const SizedBox(height: 12),
                    TextField(
                      controller: _ipController,
                      decoration: InputDecoration(
                        labelText: 'Địa chỉ IP của ESP32',
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(8),
                        ),
                        prefixIcon: const Icon(Icons.router),
                        enabled: !provider.isConnected,
                      ),
                    ),
                    const SizedBox(height: 12),
                    SizedBox(
                      width: double.infinity,
                      height: 48,
                      child: ElevatedButton(
                        onPressed: provider.isConnecting
                            ? null
                            : () => _connectToESP32(provider),
                        child: Text(
                          provider.isConnecting
                              ? 'Đang kết nối...'
                              : (provider.isConnected ? 'Đã kết nối' : 'Kết nối'),
                        ),
                      ),
                    ),
                    if (provider.isConnected)
                      Padding(
                        padding: const EdgeInsets.only(top: 12),
                        child: SizedBox(
                          width: double.infinity,
                          height: 48,
                          child: ElevatedButton(
                            onPressed: () => provider.disconnect(),
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Colors.red,
                            ),
                            child: const Text('Ngắt kết nối'),
                          ),
                        ),
                      ),
                  ],
                ),
              ),
              if (provider.isConnected)
                Expanded(
                  child: Column(
                    children: [
                      Expanded(
                        child: Container(
                          color: Colors.black,
                          child: provider.currentFrame != null
                              ? Image.memory(
                                  provider.currentFrame!,
                                  fit: BoxFit.cover,
                                )
                              : const Center(
                                  child: CircularProgressIndicator(
                                    valueColor: AlwaysStoppedAnimation<Color>(
                                      Colors.white,
                                    ),
                                  ),
                                ),
                        ),
                      ),
                      if (provider.detectedFaces.isNotEmpty)
                        Container(
                          color: Colors.grey[900],
                          padding: const EdgeInsets.all(12),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              const Text(
                                'Phát hiện gần đây:',
                                style: TextStyle(
                                  color: Colors.white,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                              SizedBox(
                                height: 80,
                                child: ListView.builder(
                                  scrollDirection: Axis.horizontal,
                                  itemCount: provider.detectedFaces.length,
                                  itemBuilder: (context, index) {
                                    final face = provider.detectedFaces[index];
                                    return Container(
                                      margin: const EdgeInsets.all(8),
                                      padding: const EdgeInsets.all(8),
                                      decoration: BoxDecoration(
                                        color: Colors.grey[800],
                                        borderRadius: BorderRadius.circular(8),
                                      ),
                                      child: Column(
                                        mainAxisAlignment: MainAxisAlignment.center,
                                        children: [
                                          Text(
                                            face.emotion.toUpperCase(),
                                            style: TextStyle(
                                              color: _getEmotionColor(face.emotion),
                                              fontWeight: FontWeight.bold,
                                              fontSize: 12,
                                            ),
                                          ),
                                          Text(
                                            '${(face.confidence * 100).toStringAsFixed(1)}%',
                                            style: const TextStyle(
                                              color: Colors.white70,
                                              fontSize: 10,
                                            ),
                                          ),
                                        ],
                                      ),
                                    );
                                  },
                                ),
                              ),
                            ],
                          ),
                        ),
                    ],
                  ),
                )
              else
                Expanded(
                  child: Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(
                          Icons.videocam_off,
                          size: 64,
                          color: Colors.grey[400],
                        ),
                        const SizedBox(height: 16),
                        Text(
                          'Chưa kết nối với ESP32',
                          style: TextStyle(fontSize: 16, color: Colors.grey[600]),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          'Nhập IP ESP32 và nhấn kết nối',
                          style: TextStyle(fontSize: 12, color: Colors.grey[500]),
                        ),
                      ],
                    ),
                  ),
                ),
            ],
          );
        },
      ),
    );
  }

  Color _getEmotionColor(String emotion) {
    switch (emotion) {
      case 'happy':
        return Colors.yellow;
      case 'sad':
        return Colors.blue;
      case 'angry':
        return Colors.red;
      case 'surprised':
        return Colors.orange;
      case 'fear':
        return Colors.purple;
      case 'disgust':
        return Colors.green;
      default:
        return Colors.white;
    }
  }
}
