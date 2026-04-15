import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:webview_flutter/webview_flutter.dart';

import 'widgets/app_logo.dart';

class AppConfig {
  static const String defaultServerUrl = 'http://10.62.123.183:5000';
  static const String defaultEsp32Url = 'http://10.62.123.117/';
  static const int defaultPanAngle = 90;
  static const int defaultTiltAngle = 90;
  static const int minTiltAngle = 0;
  static const int maxTiltAngle = 180;
}

class HistoryEntry {
  HistoryEntry({
    required this.type,
    required this.title,
    required this.timestamp,
    required this.summary,
    this.resultUrl,
    this.previewUrl,
  });

  final String type;
  final String title;
  final String timestamp;
  final String summary;
  final String? resultUrl;
  final String? previewUrl;

  Map<String, dynamic> toJson() => {
        'type': type,
        'title': title,
        'timestamp': timestamp,
        'summary': summary,
        'resultUrl': resultUrl,
        'previewUrl': previewUrl,
      };

  factory HistoryEntry.fromJson(Map<String, dynamic> json) => HistoryEntry(
        type: json['type'] as String? ?? '',
        title: json['title'] as String? ?? '',
        timestamp: json['timestamp'] as String? ?? '',
        summary: json['summary'] as String? ?? '',
        resultUrl: json['resultUrl'] as String?,
        previewUrl: json['previewUrl'] as String?,
      );
}

class AppProvider extends ChangeNotifier {
  AppProvider() {
    _loadPreferences();
  }

  bool isLoading = false;
  bool historyLoaded = false;
  bool isServoBusy = false;

  String? error;
  String serverUrl = AppConfig.defaultServerUrl;
  String esp32BaseUrl = AppConfig.defaultEsp32Url;
  int panAngle = AppConfig.defaultPanAngle;
  int tiltAngle = AppConfig.defaultTiltAngle;
  int ledIntensity = 0;

  String? cameraResultImageUrl;
  String? imageResultImageUrl;
  Map<String, dynamic>? cameraResult;
  Map<String, dynamic>? imageResult;
  Map<String, dynamic>? videoResult;

  List<HistoryEntry> cameraHistory = [];
  List<HistoryEntry> imageHistory = [];
  List<HistoryEntry> videoHistory = [];

  static const _cameraHistoryKey = 'camera_history';
  static const _imageHistoryKey = 'image_history';
  static const _videoHistoryKey = 'video_history';
  static const _esp32UrlKey = 'esp32_url';
  static const _serverUrlKey = 'server_url';

  String get directEsp32StreamUrl => _normalizeEsp32StreamUrl(esp32BaseUrl);
  String get processedEsp32StreamUrl =>
      '$serverUrl/esp32-stream-analyze?base_url=${Uri.encodeComponent(esp32BaseUrl)}';
  String get cameraSnapshotAnalyzeUrl => '$serverUrl/esp32-snapshot-analyze';

  static String _normalizeEsp32BaseUrl(String input) {
    var value = input.trim();
    if (!value.startsWith('http://') && !value.startsWith('https://')) {
      value = 'http://$value';
    }

    value = value.replaceFirst(RegExp(r':81/stream/?$'), '');
    value = value.replaceFirst(RegExp(r'/stream/?$'), '');
    value = value.replaceFirst(RegExp(r'/status/?$'), '');

    final uri = Uri.parse(value);
    var normalizedPath = uri.path.replaceFirst(RegExp(r'/$'), '');
    if (normalizedPath.endsWith('/stream')) {
      normalizedPath =
          normalizedPath.substring(0, normalizedPath.length - '/stream'.length);
    } else if (normalizedPath.endsWith('/status')) {
      normalizedPath =
          normalizedPath.substring(0, normalizedPath.length - '/status'.length);
    }

    final normalizedUri = Uri(
      scheme: uri.scheme,
      userInfo: uri.userInfo.isEmpty ? null : uri.userInfo,
      host: uri.host,
      port: uri.hasPort && uri.port != 81 ? uri.port : null,
      path: normalizedPath.isEmpty ? '/' : '$normalizedPath/',
    );
    return normalizedUri.toString();
  }

  static String _normalizeEsp32StreamUrl(String input) {
    final base = _normalizeEsp32BaseUrl(input).replaceFirst(RegExp(r'/$'), '');
    if (base.endsWith(':81/stream')) return base;
    if (base.endsWith('/stream')) return base;
    return '$base:81/stream';
  }

  static String _normalizeServerUrl(String input) {
    var value = input.trim();
    if (!value.startsWith('http://') && !value.startsWith('https://')) {
      value = 'http://$value';
    }
    if (value.endsWith('/')) value = value.substring(0, value.length - 1);
    return value;
  }

  Future<void> _loadPreferences() async {
    final prefs = await SharedPreferences.getInstance();
    esp32BaseUrl =
        _normalizeEsp32BaseUrl(prefs.getString(_esp32UrlKey) ?? esp32BaseUrl);
    serverUrl =
        _normalizeServerUrl(prefs.getString(_serverUrlKey) ?? serverUrl);
    cameraHistory = _decodeHistory(prefs.getStringList(_cameraHistoryKey));
    imageHistory = _decodeHistory(prefs.getStringList(_imageHistoryKey));
    videoHistory = _decodeHistory(prefs.getStringList(_videoHistoryKey));
    historyLoaded = true;
    notifyListeners();
  }

  List<HistoryEntry> _decodeHistory(List<String>? raw) {
    if (raw == null) return [];
    return raw
        .map((item) =>
            HistoryEntry.fromJson(jsonDecode(item) as Map<String, dynamic>))
        .toList();
  }

  Future<void> _saveHistories() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setStringList(
      _cameraHistoryKey,
      cameraHistory.map((e) => jsonEncode(e.toJson())).toList(),
    );
    await prefs.setStringList(
      _imageHistoryKey,
      imageHistory.map((e) => jsonEncode(e.toJson())).toList(),
    );
    await prefs.setStringList(
      _videoHistoryKey,
      videoHistory.map((e) => jsonEncode(e.toJson())).toList(),
    );
  }

  Future<void> _saveConnectionSettings() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_esp32UrlKey, esp32BaseUrl);
    await prefs.setString(_serverUrlKey, serverUrl);
  }

  String _controlUrl(String variable, int value) {
    final base = _normalizeEsp32BaseUrl(esp32BaseUrl);
    return '${base}control?var=$variable&val=$value';
  }

  String _legacyServoUrl(String variable, int value) {
    final base = _normalizeEsp32BaseUrl(esp32BaseUrl);
    return '${base}servo?$variable=$value';
  }

  Future<String> _sendServoCommand(String variable, int value) async {
    final primaryUrl = _legacyServoUrl(variable, value);
    try {
      final response = await http
          .get(Uri.parse(primaryUrl))
          .timeout(const Duration(milliseconds: 650));
      if (response.statusCode < 400) {
        return primaryUrl;
      }
    } catch (_) {
      // fall through
    }

    final fallbackUrl = _controlUrl(variable, value);
    final fallbackResponse = await http
        .get(Uri.parse(fallbackUrl))
        .timeout(const Duration(milliseconds: 1500));
    if (fallbackResponse.statusCode >= 400) {
      throw Exception(
        'Khong dieu khien duoc servo $variable '
        '(servo va control deu that bai, HTTP ${fallbackResponse.statusCode})',
      );
    }
    return fallbackUrl;
  }

  Future<void> updateEspLight(int value) async {
    error = null;
    final nextValue = value.clamp(0, 255);
    try {
      final response = await http
          .get(Uri.parse(_controlUrl('led_intensity', nextValue)))
          .timeout(const Duration(seconds: 5));
      if (response.statusCode >= 400) {
        throw Exception(
          'Khong dieu chinh duoc do sang ESP (HTTP ${response.statusCode})',
        );
      }
      ledIntensity = nextValue;
      notifyListeners();
    } catch (exc) {
      error = 'Loi dieu khien den ESP: $exc';
      notifyListeners();
    }
  }

  Future<void> setEsp32Url(String input) async {
    error = null;
    try {
      esp32BaseUrl = _normalizeEsp32BaseUrl(input);
      await _saveConnectionSettings();
      notifyListeners();
    } catch (exc) {
      error = exc.toString();
      notifyListeners();
    }
  }

  Future<void> setServerUrl(String input) async {
    error = null;
    try {
      final normalized = _normalizeServerUrl(input);
      final response =
          await http.get(Uri.parse('$normalized/health')).timeout(const Duration(seconds: 5));
      final data = jsonDecode(response.body) as Map<String, dynamic>;
      if (response.statusCode >= 400) {
        throw Exception(data['error'] ?? 'Khong ket noi duoc server');
      }
      serverUrl = normalized;
      await _saveConnectionSettings();
      notifyListeners();
    } catch (exc) {
      error = exc.toString();
      notifyListeners();
    }
  }

  Future<void> updatePanTilt({int? pan, int? tilt}) async {
    error = null;
    isServoBusy = true;
    notifyListeners();
    String? lastControlUrl;
    try {
      if (pan != null) {
        final nextPan = pan.clamp(0, 180);
        lastControlUrl = await _sendServoCommand('pan', nextPan);
        panAngle = nextPan;
      }
      if (tilt != null) {
        final nextTilt =
            tilt.clamp(AppConfig.minTiltAngle, AppConfig.maxTiltAngle);
        lastControlUrl = await _sendServoCommand('tilt', nextTilt);
        tiltAngle = nextTilt;
      }
      await _saveConnectionSettings();
    } catch (exc) {
      error = 'Loi dieu khien ESP32: $exc\n'
          "Control URL: ${lastControlUrl ?? _controlUrl('pan', pan ?? panAngle)}";
    } finally {
      isServoBusy = false;
      notifyListeners();
    }
  }

  Future<void> centerPanTilt() async {
    await updatePanTilt(
      pan: AppConfig.defaultPanAngle,
      tilt: AppConfig.defaultTiltAngle,
    );
  }

  String _now() => DateTime.now().toString().replaceFirst('.', ' - ');

  List<Map<String, dynamic>> subjectsOf(Map<String, dynamic> data) {
    final raw = data['subjects'];
    if (raw is List) {
      return raw
          .whereType<Map>()
          .map((item) => item.cast<String, dynamic>())
          .toList();
    }
    return [];
  }

  String subjectLine(Map<String, dynamic> subject) {
  final verified = subject['verified'] == true;
  final idConf = (subject['identity_confidence'] as num?)?.toDouble() ?? 0.0;

  final reliable = subject['emotion_reliable'] == true;
  final emoConf = (subject['emotion_confidence'] as num?)?.toDouble() ?? 0.0;

  final rawId = (subject['display_identity'] ??
          subject['raw_identity'] ??
          subject['identity'] ??
          'unknown')
      .toString();

  final rawEmotion = (subject['display_emotion'] ??
          subject['raw_emotion'] ??
          subject['emotion'] ??
          'no-face')
      .toString();

  // ===== IDENTITY =====
  String identity;
  if (verified && rawId != 'unknown') {
    identity = rawId;
  } else if (idConf >= 0.75 && rawId != 'unknown') {
    identity = '$rawId (?)';
  } else {
    identity = 'unknown';
  }

  // ===== EMOTION =====
  String emotion;
  if (reliable && rawEmotion != 'uncertain' && rawEmotion != 'no-face') {
    emotion = rawEmotion;
  } else if (emoConf >= 0.65 &&
      rawEmotion != 'uncertain' &&
      rawEmotion != 'no-face') {
    emotion = '$rawEmotion (?)';
  } else {
    emotion = 'uncertain';
  }

  final faceStatus =
      subject['face_status'] ?? 'Khong co thong tin khuon mat';

  return '$identity | Cam xuc: $emotion | $faceStatus';
}
  String subjectsSummary(Map<String, dynamic> data) {
    final subjects = subjectsOf(data);
    if (subjects.isEmpty) {
      return 'Khong phat hien khuon mat';
    }
    return subjects.map(subjectLine).join('\n');
  }

  Future<void> analyzeImageFile(File file) async {
    isLoading = true;
    error = null;
    imageResult = null;
    imageResultImageUrl = null;
    notifyListeners();
    try {
      final data = await _sendMultipart(
        '/analyze-image',
        await http.MultipartFile.fromPath('file', file.path),
      );
      imageResult = data;
      imageResultImageUrl = data['result_url'] as String?;
      await _addHistory(
        type: 'image',
        title: 'Nhan dien bang anh',
        summary: imageSummary(data),
        resultUrl: data['result_url'] as String?,
        previewUrl: data['result_url'] as String?,
      );
    } catch (exc) {
      error = exc.toString();
    } finally {
      isLoading = false;
      notifyListeners();
    }
  }

  Future<void> analyzeVideoFile(File file) async {
    isLoading = true;
    error = null;
    videoResult = null;
    notifyListeners();
    try {
      final data = await _sendMultipart(
        '/analyze-video',
        await http.MultipartFile.fromPath('file', file.path),
      );
      videoResult = data;
      await _addHistory(
        type: 'video',
        title: 'Nhan dien bang video',
        summary: videoSummary(data),
        resultUrl: data['result_url'] as String?,
        previewUrl: data['preview_url'] as String?,
      );
    } catch (exc) {
      error = exc.toString();
    } finally {
      isLoading = false;
      notifyListeners();
    }
  }

  Future<Map<String, dynamic>> _sendMultipart(
    String endpoint,
    http.MultipartFile file,
  ) async {
    final request =
        http.MultipartRequest('POST', Uri.parse('$serverUrl$endpoint'))
          ..files.add(file);
    final response = await request.send();
    final body = await response.stream.bytesToString();
    final data = jsonDecode(body) as Map<String, dynamic>;
    if (response.statusCode >= 400) {
      throw Exception(data['error'] ?? 'Request failed');
    }
    return data;
  }

  String imageSummary(Map<String, dynamic> data) {
    final subjects = subjectsOf(data);

    if (subjects.isEmpty) {
      return 'Khong phat hien khuon mat';
    }

    final lines = subjects.map(subjectLine).toList();

    return lines.join('\n');
  }

  String videoSummary(Map<String, dynamic> data) {
    final subjects = subjectsOf(data);

    if (subjects.isEmpty) {
      return 'Khong phat hien khuon mat trong video';
    }

    final lines = subjects.map(subjectLine).toList();

    return lines.join('\n');
  }
  Future<void> _addHistory({
    required String type,
    required String title,
    required String summary,
    String? resultUrl,
    String? previewUrl,
  }) async {
    final entry = HistoryEntry(
      type: type,
      title: title,
      timestamp: _now(),
      summary: summary,
      resultUrl: resultUrl,
      previewUrl: previewUrl,
    );
    if (type == 'camera') {
      cameraHistory = [entry, ...cameraHistory];
    } else if (type == 'image') {
      imageHistory = [entry, ...imageHistory];
    } else {
      videoHistory = [entry, ...videoHistory];
    }
    await _saveHistories();
    notifyListeners();
  }
}

class CameraScreen extends StatelessWidget {
  const CameraScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return const MainFeatureScreen();
  }
}

class MainFeatureScreen extends StatelessWidget {
  const MainFeatureScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: AppLogo(), centerTitle: true),
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Color(0xFFF6F8FC), Color(0xFFE8EEF9)],
          ),
        ),
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            _FeatureTile(
              icon: Icons.videocam_outlined,
              title: 'Mo camera',
              subtitle: 'ESP -> Backend -> App',
              onTap: () => Navigator.of(context)
                  .push(MaterialPageRoute(builder: (_) => const CameraTab())),
            ),
            _FeatureTile(
              icon: Icons.image_outlined,
              title: 'Tai anh',
              subtitle: 'Nhan dien bang anh',
              onTap: () => Navigator.of(context)
                  .push(MaterialPageRoute(builder: (_) => const ImageTab())),
            ),
            _FeatureTile(
              icon: Icons.video_file_outlined,
              title: 'Tai video',
              subtitle: 'Nhan dien bang video',
              onTap: () => Navigator.of(context)
                  .push(MaterialPageRoute(builder: (_) => const VideoTab())),
            ),
          ],
        ),
      ),
    );
  }
}

class CameraTab extends StatefulWidget {
  const CameraTab({super.key});

  @override
  State<CameraTab> createState() => _CameraTabState();
}

class _CameraTabState extends State<CameraTab> {
  late final TextEditingController _espCtrl;
  late final TextEditingController _serverCtrl;

  bool _settingsExpanded = false;

  @override
  void initState() {
    super.initState();
    final provider = context.read<AppProvider>();
    _espCtrl = TextEditingController(text: provider.esp32BaseUrl);
    _serverCtrl = TextEditingController(text: provider.serverUrl);
  }

  @override
  void dispose() {
    _espCtrl.dispose();
    _serverCtrl.dispose();
    super.dispose();
  }

  Future<void> _saveConnections() async {
    final provider = context.read<AppProvider>();
    await provider.setEsp32Url(_espCtrl.text);
    await provider.setServerUrl(_serverCtrl.text);
  }

  @override
  Widget build(BuildContext context) {
    final provider = context.watch<AppProvider>();

    return Scaffold(
      appBar: AppBar(
        title: const Text('Mo camera'),
        actions: [
          IconButton(
            onPressed: () => Navigator.of(context).push(MaterialPageRoute(
              builder: (_) => HistoryScreen(
                title: 'Lich su camera',
                entries: provider.cameraHistory,
              ),
            )),
            icon: const Icon(Icons.history),
          ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Container(
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(22),
            ),
            child: Theme(
              data: Theme.of(context)
                  .copyWith(dividerColor: Colors.transparent),
              child: ExpansionTile(
                initiallyExpanded: _settingsExpanded,
                onExpansionChanged: (value) {
                  setState(() {
                    _settingsExpanded = value;
                  });
                },
                leading: const Icon(Icons.router_outlined),
                title: const Text(
                  'Ket noi camera va server',
                  style: TextStyle(fontWeight: FontWeight.w700),
                ),
                subtitle: Text(
                  'ESP32: ${provider.esp32BaseUrl}',
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
                childrenPadding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
                children: [
                  TextField(
                    controller: _espCtrl,
                    decoration: const InputDecoration(
                      labelText: 'Link ESP32',
                      hintText: 'http://10.62.123.117/',
                      border: OutlineInputBorder(),
                    ),
                  ),
                  const SizedBox(height: 12),
                  TextField(
                    controller: _serverCtrl,
                    decoration: const InputDecoration(
                      labelText: 'Link server nhan dien',
                      hintText: 'http://10.62.123.183:5000',
                      border: OutlineInputBorder(),
                    ),
                  ),
                  const SizedBox(height: 12),
                  SizedBox(
                    width: double.infinity,
                    child: FilledButton.icon(
                      onPressed: provider.isLoading ? null : _saveConnections,
                      icon: const Icon(Icons.link),
                      label: const Text('Cap nhat ket noi'),
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),
          Container(
            decoration: BoxDecoration(
              color: const Color(0xFF1E1E1E),
              borderRadius: BorderRadius.circular(30),
              border: Border.all(color: const Color(0xFF1E1E1E), width: 4),
            ),
            child: AspectRatio(
              aspectRatio: 4 / 3,
              child: ClipRRect(
                borderRadius: BorderRadius.circular(22),
                child: Container(
                  color: Colors.black,
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      _Esp32StreamView(
                        streamUrl: provider.processedEsp32StreamUrl,
                        showGuideFrame: false,
                      ),
                      Positioned(
                        left: 12,
                        top: 12,
                        right: 12,
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 10,
                            vertical: 6,
                          ),
                          decoration: BoxDecoration(
                            color: Colors.black.withValues(alpha: 0.45),
                            borderRadius: BorderRadius.circular(999),
                          ),
                          child: Text(
                            provider.processedEsp32StreamUrl,
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 11,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
          const SizedBox(height: 12),
          _PanTiltPad(provider: provider),
          const SizedBox(height: 10),
          _LightSlider(
            value: provider.ledIntensity.toDouble(),
            onChanged: (value) =>
                context.read<AppProvider>().updateEspLight(value.round()),
          ),
          const SizedBox(height: 10),
          Row(
            children: [
              Expanded(
                child: _CompactMetric(
                  title: 'Pan',
                  value: '${provider.panAngle} deg',
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: _CompactMetric(
                  title: 'Tilt',
                  value: '${provider.tiltAngle} deg',
                ),
              ),
            ],
          ),
          if (provider.error != null) ...[
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.all(14),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(18),
              ),
              child: Text(
                provider.error!,
                style: const TextStyle(color: Colors.red),
              ),
            ),
          ],
        ],
      ),
    );
  }
}

class _CompactMetric extends StatelessWidget {
  const _CompactMetric({required this.title, required this.value});

  final String title;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFFF4F7F5),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title, style: const TextStyle(color: Colors.black54)),
          const SizedBox(height: 6),
          Text(
            value,
            style: const TextStyle(
              fontSize: 22,
              fontWeight: FontWeight.w700,
            ),
          ),
        ],
      ),
    );
  }
}

class _PanTiltPad extends StatefulWidget {
  const _PanTiltPad({required this.provider});

  final AppProvider provider;

  @override
  State<_PanTiltPad> createState() => _PanTiltPadState();
}

class _PanTiltPadState extends State<_PanTiltPad> {
  static const _commandDelay = Duration(milliseconds: 120);

  late double _panValue;
  late double _tiltValue;
  Timer? _panTimer;
  Timer? _tiltTimer;
  int? _queuedPan;
  int? _queuedTilt;
  bool _panSending = false;
  bool _tiltSending = false;

  @override
  void initState() {
    super.initState();
    _panValue = widget.provider.panAngle.toDouble();
    _tiltValue = widget.provider.tiltAngle.toDouble();
  }

  @override
  void dispose() {
    _panTimer?.cancel();
    _tiltTimer?.cancel();
    super.dispose();
  }

  @override
  void didUpdateWidget(covariant _PanTiltPad oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (!_panSending) {
      _panValue = widget.provider.panAngle.toDouble();
    }
    if (!_tiltSending) {
      _tiltValue = widget.provider.tiltAngle.toDouble();
    }
  }

  void _queuePan(double value, {bool immediate = false}) {
    _queuedPan = value.round().clamp(0, 180);
    _panTimer?.cancel();
    if (immediate) {
      _dispatchPan();
      return;
    }
    _panTimer = Timer(_commandDelay, _dispatchPan);
  }

  void _queueTilt(double value, {bool immediate = false}) {
    _queuedTilt = value.round().clamp(0, 180);
    _tiltTimer?.cancel();
    if (immediate) {
      _dispatchTilt();
      return;
    }
    _tiltTimer = Timer(_commandDelay, _dispatchTilt);
  }

  Future<void> _dispatchPan() async {
    if (_panSending || _queuedPan == null || !mounted) return;
    final target = _queuedPan!;
    _queuedPan = null;
    _panSending = true;
    try {
      await context.read<AppProvider>().updatePanTilt(pan: target);
    } finally {
      _panSending = false;
      if (_queuedPan != null && _queuedPan != target && mounted) {
        unawaited(_dispatchPan());
      }
    }
  }

  Future<void> _dispatchTilt() async {
    if (_tiltSending || _queuedTilt == null || !mounted) return;
    final target = _queuedTilt!;
    _queuedTilt = null;
    _tiltSending = true;
    try {
      await context.read<AppProvider>().updatePanTilt(tilt: target);
    } finally {
      _tiltSending = false;
      if (_queuedTilt != null && _queuedTilt != target && mounted) {
        unawaited(_dispatchTilt());
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFFF4F7F5),
        borderRadius: BorderRadius.circular(18),
      ),
      child: Column(
        children: [
          _ServoSlider(
            label: 'Pan',
            value: _panValue,
            min: 0,
            max: 180,
            onChanged: (value) {
              setState(() {
                _panValue = value;
              });
              _queuePan(value);
            },
            onChangeEnd: (value) {
              _queuePan(value, immediate: true);
            },
          ),
          const SizedBox(height: 12),
          _ServoSlider(
            label: 'Tilt',
            value: _tiltValue,
            min: 0,
            max: 180,
            onChanged: (value) {
              setState(() {
                _tiltValue = value;
              });
              _queueTilt(value);
            },
            onChangeEnd: (value) {
              _queueTilt(value, immediate: true);
            },
          ),
          const SizedBox(height: 12),
          Align(
            alignment: Alignment.centerRight,
            child: OutlinedButton.icon(
              onPressed: () => context.read<AppProvider>().centerPanTilt(),
              icon: const Icon(Icons.center_focus_strong),
              label: const Text('Dua ve giua'),
            ),
          ),
        ],
      ),
    );
  }
}

class _LightSlider extends StatelessWidget {
  const _LightSlider({required this.value, required this.onChanged});

  final double value;
  final ValueChanged<double> onChanged;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFFF9FBFA),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFFE1EBE5)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Text(
                'Do sang ESP',
                style: TextStyle(fontSize: 15, fontWeight: FontWeight.w700),
              ),
              const Spacer(),
              Text(
                '${value.round()}',
                style: const TextStyle(color: Colors.black54),
              ),
            ],
          ),
          SliderTheme(
            data: SliderTheme.of(context).copyWith(
              activeTrackColor: const Color(0xFFFFC046),
              inactiveTrackColor: const Color(0xFFF0E2B8),
              thumbColor: const Color(0xFFF09A00),
              overlayColor: const Color(0xFFFFC046).withValues(alpha: 0.18),
            ),
            child: Slider(
              min: 0,
              max: 255,
              divisions: 255,
              value: value.clamp(0, 255),
              onChanged: onChanged,
            ),
          ),
        ],
      ),
    );
  }
}

class _ServoSlider extends StatelessWidget {
  const _ServoSlider({
    required this.label,
    required this.value,
    required this.min,
    required this.max,
    required this.onChanged,
    required this.onChangeEnd,
  });

  final String label;
  final double value;
  final double min;
  final double max;
  final ValueChanged<double>? onChanged;
  final ValueChanged<double>? onChangeEnd;

  @override
  Widget build(BuildContext context) {
    final safeValue = value.clamp(min, max);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Text(
              label,
              style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w700),
            ),
            const Spacer(),
            Text(
              '${safeValue.round()} do',
              style: const TextStyle(color: Colors.black54),
            ),
          ],
        ),
        const SizedBox(height: 4),
        SliderTheme(
          data: SliderTheme.of(context).copyWith(
            activeTrackColor: const Color(0xFF1E9E61),
            inactiveTrackColor: const Color(0xFFD6E6DD),
            thumbColor: const Color(0xFF125A45),
            overlayColor: const Color(0xFF1E9E61).withValues(alpha: 0.16),
          ),
          child: Slider(
            min: min,
            max: max,
            divisions: (max - min).round(),
            value: safeValue,
            onChanged: onChanged,
            onChangeEnd: onChangeEnd,
          ),
        ),
      ],
    );
  }
}

class _Esp32StreamView extends StatefulWidget {
  const _Esp32StreamView({
    required this.streamUrl,
    this.showGuideFrame = false,
    this.badgeText,
  });

  final String streamUrl;
  final bool showGuideFrame;
  final String? badgeText;

  @override
  State<_Esp32StreamView> createState() => _Esp32StreamViewState();
}

class _PortraitGuideFrame extends StatelessWidget {
  const _PortraitGuideFrame();

  @override
  Widget build(BuildContext context) {
    const frameColor = Color(0xFF46D67C);
    return Container(
      width: 154,
      height: 198,
      decoration: BoxDecoration(
        border: Border.all(color: frameColor, width: 3),
        borderRadius: BorderRadius.circular(6),
        boxShadow: [
          BoxShadow(
            color: frameColor.withValues(alpha: 0.28),
            blurRadius: 16,
            spreadRadius: 1,
          ),
        ],
      ),
    );
  }
}

class _Esp32StreamViewState extends State<_Esp32StreamView> {
  late final WebViewController _controller;
  bool _loading = true;
  bool _hasError = false;
  String _statusMessage = 'Dang ket noi camera backend...';
  String? _errorDetails;

  Future<void> _loadStream() async {
    if (!mounted) return;
    setState(() {
      _loading = true;
      _hasError = false;
      _errorDetails = null;
      _statusMessage = 'Dang tai stream backend: ${widget.streamUrl}';
    });
    try {
      await _controller.loadRequest(Uri.parse(widget.streamUrl));
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _loading = false;
        _hasError = true;
        _errorDetails = error.toString();
        _statusMessage = 'Khong mo duoc stream backend';
      });
    }
  }

  @override
  void initState() {
    super.initState();
    _controller = WebViewController()
      ..setJavaScriptMode(JavaScriptMode.unrestricted)
      ..setBackgroundColor(const Color(0xFF000000))
      ..setNavigationDelegate(
        NavigationDelegate(
          onPageStarted: (_) {
            if (mounted) {
              setState(() {
                _loading = true;
                _hasError = false;
                _errorDetails = null;
              });
            }
          },
          onPageFinished: (_) {
            if (mounted) {
              setState(() {
                _loading = false;
                _hasError = false;
                _errorDetails = null;
                _statusMessage = 'Da tai xong stream backend';
              });
            }
          },
          onWebResourceError: (error) {
            if (mounted) {
              setState(() {
                _loading = false;
                _hasError = true;
                _errorDetails =
                    'Code: ${error.errorCode} | Type: ${error.errorType}\n'
                    'Desc: ${error.description}\n'
                    'URL: ${error.url ?? widget.streamUrl}';
                _statusMessage = 'Khong mo duoc stream backend';
              });
            }
          },
        ),
      );
    WidgetsBinding.instance.addPostFrameCallback((_) => _loadStream());
  }

  @override
  void didUpdateWidget(covariant _Esp32StreamView oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.streamUrl != widget.streamUrl) {
      _loadStream();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      fit: StackFit.expand,
      children: [
        WebViewWidget(controller: _controller),
        if (_loading && !_hasError)
          const Center(
            child: CircularProgressIndicator(color: Color(0xFF46D67C)),
          ),
        if (widget.showGuideFrame)
          const IgnorePointer(
            child: Center(child: _PortraitGuideFrame()),
          ),
        if (_hasError)
          Center(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Text(
                _statusMessage,
                textAlign: TextAlign.center,
                style: const TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ),
        if (_hasError)
          Positioned(
            left: 12,
            right: 12,
            bottom: 12,
            child: DecoratedBox(
              decoration: BoxDecoration(
                color: Colors.black.withValues(alpha: 0.6),
                borderRadius: BorderRadius.circular(14),
              ),
              child: Padding(
                padding: const EdgeInsets.all(10),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      _statusMessage,
                      style: const TextStyle(color: Colors.white, fontSize: 12),
                    ),
                    if (_errorDetails != null) ...[
                      const SizedBox(height: 6),
                      Text(
                        _errorDetails!,
                        style: const TextStyle(
                          color: Colors.redAccent,
                          fontSize: 11,
                        ),
                      ),
                    ],
                  ],
                ),
              ),
            ),
          ),
      ],
    );
  }
}

class ImageTab extends StatefulWidget {
  const ImageTab({super.key});

  @override
  State<ImageTab> createState() => _ImageTabState();
}

class _ImageTabState extends State<ImageTab> {
  late final TextEditingController _serverCtrl;

  @override
  void initState() {
    super.initState();
    _serverCtrl =
        TextEditingController(text: context.read<AppProvider>().serverUrl);
  }

  @override
  void dispose() {
    _serverCtrl.dispose();
    super.dispose();
  }

  Future<void> _pickAndAnalyzeImage(BuildContext context) async {
    final result = await FilePicker.platform.pickFiles(type: FileType.image);
    if (result == null || result.files.single.path == null) return;
    await context
        .read<AppProvider>()
        .analyzeImageFile(File(result.files.single.path!));
  }

  @override
  Widget build(BuildContext context) {
    final provider = context.watch<AppProvider>();
    final data = provider.imageResult;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Tai anh'),
        actions: [
          IconButton(
            onPressed: () => Navigator.of(context).push(MaterialPageRoute(
              builder: (_) =>
                  HistoryScreen(title: 'Lich su anh', entries: provider.imageHistory),
            )),
            icon: const Icon(Icons.history),
          ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          TextField(
            controller: _serverCtrl,
            decoration: const InputDecoration(
              labelText: 'Link server',
              hintText: 'http://10.62.123.183:5000',
              border: OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 12),
          FilledButton.icon(
            onPressed: provider.isLoading
                ? null
                : () => context.read<AppProvider>().setServerUrl(_serverCtrl.text),
            icon: const Icon(Icons.cloud_done_outlined),
            label: const Text('Cap nhat server'),
          ),
          const SizedBox(height: 12),
          FilledButton.icon(
            onPressed:
                provider.isLoading ? null : () => _pickAndAnalyzeImage(context),
            icon: const Icon(Icons.upload_file),
            label: Text(provider.isLoading ? 'Dang xu ly...' : 'Chon anh'),
          ),
          if (provider.error != null)
            Padding(
              padding: const EdgeInsets.only(top: 12),
              child: Text(
                provider.error!,
                style: const TextStyle(color: Colors.red),
              ),
            ),
          if (provider.imageResultImageUrl != null) ...[
            const SizedBox(height: 16),
            ClipRRect(
              borderRadius: BorderRadius.circular(18),
              child: Image.network(
                provider.imageResultImageUrl!,
                fit: BoxFit.cover,
              ),
            ),
          ],
          if (data != null)
            _InfoCard(
              title: 'Ket qua anh',
              content: provider.imageSummary(data),
            ),
        ],
      ),
    );
  }
}

class VideoTab extends StatefulWidget {
  const VideoTab({super.key});

  @override
  State<VideoTab> createState() => _VideoTabState();
}

class _VideoTabState extends State<VideoTab> {
  late final TextEditingController _serverCtrl;

  @override
  void initState() {
    super.initState();
    _serverCtrl =
        TextEditingController(text: context.read<AppProvider>().serverUrl);
  }

  @override
  void dispose() {
    _serverCtrl.dispose();
    super.dispose();
  }

  Future<void> _pickAndAnalyzeVideo(BuildContext context) async {
    final result = await FilePicker.platform.pickFiles(type: FileType.video);
    if (result == null || result.files.single.path == null) return;
    await context
        .read<AppProvider>()
        .analyzeVideoFile(File(result.files.single.path!));
  }

  @override
  Widget build(BuildContext context) {
    final provider = context.watch<AppProvider>();
    final data = provider.videoResult;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Tai video'),
        actions: [
          IconButton(
            onPressed: () => Navigator.of(context).push(MaterialPageRoute(
              builder: (_) => HistoryScreen(
                title: 'Lich su video',
                entries: provider.videoHistory,
              ),
            )),
            icon: const Icon(Icons.history),
          ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          TextField(
            controller: _serverCtrl,
            decoration: const InputDecoration(
              labelText: 'Link server',
              hintText: 'http://192.168.1.75:5000',
              border: OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 12),
          FilledButton.icon(
            onPressed: provider.isLoading
                ? null
                : () => context.read<AppProvider>().setServerUrl(_serverCtrl.text),
            icon: const Icon(Icons.cloud_done_outlined),
            label: const Text('Cap nhat server'),
          ),
          const SizedBox(height: 12),
          FilledButton.icon(
            onPressed:
                provider.isLoading ? null : () => _pickAndAnalyzeVideo(context),
            icon: const Icon(Icons.cloud_upload_outlined),
            label: Text(provider.isLoading ? 'Dang xu ly...' : 'Chon video'),
          ),
          if (provider.error != null)
            Padding(
              padding: const EdgeInsets.only(top: 12),
              child: Text(
                provider.error!,
                style: const TextStyle(color: Colors.red),
              ),
            ),
          if (data != null) ...[
            if (data['preview_url'] != null)
              Padding(
                padding: const EdgeInsets.only(top: 16),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(18),
                  child: Image.network(
                    data['preview_url'] as String,
                    fit: BoxFit.cover,
                  ),
                ),
              ),
            _InfoCard(
              title: 'Ket qua video',
              content: provider.videoSummary(data),
            ),
          ],
        ],
      ),
    );
  }
}

class HistoryScreen extends StatelessWidget {
  const HistoryScreen({super.key, required this.title, required this.entries});

  final String title;
  final List<HistoryEntry> entries;

  bool _isImageLikeUrl(String? url) {
    final value = (url ?? '').toLowerCase();
    return value.endsWith('.jpg') ||
        value.endsWith('.jpeg') ||
        value.endsWith('.png') ||
        value.endsWith('.webp');
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(title)),
      body: entries.isEmpty
          ? const Center(child: Text('Chua co ket qua'))
          : ListView.separated(
              padding: const EdgeInsets.all(16),
              itemCount: entries.length,
              separatorBuilder: (_, __) => const SizedBox(height: 12),
              itemBuilder: (_, index) {
                final entry = entries[index];
                final previewUrl = entry.previewUrl ?? entry.resultUrl;
                final hasPreview = _isImageLikeUrl(previewUrl);
                return Material(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(20),
                  child: InkWell(
                    borderRadius: BorderRadius.circular(20),
                    onTap: () => Navigator.of(context).push(MaterialPageRoute(
                      builder: (_) => HistoryDetailScreen(entry: entry),
                    )),
                    child: Padding(
                      padding: const EdgeInsets.all(14),
                      child: Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Container(
                            width: 82,
                            height: 82,
                            decoration: BoxDecoration(
                              color: const Color(0xFFF2F6F3),
                              borderRadius: BorderRadius.circular(16),
                            ),
                            clipBehavior: Clip.antiAlias,
                            child: hasPreview
                                ? Image.network(previewUrl!, fit: BoxFit.cover)
                                : const Icon(
                                    Icons.history,
                                    color: Color(0xFF1E6F5C),
                                  ),
                          ),
                          const SizedBox(width: 14),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  entry.title,
                                  style: const TextStyle(
                                    fontSize: 16,
                                    fontWeight: FontWeight.w700,
                                  ),
                                ),
                                const SizedBox(height: 6),
                                Text(
                                  entry.timestamp,
                                  style: const TextStyle(
                                    color: Colors.black54,
                                    fontSize: 12,
                                  ),
                                ),
                                const SizedBox(height: 8),
                                Text(
                                  entry.summary,
                                  maxLines: 4,
                                  overflow: TextOverflow.ellipsis,
                                ),
                              ],
                            ),
                          ),
                          const SizedBox(width: 8),
                          const Icon(Icons.chevron_right),
                        ],
                      ),
                    ),
                  ),
                );
              },
            ),
    );
  }
}

class HistoryDetailScreen extends StatelessWidget {
  const HistoryDetailScreen({super.key, required this.entry});

  final HistoryEntry entry;

  bool get _isImageLike {
    final url = (entry.previewUrl ?? entry.resultUrl ?? '').toLowerCase();
    return url.endsWith('.jpg') ||
        url.endsWith('.jpeg') ||
        url.endsWith('.png') ||
        url.endsWith('.webp');
  }

  @override
  Widget build(BuildContext context) {
    final mediaUrl = entry.previewUrl ?? entry.resultUrl;

    return Scaffold(
      appBar: AppBar(title: Text(entry.title)),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          if (mediaUrl != null && _isImageLike)
            ClipRRect(
              borderRadius: BorderRadius.circular(18),
              child: Image.network(mediaUrl, fit: BoxFit.cover),
            ),
          _InfoCard(title: 'Thoi gian', content: entry.timestamp),
          _InfoCard(title: 'Tom tat', content: entry.summary),
          if (entry.resultUrl != null)
            _InfoCard(title: 'Link ket qua', content: entry.resultUrl!),
        ],
      ),
    );
  }
}

class _FeatureTile extends StatelessWidget {
  const _FeatureTile({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.onTap,
  });

  final IconData icon;
  final String title;
  final String subtitle;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 14),
      child: Material(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        child: InkWell(
          borderRadius: BorderRadius.circular(20),
          onTap: onTap,
          child: Padding(
            padding: const EdgeInsets.all(18),
            child: Row(
              children: [
                Container(
                  width: 54,
                  height: 54,
                  decoration: BoxDecoration(
                    color: const Color(0xFFEAF3EF),
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Icon(icon, color: const Color(0xFF1E6F5C)),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        title,
                        style: const TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        subtitle,
                        style: const TextStyle(color: Colors.black54),
                      ),
                    ],
                  ),
                ),
                const Icon(Icons.chevron_right),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _InfoCard extends StatelessWidget {
  const _InfoCard({required this.title, required this.content});

  final String title;
  final String content;

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(top: 16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w700),
          ),
          const SizedBox(height: 8),
          Text(content),
        ],
      ),
    );
  }
}