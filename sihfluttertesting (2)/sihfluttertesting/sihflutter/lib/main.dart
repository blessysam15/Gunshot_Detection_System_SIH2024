import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Gunshot Detection',
      home: const GunshotDetection(),
    );
  }
}

class GunshotDetection extends StatefulWidget {
  const GunshotDetection({Key? key}) : super(key: key);

  @override
  _GunshotDetectionState createState() => _GunshotDetectionState();
}

class _GunshotDetectionState extends State<GunshotDetection> {
  Map<String, dynamic>? detectionData;
  late Timer timer;

  Future<void> fetchDetectionData() async {
    // Replace this URL with your Flask server's IP address
    final response = await http.get(Uri.parse('http://192.168.19.237:5000/detection'));
    if (response.statusCode == 200) {
      setState(() {
        detectionData = json.decode(response.body);
      });
    } else {
      throw Exception('Failed to load data');
    }
  }

  @override
  void initState() {
    super.initState();
    fetchDetectionData();
    timer = Timer.periodic(const Duration(seconds: 7), (timer) => fetchDetectionData());
  }

  @override
  void dispose() {
    timer.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Gunshot Detection')),
      body: detectionData == null
          ? const Center(child: CircularProgressIndicator())
          : Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Timestamp: ${detectionData!["timestamp"]}'),
                  Text('Predicted Label: ${detectionData!["predicted_label"]}'),
                  Text('Confidence Scores: ${detectionData!["confidence_scores"]}')
                ],
              ),
            ),
    );
  }
}
