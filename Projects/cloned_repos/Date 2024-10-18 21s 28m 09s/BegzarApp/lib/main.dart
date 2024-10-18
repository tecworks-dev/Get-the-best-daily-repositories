
import 'package:begzar/common/theme.dart';
import 'package:begzar/screens/home_screens.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:firebase_core/firebase_core.dart';
import 'firebase_options.dart';
import 'package:firebase_analytics/firebase_analytics.dart';

@pragma('vm:entry-point')
Future<void> _firebaseMessagingBackgroundHandler(RemoteMessage message) async {
  // If you're going to use other Firebase services in the background, such as Firestore,
  // make sure you call `initializeApp` before using other Firebase services.
  await Firebase.initializeApp();

  print("Handling a background message: ${message.messageId}");
}

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
  ]);
  SystemChrome.setSystemUIOverlayStyle(SystemUiOverlayStyle(
    systemNavigationBarColor:
        ThemeColor.backgroundColor, // Set your desired color here
    systemNavigationBarIconBrightness:
        Brightness.light, // Adjust icon brightness as needed
  ));
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );
  FirebaseAnalytics analytics = FirebaseAnalytics.instance;
  var isSupported = await analytics.isSupported();
  debugPrint('analytics supported is : $isSupported');
  final fcmToken = await FirebaseMessaging.instance.getToken();
  debugPrint('fcm token : $fcmToken');
  FirebaseMessaging.onBackgroundMessage(_firebaseMessagingBackgroundHandler);
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    final defaultTextStyle =
        TextStyle(fontFamily: 'sm', color: Color(0xffF7FAFF));
    return MaterialApp(
      title: 'Begzar VPN',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        textTheme: TextTheme(
          titleLarge: defaultTextStyle,
          titleMedium: defaultTextStyle,
          titleSmall: defaultTextStyle,
          bodyLarge: defaultTextStyle,
          bodyMedium: defaultTextStyle,
          bodySmall: defaultTextStyle,
          labelLarge: defaultTextStyle,
          labelMedium: defaultTextStyle,
          labelSmall: defaultTextStyle,
        ),
        useMaterial3: true,
        brightness: Brightness.dark,
        inputDecorationTheme: const InputDecorationTheme(
          border: OutlineInputBorder(),
        ),
      ),
      home: Directionality(
        textDirection: TextDirection.rtl,
        child: const HomePage(),
      ),
    );
  }
}
