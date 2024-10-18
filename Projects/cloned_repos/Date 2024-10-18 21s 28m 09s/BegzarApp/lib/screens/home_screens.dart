import 'dart:convert';

import 'package:awesome_dialog/awesome_dialog.dart';
import 'package:begzar/common/http_client.dart';
import 'package:begzar/common/utils.dart';
import 'package:begzar/widgets/connection_widgets.dart';
import 'package:begzar/widgets/drawer_widget.dart';
import 'package:begzar/widgets/option_tile_widget.dart';
import 'package:begzar/widgets/server_selection_modal_widget.dart';
import 'package:begzar/widgets/vpn_status.dart';
import 'package:device_info_plus/device_info_plus.dart';
import 'package:dio/dio.dart';
import 'package:encrypt/encrypt.dart' as enc;
import 'package:firebase_analytics/firebase_analytics.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:flutter_v2ray/flutter_v2ray.dart';
import 'package:package_info_plus/package_info_plus.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:url_launcher/url_launcher.dart';
import '../common/theme.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  var v2rayStatus = ValueNotifier<V2RayStatus>(V2RayStatus());
  late final FlutterV2ray flutterV2ray = FlutterV2ray(
    onStatusChanged: (status) {
      v2rayStatus.value = status;
    },
  );

  bool proxyOnly = false;
  List<String> bypassSubnets = [];
  String? coreVersion;
  String? versionName;
  bool isLoading = false;
  final GlobalKey<ScaffoldState> _scaffoldKey = new GlobalKey<ScaffoldState>();
  int? connectedServerDelay;
  late SharedPreferences _prefs;
  String selectedServer = 'اتوماتیک';
  String? selectedServerLogo;
  String? domainName;
  bool isFetchingPing = false;

  @override
  void initState() {
    super.initState();
    getVersionName();
    _loadServerSelection();
    flutterV2ray
        .initializeV2Ray(
      notificationIconResourceType: "mipmap",
      notificationIconResourceName: "launcher_icon",
    )
        .then((value) async {
      coreVersion = await flutterV2ray.getCoreVersion();

      setState(() {});
      Future.delayed(
        Duration(seconds: 1),
        () {
          if (v2rayStatus.value.state == 'CONNECTED') {
            delay(); // اگر کاربر متصل است، دوباره delay را اجرا کنید
          }
        },
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      key: _scaffoldKey,
      appBar: AppBar(
        title: Text(
          'بِگذَر',
          style: TextStyle(color: ThemeColor.foregroundColor, fontSize: 18),
        ),
        leading: IconButton(
          onPressed: () => _scaffoldKey.currentState!.openDrawer(),
          icon: Icon(
            Icons.menu,
            color: ThemeColor.foregroundColor,
          ),
        ),
        actions: [
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Image.asset(
              'assets/images/logo_transparent.png',
              color: ThemeColor.foregroundColor,
              height: 50,
            ),
          ),
        ],
        centerTitle: true,
        backgroundColor: ThemeColor.backgroundColor,
        elevation: 0,
      ),
      drawer: DrawerWidget(),
      backgroundColor: ThemeColor.backgroundColor,
      body: Stack(
        children: [
          Padding(
            padding: const EdgeInsets.only(bottom: 100),
            child: Center(
              child: ValueListenableBuilder(
                valueListenable: v2rayStatus,
                builder: (context, value, child) {
                  return Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    crossAxisAlignment: CrossAxisAlignment.center,
                    children: [
                      ConnectionWidget(
                        onTap: () async {
                          if (value.state == "DISCONNECTED") {
                            getDomain();
                          } else {
                            await FirebaseAnalytics.instance.logEvent(
                              name: "Disconnected",
                            );
                            flutterV2ray.stopV2Ray();
                          }
                        },
                        isLoading: isLoading,
                        status: value.state,
                      ),
                      SizedBox(height: 10),
                      if (value.state == 'CONNECTED')
                        Container(
                          width: connectedServerDelay == null ? 220 : 80,
                          height: 30,
                          child: Center(
                            child: isFetchingPing == false
                                ? CupertinoActivityIndicator()
                                : InkWell(
                                    borderRadius: BorderRadius.circular(12),
                                    onTap:
                                        delay, // به شما اجازه می‌دهد دوباره پینگ را چک کنید
                                    child: Center(
                                      child: Row(
                                        mainAxisAlignment:
                                            MainAxisAlignment.center,
                                        crossAxisAlignment:
                                            CrossAxisAlignment.center,
                                        children: [
                                          Text('ms'),
                                          SizedBox(width: 2),
                                          Text(
                                            connectedServerDelay.toString(),
                                          ),
                                          SizedBox(width: 6),
                                          Icon(CupertinoIcons.wifi,
                                              color: Colors.white, size: 16),
                                        ],
                                      ),
                                    ),
                                  ),
                          ),
                        ),
                    ],
                  );
                },
              ),
            ),
          ),
          Align(
            alignment: Alignment.bottomCenter,
            child: ValueListenableBuilder(
              valueListenable: v2rayStatus,
              builder: (context, value, child) {
                return Padding(
                  padding: const EdgeInsets.all(12.0),
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.end,
                    children: [
                      if (value.state == 'CONNECTED')
                        VpnCard(
                          downloadSpeed: value.download,
                          uploadSpeed: value.upload,
                          selectedServer: selectedServer,
                          selectedServerLogo:
                              selectedServerLogo ?? 'assets/images/auto.png',
                          duration: value.duration,
                        ),
                      if (value.state == 'DISCONNECTED')
                        Padding(
                          padding: const EdgeInsets.symmetric(horizontal: 25),
                          child: Column(
                            children: [
                              GestureDetector(
                                onTap: () {
                                  showModalBottomSheet(
                                    context: context,
                                    shape: RoundedRectangleBorder(
                                      borderRadius: BorderRadius.vertical(
                                          top: Radius.circular(25.0)),
                                    ),
                                    builder: (BuildContext context) {
                                      return ServerSelectionModal(
                                        selectedServer: selectedServer,
                                        onServerSelected: (server) {
                                          String? logoPath;
                                          if (server == 'اتوماتیک') {
                                            logoPath = 'assets/images/auto.png';
                                          } else if (server == 'همراه اول') {
                                            logoPath = 'assets/images/mci.png';
                                          } else if (server == 'ایرانسل') {
                                            logoPath = 'assets/images/mtn.png';
                                          }
                                          setState(() {
                                            selectedServer = server;
                                          });
                                          _saveServerSelection(
                                              server, logoPath!);

                                          Navigator.pop(context);
                                        },
                                      );
                                    },
                                  );
                                },
                                child: OptionTile(
                                  icon: Icons.settings,
                                  title: 'انتخاب سرور',
                                  subtitle: selectedServer,
                                ),
                              ),
                            ],
                          ),
                        ),
                      SizedBox(height: 30),
                      Column(
                        children: [
                          Text('نسخه : ${versionName}'),
                          Visibility(
                            visible: coreVersion != null,
                            child: Text(
                              coreVersion ?? '',
                              style: TextStyle(fontSize: 12),
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  String getServerParam() {
    if (selectedServer == 'همراه اول') {
      return 'mci';
    } else if (selectedServer == 'ایرانسل') {
      return 'mtn';
    } else {
      return 'auto';
    }
  }

  Future<void> _loadServerSelection() async {
    _prefs = await SharedPreferences.getInstance();
    setState(() {
      selectedServer = _prefs.getString('selectedServer') ?? 'اتوماتیک';
      selectedServerLogo = _prefs.getString('selectedServerLogo') ??
          'assets/images/auto.png'; // مقداردهی پیش‌فرض
    });
  }

  Future<void> _saveServerSelection(String server, String logoPath) async {
    await _prefs.setString('selectedServer', server);
    await _prefs.setString('selectedServerLogo', logoPath);
    setState(() {
      selectedServer = server;
      selectedServerLogo = logoPath;
    });
  }

  Future<List<String>> getDeviceArchitecture() async {
    DeviceInfoPlugin deviceInfo = DeviceInfoPlugin();
    AndroidDeviceInfo androidInfo = await deviceInfo.androidInfo;
    return androidInfo.supportedAbis;
  }

  void getVersionName() async {
    PackageInfo packageInfo = await PackageInfo.fromPlatform();
    setState(() {
      versionName = packageInfo.version;
    });
  }

  String decrypt(String data) {
    final split = data.split(':');
    final key = enc.Key.fromUtf8(Utils.enc_key);
    final ivBase64 = split[0];
    final ivBytes = base64.decode(ivBase64);
    final iv = enc.IV(ivBytes);

    // Setup decrypter
    final decrypter =
        enc.Encrypter(enc.AES(key, mode: enc.AESMode.ctr, padding: null));

    // Base64-encoded ciphertext
    final base64Ciphertext = split[1];
    final ciphertext = enc.Encrypted.fromBase64(base64Ciphertext);

    // Decrypting the data
    try {
      final decrypted = decrypter.decryptBytes(ciphertext, iv: iv);
      final decryptedData = utf8.decode(decrypted);
      return decryptedData;
    } catch (e) {
      return 'Error during decryption: $e';
    }
  }

  Future<void> getDomain() async {
    try {
      setState(() {
        isLoading = true;
      });
      final response = await httpClient.get('/');
      final decryptData = decrypt(response.data).replaceAll('"', '');
      domainName = decryptData;
      checkUpdate();
    } catch (e) {
      if (mounted) {
        setState(() {
          isLoading = false;
        });
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('خطا در دریافت اطلاعات، لطفاً مجدداً تلاش کنید.',
                textDirection: TextDirection.rtl),
          ),
        );
      }
    }
  }

  void checkUpdate() async {
    try {
      final serverParam = getServerParam();
      final response =
          await Dio().get("https://$domainName/api/?sv=$serverParam");
      final data_base64 = utf8.decode(base64.decode(response.data));
      final decode_json = jsonDecode(data_base64);
      final dec = jsonDecode(decrypt(decode_json['data']));
      final version = dec['version'];
      final serverEncode = dec['server'];

      if (version == versionName) {
        // نسخه جدید نیست، ادامه بده
        final List<String> serverList = await fetchServers(serverEncode);
        await connect(serverList);
      } else {
        // دریافت معماری دستگاه
        List<String> getArchitecture = await getDeviceArchitecture();
        String updateUrl = ''; // متغیر برای ذخیره URL درست

        // چک کردن اینکه دستگاه ARM64-v8a است یا نه
        bool isV8aDevice = getArchitecture.contains("arm64-v8a");
        bool isV7aInstalled = false;

        // چک کردن اگر نسخه ARM-v7a روی دستگاه ARM64-v8a نصب شده است
        for (String element in getArchitecture) {
          if (element == "armeabi-v7a") {
            isV7aInstalled = true;
          }
        }

        // منطق جدید: اگر دستگاه ARM64-v8a است ولی نسخه ARM-v7a نصب شده، لینک v7a را بده
        if (isV7aInstalled && isV8aDevice) {
          updateUrl = dec['url-v7a'];
        } else {
          for (String element in getArchitecture) {
            if (element == "arm64-v8a") {
              updateUrl = dec['url-v8a'];
              break;
            } else if (element == "armeabi-v7a") {
              updateUrl = dec['url-v7a'];
              break;
            }
          }
        }

        if (updateUrl.isNotEmpty) {
          AwesomeDialog(
            context: context,
            dialogType: DialogType.warning,
            animType: AnimType.rightSlide,
            title: 'آپدیت جدید',
            desc: 'برای دانلود ورژن جدید روی دکمه دانلود کلیک کنید',
            dialogBackgroundColor: Colors.white,
            btnCancelOnPress: () {},
            btnOkOnPress: () async {
              await launchUrl(Uri.parse(utf8.decode(base64Decode(updateUrl))),
                  mode: LaunchMode.externalApplication);
            },
            btnOkText: 'دانلود',
            btnCancelText: 'بستن',
            buttonsTextStyle:
                TextStyle(fontFamily: 'sm', color: Colors.white, fontSize: 14),
            titleTextStyle:
                TextStyle(fontFamily: 'sb', color: Colors.black, fontSize: 16),
            descTextStyle:
                TextStyle(fontFamily: 'sm', color: Colors.black, fontSize: 14),
          )..show();
        } else {
          if (mounted) {
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(
                content: Text('نسخه ی مجاز آپدیت برای گوشی شما یافت نشد !',
                    textDirection: TextDirection.rtl),
              ),
            );
          }
        }
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('خطا در بررسی نسخه اپلیکیشن، لطفاً مجدداً تلاش کنید.',
                textDirection: TextDirection.rtl),
          ),
        );
      }
    } finally {
      setState(() {
        isLoading = false;
      });
    }
  }

  Future<List<String>> fetchServers(String serverEncode) async {
    try {
      final List<String> serverList =
          LineSplitter.split(utf8.decode(base64Decode(serverEncode))).toList();
      return serverList;
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text(
                'خطا در دریافت اطلاعات از سرور، لطفاً مجدداً تلاش کنید.',
                textDirection: TextDirection.rtl),
          ),
        );
      }
      return [];
    }
  }

  Future<void> connect(List<String> serverList) async {
    if (serverList.isEmpty) {
      // سرور یافت نشد
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text(
                'سروری برای اتصال یافت نشد. دقایقی دیگر مجدداً تلاش کنید.',
                textDirection: TextDirection.rtl),
          ),
        );
      }
      setState(() {
        isLoading = false;
      });
      return;
    }

    setState(() {
      isLoading = true;
    });

    List<String> list = [];

    serverList.forEach((element) {
      final V2RayURL v2rayURL = FlutterV2ray.parseFromURL(element);
      list.add(v2rayURL.getFullConfiguration());
    });

    Map<String, dynamic> getAllDelay =
        jsonDecode(await flutterV2ray.getAllServerDelay(configs: list));

    list.clear();

    int minPing = 99999999;
    String bestConfig = '';

    getAllDelay.forEach(
      (key, value) {
        if (value < minPing && value != -1) {
          setState(() {
            bestConfig = key;
            minPing = value;
          });
        }
      },
    );

    if (bestConfig.isNotEmpty) {
      if (await flutterV2ray.requestPermission()) {
        await FirebaseAnalytics.instance.logEvent(
          name: "Connected",
        );
        flutterV2ray.startV2Ray(
          remark: "فیلترشکن بِگذَر",
          config: bestConfig,
          proxyOnly: false,
          bypassSubnets: null,
          notificationDisconnectButtonName: 'قطع اتصال',
        );
      } else {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('عدم دسترسی برای اتصال فیلترشکن',
                  textDirection: TextDirection.rtl),
            ),
          );
        }
      }
    } else {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text(
                'هیچ سروری برای اتصال یافت نشد. لطفاً دقایقی دیگر مجدداً تلاش کنید.',
                textDirection: TextDirection.rtl),
          ),
        );
      }
    }
    Future.delayed(
      Duration(seconds: 1),
      () {
        delay();
      },
    );
    setState(() {
      isLoading = false;
    });
  }

  void delay() async {
    if (v2rayStatus.value.state == 'CONNECTED') {
      connectedServerDelay = await flutterV2ray.getConnectedServerDelay();
      setState(() {
        isFetchingPing = true;
      });
    }
    if (!mounted) return;
  }
}
