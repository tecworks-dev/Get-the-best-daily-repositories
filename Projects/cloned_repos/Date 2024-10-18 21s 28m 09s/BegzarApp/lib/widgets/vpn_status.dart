import 'package:flutter/material.dart';

class VpnCard extends StatelessWidget {
  final int downloadSpeed;
  final int uploadSpeed;
  final String selectedServer;
  final String selectedServerLogo;
  final String duration;

  const VpnCard(
      {super.key,
      required this.downloadSpeed,
      required this.uploadSpeed,
      required this.selectedServer,
      required this.selectedServerLogo,
      required this.duration});
  @override
  Widget build(BuildContext context) {
    return Stack(
      clipBehavior: Clip.none,
      alignment: Alignment.center,
      children: [
        Positioned(
          top: -30,
          child: Container(
            width: 200,
            height: 50,
            decoration: BoxDecoration(
              color: Colors.grey.shade300,
              borderRadius: BorderRadius.circular(6),
            ),
            child: Center(
              child: Padding(
                padding: const EdgeInsets.only(bottom: 16),
                child: Text(
                  duration,
                  style: TextStyle(
                    color: Colors.grey.shade600,
                    fontFamily: 'sb',
                    fontSize: 18,
                  ),
                ),
              ),
            ),
          ),
        ),
        Container(
          width: 350,
          padding: EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: Colors.grey.shade300,
            borderRadius: BorderRadius.circular(16),
            boxShadow: [
              BoxShadow(
                color: Colors.black26,
                blurRadius: 10,
                offset: Offset(0, 4),
              ),
            ],
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.start,
                    children: [
                      Image.asset(
                        selectedServerLogo,
                        width: 40,
                      ),
                      SizedBox(width: 8),
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              Text(
                                selectedServer,
                                style: TextStyle(
                                  color: Colors.grey.shade600,
                                  fontSize: 20,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ],
                          ),
                          SizedBox(height: 8),
                          Text(
                            'فیلترشکن رایگان بِگذَر',
                            style: TextStyle(
                              color: Colors.grey.shade600,
                              fontSize: 14,
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ],
              ),
              Divider(color: Colors.black26),
              Padding(
                padding: const EdgeInsets.only(
                    right: 25, left: 25, bottom: 10, top: 10),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Row(
                      children: [
                        Icon(
                          Icons.arrow_downward,
                          color: Colors.green,
                          size: 30,
                        ),
                        SizedBox(width: 8),
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              formatBytes(downloadSpeed),
                              style: TextStyle(
                                fontSize: 18,
                                color: Colors.grey.shade600,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                            Text(
                              'دانلود',
                              style: TextStyle(
                                color: Colors.grey.shade600,
                                fontSize: 16,
                              ),
                            ),
                          ],
                        )
                      ],
                    ),
                    Row(
                      children: [
                        Icon(
                          Icons.arrow_upward,
                          color: Colors.red,
                          size: 30,
                        ),
                        SizedBox(width: 8),
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              formatBytes(uploadSpeed),
                              style: TextStyle(
                                fontSize: 18,
                                color: Colors.grey.shade600,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                            Text(
                              'آپلود',
                              style: TextStyle(
                                color: Colors.grey.shade600,
                                fontSize: 16,
                              ),
                            ),
                          ],
                        )
                      ],
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  String formatBytes(int bytes) {
    if (bytes <= 0) return '0 Byte';

    const int kb = 1024;
    const int mb = kb * 1024;
    const int gb = mb * 1024;

    if (bytes < kb) return '$bytes Byte${bytes > 1 ? 's' : ''}';
    if (bytes < mb) return '${(bytes / kb).toStringAsFixed(2)} KB';
    if (bytes < gb) return '${(bytes / mb).toStringAsFixed(2)} MB';
    return '${(bytes / gb).toStringAsFixed(2)} GB';
  }
}
