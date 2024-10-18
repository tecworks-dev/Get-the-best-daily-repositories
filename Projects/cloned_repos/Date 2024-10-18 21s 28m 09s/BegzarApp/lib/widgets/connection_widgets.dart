import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';

class ConnectionWidget extends StatelessWidget {
  const ConnectionWidget(
      {super.key,
      required this.onTap,
      required this.isLoading,
      required this.status});

  final bool isLoading;
  final GestureTapCallback onTap;
  final String status;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        InkWell(
          onTap: isLoading ? null : onTap,
          customBorder: CircleBorder(),
          child: Container(
            height: 110,
            width: 110,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              // color: Colors.white,
            ),
            child: Center(
              child: Icon(
                CupertinoIcons.power,
                color: isLoading
                    ? Colors.yellow
                    : status == "DISCONNECTED"
                        ? Colors.red
                        : Colors.green,
                size: 90,
              ),
            ),
          ),
        ),
        SizedBox(height: 8),
        Text(
          isLoading
              ? 'در حال اتصال ...'
              : status == "DISCONNECTED"
                  ? 'برای اتصال کلیک کنید.'
                  : 'متصل شدید !',
          style: TextStyle(fontSize: 16),
        ),
      ],
    );
  }
}
