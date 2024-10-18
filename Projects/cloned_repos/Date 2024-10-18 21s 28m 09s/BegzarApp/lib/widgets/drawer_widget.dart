import 'package:begzar/common/theme.dart';
import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';

class DrawerWidget extends StatelessWidget {
  const DrawerWidget({
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    return Drawer(
      elevation: 0,
      backgroundColor: ThemeColor.backgroundColor,
      child: ListView(
        // Important: Remove any padding from the ListView.
        padding: EdgeInsets.zero,
        children: [
          DrawerHeader(
            decoration: BoxDecoration(
              color: ThemeColor.backgroundColor,
            ),
            child: Column(
              children: [
                SizedBox(
                  height: 80,
                  child: Image.asset(
                    'assets/images/logo_transparent.png',
                    color: ThemeColor.foregroundColor,
                  ),
                ),
                SizedBox(height: 16),
                Text('بِگذَر')
              ],
            ),
          ),
          ListTile(
            title: Text(
              'کانال تلگرام',
              style: TextStyle(color: ThemeColor.foregroundColor, fontSize: 12),
            ),
            leading: Icon(
              Icons.telegram,
              color: ThemeColor.foregroundColor,
            ),
            onTap: () async {
              await launchUrl(Uri.parse('https://t.me/BegzarVPN'),
                  mode: LaunchMode.externalApplication);
            },
          ),
          ListTile(
            title: Text(
              'ایمیل : info@begzar.xyz',
              style: TextStyle(color: ThemeColor.foregroundColor, fontSize: 12),
            ),
            leading: Icon(
              Icons.email,
              color: ThemeColor.foregroundColor,
            ),
            onTap: () async {
              final Uri emailLaunchUri = Uri(
                scheme: 'mailto',
                path: 'info@begzar.xyz',
              );
              await launchUrl(emailLaunchUri);
            },
          ),
        ],
      ),
    );
  }
}
