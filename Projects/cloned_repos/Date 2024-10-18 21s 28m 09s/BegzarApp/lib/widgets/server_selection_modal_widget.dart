import 'package:flutter/material.dart';

class ServerSelectionModal extends StatelessWidget {
  final String selectedServer;
  final Function(String) onServerSelected;

  ServerSelectionModal(
      {required this.selectedServer, required this.onServerSelected});

  @override
  Widget build(BuildContext context) {
    return Directionality(
      textDirection: TextDirection.rtl,
      child: Container(
        padding: EdgeInsets.all(20),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'انتخاب سرور',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            SizedBox(height: 20),
            ListTile(
              leading: Image.asset('assets/images/auto.png', width: 32),
              title: Text('اتوماتیک'),
              subtitle: Text('بر روی تمامی اپراتور ها متصل میشود.'),
              trailing: selectedServer == 'اتوماتیک'
                  ? Icon(Icons.check, color: Colors.green)
                  : null,
              onTap: () => onServerSelected('اتوماتیک'),
            ),
            Divider(),
            ListTile(
              // leading: Icon(Icons.flag, color: Colors.white, size: 32),
              leading: Image.asset('assets/images/mci.png', width: 32, color: Colors.white),
              title: Text('همراه اول'),
              subtitle: Text('سرورهای مخصوص اپراتور همراه اول'),
              trailing: selectedServer == 'همراه اول'
                  ? Icon(Icons.check, color: Colors.green)
                  : null,
              onTap: () => onServerSelected('همراه اول'),
            ),
            ListTile(
              leading: Image.asset('assets/images/mtn.png', width: 32, color: Colors.white),
              title: Text('ایرانسل'),
              subtitle: Text('سرورهای مخصوص اپراتور ایرانسل'),
              trailing: selectedServer == 'ایرانسل'
                  ? Icon(Icons.check, color: Colors.green)
                  : null,
              onTap: () => onServerSelected('ایرانسل'),
            ),
          ],
        ),
      ),
    );
  }
}