<h1 align="center">Privacy Radar</h1>
<h5 align="center"><i>Scope Version for HarmonyOS NEXT</i></h5>
<p align="center">隐私雷达通过模拟应用在真实权限使用环境下，恶意应用实际可以查看的手机数据内容。</p>

---

> [!TIP]
> 此项目推荐使用华为 [DevEco Studio](https://developer.huawei.com/consumer/cn/deveco-studio/) 运行

### 使用

> [!IMPORTANT]
> 此项目部署和编译需要在安装了鸿蒙 NEXT 的实机或 Huawei Simulator 虚拟机上进行。

> [!NOTE]
> 本项目使用了基于传统权限申请的方式获取了系统**相册**和**联系人**数据，需要申请对应的 ACL 受限权限。
> 参考 [申请使用受限权限](https://developer.huawei.com/consumer/cn/doc/harmonyos-guides-V5/declare-permissions-in-acl-V5)。
> 
> DevEco 可在 `File > Project Structure > Signing Configs` 中配置自动完成测试部署的 ACL 证书。

1. 克隆或下载此仓库至本地，并使用 DevEco Studio 打开。
2. 在 DevEco 内登录自己的华为开发者账号，IDE 将自动生成签名文件结构。可能需要确认 ACL 授权情况。
3. 将开启了开发者模式的 HarmonyOS NEXT 实机接入本地设备。
4. 待 IDE 链接后通过默认 entry 配置启动运行即可。

### 文件结构

项目主要代码内容位于 `entry/src/main/ets`。  
各检测页面逻辑位于 `entry/src/main/ets/pages`。 

