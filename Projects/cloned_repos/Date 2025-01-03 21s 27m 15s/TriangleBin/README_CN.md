# TriangleBin

[English](https://github.com/Swung0x48/TriangleBin/blob/main/README.md) | 中文

一个渲染顺序测试器（看看你的 GPU 是 IMR 还是 TBR！）

## 第三方库
[ImGui](https://github.com/ocornut/imgui.git)

[SDL](https://github.com/libsdl-org/SDL.git)

[glew](https://github.com/nigels-com/glew.git)

ImGui/SDL 框架代码来自 [sfalexrog/Imgui_Android](https://github.com/sfalexrog/Imgui_Android.git)

## 许可协议
MIT

## 说明
在 [这里](https://github.com/Swung0x48/TriangleBin/releases) 下载可执行文件

包括一个 Android apk 和一个 Windows exe.

`Rendered/Screen %` - 渲染的像素 / 屏幕画布像素数，用百分比表示。

`Tris` - 三角形数量。（除去两个全屏的三角形）

`Auto Increment` - 自动增加 `Rendered/Screen %`.

`ppf` - 当 `Auto Increment` 勾选后，每帧自动增加到 `Rendered/Screen %` 的值.
