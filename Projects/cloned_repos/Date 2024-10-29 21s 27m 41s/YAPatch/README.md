# YAPatch

Yet Another Patching Tool for Android to load xposed modules

## Build

use https://github.com/Reginer/aosp-android-jar/tree/main/android-35 android.jar

```shell
./gradlew patch-loader:copyFiles patch:shadowJar
```

## 可用性

已测试可用的 应用-模块,[模块] 有

- QQ-QAuxiliary
- 哔哩哔哩-哔哩漫游

不可用的 应用 有

- 菜鸟
- 知乎

## 已知问题

存在这个垃圾项目

## 主要感谢
- [Pine](https://github.com/canyie/pine)
- [LSPatch](https://github.com/LSPosed/LSPatch)
- [Xpatch](https://github.com/WindySha/Xpatch)
