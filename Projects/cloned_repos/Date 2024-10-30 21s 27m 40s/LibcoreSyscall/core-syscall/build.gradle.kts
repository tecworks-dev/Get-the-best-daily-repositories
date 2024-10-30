@file:Suppress("UnstableApiUsage")

plugins {
    alias(libs.plugins.android.library)
}

android {
    namespace = "dev.tmpfs.libcoresyscall.core"

    compileSdk = libs.versions.compileSdk.get().toInt()

    defaultConfig {
        minSdk = 21
    }

    buildFeatures {
        viewBinding = false
        buildConfig = false
        resValues = false
    }

    compileOptions {
        targetCompatibility = JavaVersion.VERSION_1_8
        sourceCompatibility = JavaVersion.VERSION_1_8
    }

    dependencies {
        compileOnly(projects.hiddenapiStub)
        compileOnly(libs.androidx.annotation)
    }
}
