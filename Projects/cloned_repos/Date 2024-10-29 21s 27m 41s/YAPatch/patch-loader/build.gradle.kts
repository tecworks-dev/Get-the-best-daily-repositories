plugins {
   alias(libs.plugins.android.application)
}

android {
    namespace = "io.github.duzhaokun123.patch_loader"
    compileSdk = 35

    defaultConfig {
        minSdk = 28
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}

dependencies {
    implementation("top.canyie.pine:core:0.3.0")
    implementation("top.canyie.pine:xposed:0.2.0")
}

task("copyDex") {
    dependsOn("assembleRelease")

    doLast {
        val dexOutPath = "$buildDir/intermediates/dex/release/mergeDexRelease/classes.dex"
        val outPath = "${rootProject.projectDir}/patch/src/main/resources"
        copy {
            from(dexOutPath)
            rename("classes.dex", "patch.dex")
            into(outPath)
        }
        println("Patch dex has been copied to $outPath")
    }
}

task("copyFiles") {
    dependsOn("copyDex")
}