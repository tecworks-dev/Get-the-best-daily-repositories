import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("java-library")
    alias(libs.plugins.kotlin.jvm)
    id("com.github.johnrengelman.shadow") version "8.1.1"
}

java {
    sourceCompatibility = JavaVersion.VERSION_11
    targetCompatibility = JavaVersion.VERSION_11
}

tasks.jar {
    manifest.attributes["Main-Class"] = "io.github.duzhaokun123.yapatch.patch.Main"
}

kotlin {
    compilerOptions {
        jvmTarget = JvmTarget.JVM_11
    }
}

dependencies {
    implementation(libs.beust.jcommander)
    implementation(libs.zip4j)
    implementation("com.android.tools.build:apkzlib:8.7.1")
    implementation("com.google.code.gson:gson:2.11.0")
    implementation(fileTree(mapOf("dir" to "libs", "include" to listOf("*.jar"))))
}