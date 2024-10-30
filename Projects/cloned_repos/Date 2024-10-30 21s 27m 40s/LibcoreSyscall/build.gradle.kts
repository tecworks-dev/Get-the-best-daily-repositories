// top-level build file
tasks.register<Delete>("clean").configure {
    delete(rootProject.buildDir)
}
