package app.termora

/**
 * 将在 JVM 进程退出时释放
 */
class ApplicationDisposable : Disposable {
    companion object {
        val instance by lazy { ApplicationDisposable() }
    }
}