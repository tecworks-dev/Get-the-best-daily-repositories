package app.termora.sync

class SyncerProvider private constructor() {
    companion object {
        val instance by lazy { SyncerProvider() }
    }


    fun getSyncer(type: SyncType): Syncer {
        if (type == SyncType.GitHub) {
            return GitHubSyncer.instance
        } else if (type == SyncType.Gitee) {
            return GiteeSyncer.instance
        }else if (type == SyncType.GitLab) {
            return GitLabSyncer.instance
        }
        throw UnsupportedOperationException("Type $type is not supported.")
    }
}