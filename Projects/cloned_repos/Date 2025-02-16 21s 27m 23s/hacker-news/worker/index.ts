interface Env {
  HACKER_NEWS_WORKFLOW: Workflow
}

export default {
  async scheduled(event: ScheduledEvent, env: Env, ctx: ExecutionContext) {
    console.info('trigger event by:', event)

    const createWorkflow = async () => {
      const instance = await env.HACKER_NEWS_WORKFLOW.create()

      const instanceDetails = {
        id: instance.id,
        details: await instance.status(),
      }

      console.info('instance detail:', instanceDetails)
      return instanceDetails
    }

    ctx.waitUntil(createWorkflow())

    return new Response('create workflow success')
  },
}
