import worker from '../worker'

export * from './'

export default {
  fetch: worker.scheduled,
}
