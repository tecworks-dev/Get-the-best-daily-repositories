
class EventBus {
  listeners:any = {}
  constructor() {
    this.listeners = {};
  }

  on(eventName:string, callback:any) {
    if (!this.listeners[eventName]) {
      this.listeners[eventName] = [];
    }
    this.listeners[eventName].push(callback);
  }

  emit(eventName:string, data:any) {
    const callbacks = this.listeners[eventName];
    if (callbacks) {
      callbacks.forEach((callback: (arg0: any) => any) => callback(data));
    }
  }
}
const eventBus = new EventBus()
export default eventBus;