import Vue from 'vue'
import App from './App.vue'
import vuetify from './plugins/vuetify';
import websocket from 'vue-native-websocket'

Vue.config.productionTip = false

Vue.use(websocket, 'ws://localhost:6060', {
reconnection: true,
reconnectionAttempts: 5,
reconnectionDelay: 3000
})

new Vue({
  vuetify,
  render: h => h(App)
}).$mount('#app')
