import { sleep } from 'k6';
import http from 'k6/http';

export let options = {
  thresholds: {},
  scenarios: {
    Scenario_1: {
      executor: 'ramping-vus',
      gracefulStop: '30s',
      stages: [
        { target: 100, duration: '1m' },  // Ramp-up to 20 users in 1 minute
        { target: 100, duration: '30s' }, // Stay at 20 users for 30 seconds
        { target: 0, duration: '15s' },  // Ramp-down to 0 users in 15 seconds
      ],
      startVUs: 10,  // Start with 10 virtual users
      gracefulRampDown: '30s',
      exec: 'scenario_1',  // Specify the function to execute
    },
  },
};

export function scenario_1() {
  let response;

  // Send an HTTP GET request to the target URL
  response = http.get('https://public-cosmos-vms.cosmostation.io/?orgId=1&refresh=5s');

  // Simulate the user waiting for 0.1 seconds
  sleep(0.1);
}
