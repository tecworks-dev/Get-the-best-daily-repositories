package healthcheck

import (
	"bytes"
	"net/http"
	"time"
)

const healthCheckerTimeInterval = 5 * time.Second

func healthCheckForCosmos(client *http.Client, url string, resultChan chan<- string) {
	var checkPath string = "/cosmos/base/tendermint/v1beta1/node_info"

	resp, err := client.Get((url + checkPath))
	if err != nil {
		resultChan <- ""
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusOK {
		resultChan <- url
		return
	} else {
		resultChan <- ""
		return
	}
}

func healthCheckForEthereum(client *http.Client, url string, resultChan chan<- string) {
	var checkPayload string = `{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}`
	resp, err := client.Post(url, "application/json", bytes.NewBuffer([]byte(checkPayload)))
	if err != nil {
		resultChan <- ""
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusOK {
		resultChan <- url
	} else {
		resultChan <- ""
	}
}

func FilterHealthEndpoints(endpoints []string, chaintype string) []string {
	newValidURLs := make([]string, 0)
	results := make(chan string, len(endpoints))
	client := &http.Client{Timeout: healthCheckerTimeInterval}

	for _, url := range endpoints {
		switch chaintype {
		case "cosmos":
			go healthCheckForCosmos(client, url, results)
		case "ethereum":
			go healthCheckForEthereum(client, url, results)
		}
	}

	for i := 0; i < len(endpoints); i++ {
		if healthEndpoint := <-results; healthEndpoint != "" {
			newValidURLs = append(newValidURLs, healthEndpoint)
		}
	}

	return newValidURLs
}

func healthCheckForCosmosRPC(client *http.Client, url string, resultChan chan<- string) {
	var checkPath string = "/status"

	resp, err := client.Get((url + checkPath))
	if err != nil {
		resultChan <- ""
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusOK {
		resultChan <- url
		return
	} else {
		resultChan <- ""
		return
	}
}

func healthCheckForEthereumRPC(client *http.Client, url string, resultChan chan<- string) {
	var checkPayload string = `{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}`
	resp, err := client.Post(url, "application/json", bytes.NewBuffer([]byte(checkPayload)))
	if err != nil {
		resultChan <- ""
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusOK {
		resultChan <- url
	} else {
		resultChan <- ""
	}
}

func FilterHealthRPCEndpoints(endpoints []string, chaintype string) []string {
	newValidURLs := make([]string, 0)
	results := make(chan string, len(endpoints))
	client := &http.Client{Timeout: healthCheckerTimeInterval}

	for _, url := range endpoints {
		switch chaintype {
		case "cosmos":
			go healthCheckForCosmosRPC(client, url, results)
		case "ethereum":
			go healthCheckForEthereumRPC(client, url, results)
		}
	}

	for i := 0; i < len(endpoints); i++ {
		if healthEndpoint := <-results; healthEndpoint != "" {
			newValidURLs = append(newValidURLs, healthEndpoint)
		}
	}

	return newValidURLs
}
