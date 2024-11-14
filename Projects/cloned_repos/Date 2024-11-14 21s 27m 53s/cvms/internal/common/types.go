package common

import (
	"io"

	"github.com/cosmostation/cvms/internal/helper/logger"
	"github.com/go-resty/resty/v2"
	"github.com/sirupsen/logrus"
)

// TODO: All Methods in VoteIndexer, we need to add here?
type IIndexer interface {
	Start() error
	Loop(lastIndexPointerHeight int64)
	FetchValidatorInfoList() error
}

// TODO
type ICollector interface {
	Start(p Packager) error
}

// Collector Function Sig
type CollectorStart func(Packager) error
type CollectorLoop func(*Exporter, Packager)

// Client
type ClientType int

const (
	RPC ClientType = iota
	API
	GRPC
)

// Methods
type Method int

const (
	GET Method = iota
	POST
)

// Application Mode

type Mode int

const (
	INVALID_APP Mode = -1   // Invalid Case
	NETWORK     Mode = iota // Network Mode to provide network status overview
	VALIDATOR               // Validator Mode to provide whole chains' status overview about validator
)

func (a Mode) String() string {
	switch {
	case a == NETWORK:
		return "Validator Monitoring System"
	case a == VALIDATOR:
		return "White List"
	default:
		return "Invalid Mode"
	}
}

type CommonClient struct {
	RPCClient  *resty.Client
	APIClient  *resty.Client
	GRPCClient *resty.Client
	*logrus.Entry
}
type CommonApp struct {
	CommonClient
	EndPoint string
	// optional client
	OptionalClient CommonClient
}

func NewCommonApp(p Packager) CommonApp {
	restyLogger := logrus.New()
	restyLogger.Out = io.Discard
	rpcClient := resty.New().
		SetRetryCount(retryCount).
		SetRetryWaitTime(retryMaxWaitTimeDuration).
		SetRetryMaxWaitTime(retryMaxWaitTimeDuration).
		SetLogger(restyLogger)
	apiClient := resty.New().
		SetRetryCount(retryCount).
		SetRetryWaitTime(retryMaxWaitTimeDuration).
		SetRetryMaxWaitTime(retryMaxWaitTimeDuration).
		SetLogger(restyLogger)
	grpcClient := resty.New().
		SetRetryCount(retryCount).
		SetRetryWaitTime(retryMaxWaitTimeDuration).
		SetRetryMaxWaitTime(retryMaxWaitTimeDuration).
		SetLogger(restyLogger)
	entry := p.Logger.WithFields(
		logrus.Fields{
			logger.FieldKeyChain:   p.ChainName,
			logger.FieldKeyChainID: p.ChainID,
			logger.FieldKeyPackage: p.Package,
		})
	commonClient := CommonClient{rpcClient, apiClient, grpcClient, entry}
	return CommonApp{
		commonClient,
		"",
		CommonClient{},
	}
}

func (c *CommonClient) SetRPCEndPoint(endpoint string) *resty.Client {
	return c.RPCClient.SetBaseURL(endpoint)
}

func (c *CommonClient) GetRPCEndPoint() string {
	return c.RPCClient.BaseURL
}

func (c *CommonClient) SetAPIEndPoint(endpoint string) *resty.Client {
	return c.APIClient.SetBaseURL(endpoint)
}

func (c *CommonClient) GetAPIEndPoint() string {
	return c.APIClient.BaseURL
}

func (c *CommonClient) SetGRPCEndPoint(endpoint string) *resty.Client {
	return c.GRPCClient.SetBaseURL(endpoint)
}

func (c *CommonClient) GetGRPCEndPoint() string {
	return c.GRPCClient.BaseURL
}

func NewOptionalClient(entry *logrus.Entry) CommonClient {
	restyLogger := logrus.New()
	restyLogger.Out = io.Discard
	rpcClient := resty.New().
		SetRetryCount(retryCount).
		SetRetryWaitTime(retryMaxWaitTimeDuration).
		SetRetryMaxWaitTime(retryMaxWaitTimeDuration).
		SetLogger(restyLogger)
	apiClient := resty.New().
		SetRetryCount(retryCount).
		SetRetryWaitTime(retryMaxWaitTimeDuration).
		SetRetryMaxWaitTime(retryMaxWaitTimeDuration).
		SetLogger(restyLogger)
	return CommonClient{rpcClient, apiClient, nil, entry}
}
