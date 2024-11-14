package helper

import (
	"net/url"
	"strings"

	"github.com/go-playground/validator/v10"
)

func SetHttpURI(rawRPCEndpoint string) string {
	return "http://" + rawRPCEndpoint
}

func UnsetHttpURI(baseURL string) (string, error) {
	type URLRequest struct {
		URL string `validate:"required,url"`
	}

	validate := validator.New()
	request := URLRequest{URL: baseURL}

	// validation before parse
	err := validate.Struct(request)
	if err != nil {
		// for _, err := range err.(validator.ValidationErrors) {
		// 	fmt.Println("Validation failed for field:", err.Field())
		// 	fmt.Println("Condition failed:", err.ActualTag())
		// }
		return "", err
	}

	url, err := url.Parse(baseURL)
	if err != nil {
		return "", err
	}

	// for grpc
	if url.Host == "" {
		return url.String(), nil
	}

	return url.Host, nil
}

func MakeBaseURL(port, ipAddress string) string {
	baseURL := url.URL{
		Scheme: "http",
		Host:   strings.Join([]string{ipAddress, ":", port}, ""),
	}
	return baseURL.String()
}

func ValidateURL(inputURL string) bool {
	_, err := url.ParseRequestURI(inputURL)
	return err == nil
}

func MustExtractHostname(endpoint string) string {
	// Parse the URL
	parsedURL, _ := url.Parse(endpoint)
	return parsedURL.Hostname()
}
