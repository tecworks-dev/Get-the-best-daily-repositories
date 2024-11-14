package grpchelper

import (
	"fmt"
	"strings"

	"github.com/golang/protobuf/jsonpb"
	"github.com/jhump/protoreflect/desc"
	"github.com/jhump/protoreflect/dynamic"
	"github.com/jhump/protoreflect/grpcreflect"
)

func ResolveMessage(fullMethodName string, rcli *grpcreflect.Client) (*desc.MethodDescriptor, error) {
	// assume that fully-qualified method name cosists of
	// FULL_SERVER_NAME + "." + METHOD_NAME
	// so split the last dot to get service name
	n := strings.LastIndex(fullMethodName, ".")
	if n < 0 {
		return nil, fmt.Errorf("invalid method name: %v", fullMethodName)
	}
	serviceName := fullMethodName[0:n]
	methodName := fullMethodName[n+1:]

	sdesc, err := rcli.ResolveService(serviceName)
	if err != nil {
		return nil, fmt.Errorf("service couldn't be resolve: %v: %v", err, serviceName)
	}

	mdesc := sdesc.FindMethodByName(methodName)
	if mdesc == nil {
		return nil, fmt.Errorf("method couldn't be found")
	}

	return mdesc, nil
}

func CreateMessage(mdesc *desc.MethodDescriptor, unmarshaler *jsonpb.Unmarshaler, inputJsonString string) (*dynamic.Message, error) {
	msg := dynamic.NewMessage(mdesc.GetInputType())

	if err := msg.UnmarshalJSONPB(unmarshaler, []byte(inputJsonString)); err != nil {
		return nil, fmt.Errorf("unmarshal %v", err)
	}
	return msg, nil
}
