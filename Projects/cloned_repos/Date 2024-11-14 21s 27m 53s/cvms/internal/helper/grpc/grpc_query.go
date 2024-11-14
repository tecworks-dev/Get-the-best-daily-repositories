package grpchelper

import (
	"context"

	"github.com/golang/protobuf/jsonpb"
	"github.com/jhump/protoreflect/desc"
	"github.com/jhump/protoreflect/dynamic/grpcdynamic"
	"github.com/pkg/errors"

	"log"

	"github.com/jhump/protoreflect/grpcreflect"
	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"
)

var (
	jsonPbMarshaler = jsonpb.Marshaler{
		OrigName:     true,
		EmitDefaults: true,
		AnyResolver:  DynamicAnyResolver{},
	}

	jsonPbUnmarshaler = jsonpb.Unmarshaler{
		AllowUnknownFields: true,
	}
)

func GrpcMakeDescriptor(reflectionClient *grpcreflect.Client, queryPath string) (*desc.MethodDescriptor, error) {
	methodDescriptor, err := ResolveMessage(queryPath, reflectionClient)
	if err != nil {
		return nil, errors.Wrapf(err, "by query path: %s", queryPath)
	}

	return methodDescriptor, nil
}

func GrpcInvokeQuery(
	ctx context.Context,
	methodDescriptor *desc.MethodDescriptor,
	stub *grpcdynamic.Stub,
	queryData string,
) (string, error) {

	msg, err := CreateMessage(methodDescriptor, &jsonPbUnmarshaler, queryData)
	if err != nil {
		// client.Logger.Errorf("grpc api failed to create proto message: %v", err)
		return "", err
	}

	_, err = msg.MarshalJSONPB(&jsonPbMarshaler)
	if err != nil {
		// client.Logger.Errorf("grpc api failed to mashal jsonpb: %v", err)
		return "", err
	}

	var headerMD metadata.MD

	resp, err := stub.InvokeRpc(context.Background(), methodDescriptor, msg, grpc.Header(&headerMD))
	if err != nil {
		// client.Logger.Errorf("grpc api failed to invoke rpc: %v", err)
		return "", err
	}

	respJSON, err := jsonPbMarshaler.MarshalToString(resp)
	if err != nil {
		// client.Logger.Errorf("grpc api failed to marshal string with grpc data: %v", err)
		return "", err
	}

	return respJSON, nil
}

func GrpcDynamicQuery(ctx context.Context, reflectionClient *grpcreflect.Client, stub *grpcdynamic.Stub, queryPath string, queryData string) (string, error) {
	methodDescriptor, err := ResolveMessage(queryPath, reflectionClient)
	if err != nil {
		log.Printf("grpc api failed to resolve proto message: %s", err.Error())
		return "", err
	}

	msg, err := CreateMessage(methodDescriptor, &jsonPbUnmarshaler, queryData)
	if err != nil {
		log.Printf("grpc api failed to create proto message: %s", err.Error())
		return "", err
	}

	_, err = msg.MarshalJSONPB(&jsonPbMarshaler)
	if err != nil {
		log.Printf("grpc api failed to mashal jsonpb: %s", err.Error())
		return "", err
	}

	var headerMD metadata.MD
	resp, err := stub.InvokeRpc(ctx, methodDescriptor, msg, grpc.Header(&headerMD))
	if err != nil {
		log.Printf("grpc api failed to invoke rpc: %s", resp)
		return "", err
	}

	respJSON, err := jsonPbMarshaler.MarshalToString(resp)
	if err != nil {
		log.Printf("grpc api failed to marshal string with grpc data: %s", err.Error())
		return "", err
	}

	return respJSON, nil
}
