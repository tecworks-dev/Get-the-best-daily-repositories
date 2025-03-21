package ru.blackfan.bfscan.parsing.httprequests.processors;

import jadx.api.plugins.input.data.annotations.EncodedValue;
import jadx.api.plugins.input.data.annotations.JadxAnnotation;
import jadx.api.plugins.input.data.ILocalVar;
import jadx.core.dex.instructions.args.ArgType;
import jadx.core.dex.nodes.RootNode;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import ru.blackfan.bfscan.helpers.Helpers;
import ru.blackfan.bfscan.parsing.httprequests.Constants;
import ru.blackfan.bfscan.parsing.httprequests.MultiHTTPRequest;
import ru.blackfan.bfscan.parsing.httprequests.ParameterInfo;

public class MicronautProcessor implements AnnotationProcessor {

    @Override
    public ArgProcessingState processParameterAnnotations(MultiHTTPRequest request,
            ParameterInfo paramInfo,
            String annotationClass,
            Map<String, EncodedValue> annotationValues,
            List<ILocalVar> localVars,
            int methodArg,
            ArgType argType,
            RootNode rootNode) throws Exception {
        switch (annotationClass) {
            case Constants.Micronaut.QUERY_VALUE -> {
                String paramName = AnnotationUtils.getParamName(AnnotationUtils.getValue(annotationValues, List.of("value")), paramInfo, localVars, methodArg);
                if (annotationValues.get("defaultValue") != null) {
                    paramInfo.setDefaultValue(Helpers.stringWrapper(annotationValues.get("defaultValue")));
                }
                AnnotationUtils.processQueryParameter(request, paramName, paramInfo.getDefaultValue(), argType, rootNode);
                return ArgProcessingState.PARAMETER_CREATED;
            }
            case Constants.Micronaut.PATH_VARIABLE -> {
                String paramName = AnnotationUtils.getParamName(AnnotationUtils.getValue(annotationValues, List.of("value", "name")), paramInfo, localVars, methodArg);
                if (paramName != null) {
                    request.addPathParameter(paramName);
                }
                return ArgProcessingState.PARAMETER_CREATED;
            }
            case Constants.Micronaut.HEADER -> {
                String paramName = AnnotationUtils.getParamName(AnnotationUtils.getValue(annotationValues, List.of("value")), paramInfo, localVars, methodArg);
                if (annotationValues.get("defaultValue") != null) {
                    paramInfo.setDefaultValue(Helpers.stringWrapper(annotationValues.get("defaultValue")));
                }
                AnnotationUtils.processHeader(request, paramName, paramInfo.getDefaultValue());
                return ArgProcessingState.PARAMETER_CREATED;
            }
            case Constants.Micronaut.PART -> {
                String paramName = AnnotationUtils.getParamName(AnnotationUtils.getValue(annotationValues, List.of("value")), paramInfo, localVars, methodArg);
                AnnotationUtils.processPart(request, paramName);
                return ArgProcessingState.PARAMETER_CREATED;
            }
            case Constants.Micronaut.BODY -> {
                String paramName = AnnotationUtils.getParamName(null, paramInfo, localVars, methodArg);
                AnnotationUtils.processArbitraryBodyParameter(request, paramName, argType, rootNode);
                return ArgProcessingState.PARAMETER_CREATED;
            }
            case Constants.Micronaut.COOKIE_VALUE -> {
                String paramName = AnnotationUtils.getParamName(AnnotationUtils.getValue(annotationValues, List.of("value")), paramInfo, localVars, methodArg);
                if (annotationValues.get("defaultValue") != null) {
                    paramInfo.setDefaultValue(Helpers.stringWrapper(annotationValues.get("defaultValue")));
                }
                AnnotationUtils.processCookieParameter(request, paramName, paramInfo.getDefaultValue());
                return ArgProcessingState.PARAMETER_CREATED;
            }
            default -> {
                return ArgProcessingState.NOT_PROCESSED;
            }
        }
    }

    @Override
    public boolean processMethodAnnotations(MultiHTTPRequest request,
            String annotationClass,
            Map<String, EncodedValue> annotationValues,
            RootNode rn) {
        switch (annotationClass) {
            case Constants.Micronaut.TRACE, Constants.Micronaut.HEAD, Constants.Micronaut.GET, Constants.Micronaut.POST, Constants.Micronaut.PUT, Constants.Micronaut.DELETE, Constants.Micronaut.OPTIONS, Constants.Micronaut.PATCH -> {
                request.addAdditionalInformation("Micronaut Method");
                request.setMethod(Helpers.classSigToRawShortName(annotationClass).toUpperCase());
                processCommonHttpMethodAnnotations(request, annotationValues);
                return true;
            }
            case Constants.Micronaut.CUSTOM_HTTP_METHOD -> {
                request.addAdditionalInformation("Micronaut Method");
                if (annotationValues.get("method") != null) {
                    request.setMethod(Helpers.stringWrapper(annotationValues.get("method")));
                }
                processCommonHttpMethodAnnotations(request, annotationValues);
                return true;
            }
            case Constants.Micronaut.HEADER -> {
                processHeaderAnnotation(request, annotationValues);
                return true;
            }
            case Constants.Micronaut.HEADERS -> {
                EncodedValue value = AnnotationUtils.getValue(annotationValues, List.of("value"));
                if (value != null) {
                    ArrayList<EncodedValue> headers = (ArrayList) value.getValue();
                    if (headers != null) {
                        for (EncodedValue header : headers) {
                            processHeaderAnnotation(request, ((JadxAnnotation) header.getValue()).getValues());
                        }
                    }
                }
                return true;
            }
            case Constants.Micronaut.CONSUMES -> {
                EncodedValue value = AnnotationUtils.getValue(annotationValues, List.of("value", "a"));
                if (value != null) {
                    ArrayList<EncodedValue> consumesList = (ArrayList) value.getValue();
                    if (!consumesList.isEmpty()) {
                        AnnotationUtils.processContentTypeFromList(request, consumesList);
                    }
                }
                return true;
            }
            default -> {
                return false;
            }
        }
    }

    private void processHeaderAnnotation(MultiHTTPRequest request, Map<String, EncodedValue> annotationValues) {
        EncodedValue value = AnnotationUtils.getValue(annotationValues, List.of("name"));
        if (value != null) {
            String headerName = Helpers.stringWrapper(value);
            String headerValue = "";
            if ((value = AnnotationUtils.getValue(annotationValues, List.of("value"))) != null) {
                headerValue = Helpers.stringWrapper(value);
            }
            if ((value = AnnotationUtils.getValue(annotationValues, List.of("defaultValue"))) != null) {
                headerValue = Helpers.stringWrapper(value);
            }
            request.putHeader(headerName, headerValue.equals("") ? headerName : headerValue);
        }
    }

    private void processCommonHttpMethodAnnotations(MultiHTTPRequest request, Map<String, EncodedValue> annotationValues) {
        EncodedValue value = AnnotationUtils.getValue(annotationValues, List.of("uris"));
        if (value != null) {
            ArrayList<EncodedValue> pathlist = (ArrayList) value.getValue();
            List<String> controllerPaths = new ArrayList();
            for (EncodedValue path : pathlist) {
                controllerPaths.add(Helpers.stringWrapper(path));
            }
            request.setPaths(controllerPaths);
        }
        value = AnnotationUtils.getValue(annotationValues, List.of("uri", "value"));
        if (value != null) {
            request.setPath(Helpers.stringWrapper(value), false);
        }
        value = AnnotationUtils.getValue(annotationValues, List.of("consumes", "processes"));
        if (value != null) {
            ArrayList<EncodedValue> consumesList = (ArrayList) value.getValue();
            if (consumesList != null && !consumesList.isEmpty()) {
                AnnotationUtils.processContentTypeFromList(request, consumesList);
            }
        }
    }

    @Override
    public boolean processClassAnnotations(MultiHTTPRequest request,
            String annotationClass,
            Map<String, EncodedValue> annotationValues,
            String globalBasePath) {
        switch (annotationClass) {
            case Constants.Micronaut.HEADER -> {
                processHeaderAnnotation(request, annotationValues);
                return false;
            }
            case Constants.Micronaut.HEADERS -> {
                EncodedValue value = AnnotationUtils.getValue(annotationValues, List.of("value"));
                if (value != null) {
                    ArrayList<EncodedValue> headers = (ArrayList) value.getValue();
                    if (headers != null) {
                        for (EncodedValue header : headers) {
                            processHeaderAnnotation(request, ((JadxAnnotation) header.getValue()).getValues());
                        }
                    }
                }
                return false;
            }
            case Constants.Micronaut.CONTROLLER -> {
                EncodedValue value = AnnotationUtils.getValue(annotationValues, List.of("value"));
                if (value != null) {
                    String classPath = Helpers.stringWrapper(value);
                    String fullPath = (classPath.startsWith("/") ? globalBasePath.substring(0, globalBasePath.length() - 1) : globalBasePath) + classPath;
                    request.setBasePaths(List.of(fullPath));
                }
                value = annotationValues.get("consumes");
                if (value != null) {
                    ArrayList<EncodedValue> consumesList = (ArrayList) value.getValue();
                    if (consumesList != null && !consumesList.isEmpty()) {
                        AnnotationUtils.processContentTypeFromList(request, consumesList);
                    }
                }
                return false;
            }
            case Constants.Micronaut.CLIENT -> {
                EncodedValue value = AnnotationUtils.getValue(annotationValues, List.of("value", "path"));
                if (value != null) {
                    String classPath = Helpers.stringWrapper(value);
                    String fullPath = (classPath.startsWith("/") ? globalBasePath.substring(0, globalBasePath.length() - 1) : globalBasePath) + classPath;
                    request.setBasePaths(List.of(fullPath));
                }
                return false;
            }

            default -> {
                return false;
            }
        }
    }
}
