package ru.blackfan.bfscan.parsing.httprequests.processors;

import jadx.api.plugins.input.data.annotations.EncodedValue;
import jadx.api.plugins.input.data.ILocalVar;
import jadx.api.plugins.input.data.impl.JadxFieldRef;
import jadx.core.dex.instructions.args.ArgType;
import jadx.core.dex.nodes.RootNode;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import ru.blackfan.bfscan.helpers.Helpers;
import ru.blackfan.bfscan.parsing.httprequests.Constants;
import ru.blackfan.bfscan.parsing.httprequests.MultiHTTPRequest;
import ru.blackfan.bfscan.parsing.httprequests.ParameterInfo;

public class SpringProcessor implements AnnotationProcessor {

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
            case Constants.Spring.PARAM_HEADER -> {
                String paramName = AnnotationUtils.getParamName(AnnotationUtils.getValue(annotationValues, List.of("value", "name")), paramInfo, localVars, methodArg);
                if (annotationValues.get("defaultValue") != null) {
                    paramInfo.setDefaultValue(Helpers.stringWrapper(annotationValues.get("defaultValue")));
                }
                AnnotationUtils.processHeader(request, paramName, paramInfo.getDefaultValue());
                return ArgProcessingState.PARAMETER_CREATED;
            }
            case Constants.Spring.PARAM_PATH -> {
                String paramName = AnnotationUtils.getParamName(AnnotationUtils.getValue(annotationValues, List.of("value", "name")), paramInfo, localVars, methodArg);
                if (paramName != null) {
                    request.addPathParameter(paramName);
                }
                return ArgProcessingState.PARAMETER_CREATED;
            }
            case Constants.Spring.PARAM_PART -> {
                String paramName = AnnotationUtils.getParamName(AnnotationUtils.getValue(annotationValues, List.of("value", "name")), paramInfo, localVars, methodArg);
                AnnotationUtils.processPart(request, paramName);
                return ArgProcessingState.PARAMETER_CREATED;
            }
            case Constants.Spring.PARAM_BODY, Constants.Spring.PARAM_MODEL -> {
                String paramName = AnnotationUtils.getParamName(null, paramInfo, localVars, methodArg);
                AnnotationUtils.processArbitraryBodyParameter(request, paramName, argType, rootNode);
                return ArgProcessingState.PARAMETER_CREATED;
            }
            case Constants.Spring.BIND_PARAM, Constants.Spring.REQUEST_PARAMETER -> {
                Object paramValue;
                String paramName = AnnotationUtils.getParamName(AnnotationUtils.getValue(annotationValues, List.of("value", "name", "a")), paramInfo, localVars, methodArg);
                if (!argType.isPrimitive() && AnnotationUtils.isMultipartObject(argType.getObject())) {
                    request.setEncType("multipart");
                }
                if (annotationValues.get("defaultValue") != null) {
                    paramValue = Helpers.stringWrapper(annotationValues.get("defaultValue"));
                } else if (!paramInfo.getDefaultValue().isEmpty()) {
                    paramValue = paramInfo.getDefaultValue();
                } else {
                    paramValue = AnnotationUtils.argTypeToValue(paramName, argType, rootNode, new HashSet<>(), true);
                }
                if (paramName != null && paramValue != null) {
                    AnnotationUtils.appendParametersToRequest(request, Map.of(paramName, paramValue));
                    return ArgProcessingState.PARAMETER_CREATED;
                }
                return ArgProcessingState.PROCESSED_NO_PARAMETER;
            }
            case Constants.Spring.PARAM_COOKIE -> {
                String paramName = AnnotationUtils.getParamName(AnnotationUtils.getValue(annotationValues, List.of("value", "name")), paramInfo, localVars, methodArg);
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
            case Constants.Spring.REQUEST_MAPPING, Constants.Spring.GET_MAPPING, Constants.Spring.DELETE_MAPPING, Constants.Spring.PATCH_MAPPING, Constants.Spring.PUT_MAPPING, Constants.Spring.POST_MAPPING -> {
                request.addAdditionalInformation("Spring Method");
                if (!annotationClass.equals(Constants.Spring.REQUEST_MAPPING)) {
                    String alias = Helpers.classSigToRawShortName(annotationClass);
                    request.setMethod(alias.replace("Mapping", "").toUpperCase());
                } else {
                    processHttpMethod(request, annotationValues);
                }
                EncodedValue value = AnnotationUtils.getValue(annotationValues, List.of("value", "path"));
                if (value != null) {
                    ArrayList<EncodedValue> pathlist = (ArrayList) value.getValue();
                    List<String> controllerPaths = new ArrayList();
                    for (EncodedValue path : pathlist) {
                        controllerPaths.add(Helpers.stringWrapper(path));
                    }
                    request.setPaths(controllerPaths);
                }
                processConsumes(request, annotationValues);
                processParams(request, annotationValues);
                processHeaders(request, annotationValues);
                return true;
            }
            default -> {
                return false;
            }
        }
    }

    @Override
    public boolean processClassAnnotations(MultiHTTPRequest request,
            String annotationClass,
            Map<String, EncodedValue> annotationValues,
            String globalBasePath) {
        switch (annotationClass) {
            case Constants.Spring.REQUEST_MAPPING -> {
                EncodedValue value = AnnotationUtils.getValue(annotationValues, List.of("path", "value"));
                if (value != null) {
                    ArrayList<EncodedValue> basePathList = (ArrayList) value.getValue();
                    List<String> basePaths = new ArrayList();
                    for (EncodedValue path : basePathList) {
                        String classPath = Helpers.stringWrapper(path);
                        if (!globalBasePath.equals("/")) {
                            classPath = (classPath.startsWith("/") ? globalBasePath.substring(0, globalBasePath.length() - 1) : globalBasePath) + classPath;
                        }
                        basePaths.add(classPath);
                    }
                    request.setBasePaths(basePaths);
                }
                processHttpMethod(request, annotationValues);
                processConsumes(request, annotationValues);
                processParams(request, annotationValues);
                processHeaders(request, annotationValues);
                return false;
            }
            default -> {
                return false;
            }
        }
    }

    private void processHttpMethod(MultiHTTPRequest request, Map<String, EncodedValue> annotationValues) {
        EncodedValue value = annotationValues.get("method");
        if (value != null) {
            ArrayList<EncodedValue> methodlist = (ArrayList) value.getValue();
            List<String> controllerMethods = new ArrayList();
            for (EncodedValue method : methodlist) {
                controllerMethods.add(((JadxFieldRef) method.getValue()).getName());
            }
            request.setMethods(controllerMethods);
        }
    }

    private void processConsumes(MultiHTTPRequest request, Map<String, EncodedValue> annotationValues) {
        EncodedValue value = annotationValues.get("consumes");
        if (value != null) {
            ArrayList<EncodedValue> consumesList = (ArrayList) value.getValue();
            if (consumesList != null && !consumesList.isEmpty()) {
                AnnotationUtils.processContentTypeFromList(request, consumesList);
            }
        }
    }

    private void processParams(MultiHTTPRequest request, Map<String, EncodedValue> annotationValues) {
        EncodedValue value = annotationValues.get("params");
        if (value != null) {
            ArrayList<EncodedValue> paramlist = (ArrayList) value.getValue();
            for (EncodedValue param : paramlist) {
                String p = Helpers.stringWrapper(param);
                if (p.contains("=")) {
                    String[] parts = p.split("=", 2);
                    request.putQueryParameter(parts[0], parts[1]);
                } else {
                    request.putQueryParameter(p, p);
                }
            }
        }
    }

    private void processHeaders(MultiHTTPRequest request, Map<String, EncodedValue> annotationValues) {
        EncodedValue value = annotationValues.get("headers");
        if (value != null) {
            ArrayList<EncodedValue> headerlist = (ArrayList) value.getValue();
            for (EncodedValue header : headerlist) {
                String h = Helpers.stringWrapper(header);
                if (h.contains("=")) {
                    String[] parts = h.split("=", 2);
                    request.putHeader(parts[0].trim(), parts[1].trim());
                } else {
                    request.putHeader(h, h);
                }
            }
        }
    }
}
