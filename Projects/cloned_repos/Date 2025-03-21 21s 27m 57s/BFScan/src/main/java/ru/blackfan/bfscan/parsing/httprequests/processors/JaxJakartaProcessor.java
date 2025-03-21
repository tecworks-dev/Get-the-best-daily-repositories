package ru.blackfan.bfscan.parsing.httprequests.processors;

import jadx.api.plugins.input.data.annotations.EncodedValue;
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

public class JaxJakartaProcessor implements AnnotationProcessor {

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
            case Constants.JaxRs.PARAM_QUERY, Constants.Jakarta.PARAM_QUERY -> {
                String paramName = AnnotationUtils.getParamName(AnnotationUtils.getValue(annotationValues, List.of("value", "a")), paramInfo, localVars, methodArg);
                AnnotationUtils.processQueryParameter(request, paramName, paramInfo.getDefaultValue(), argType, rootNode);
                return ArgProcessingState.PARAMETER_CREATED;
            }
            case Constants.JaxRs.PARAM_PATH, Constants.Jakarta.PARAM_PATH -> {
                String paramName = AnnotationUtils.getParamName(AnnotationUtils.getValue(annotationValues, List.of("value", "a")), paramInfo, localVars, methodArg);
                if (paramName != null) {
                    request.addPathParameter(paramName);
                }
                return ArgProcessingState.PARAMETER_CREATED;
            }
            case Constants.JaxRs.PARAM_HEADER, Constants.Jakarta.PARAM_HEADER -> {
                String paramName = AnnotationUtils.getParamName(AnnotationUtils.getValue(annotationValues, List.of("value", "a")), paramInfo, localVars, methodArg);
                AnnotationUtils.processHeader(request, paramName, paramInfo.getDefaultValue());
                return ArgProcessingState.PARAMETER_CREATED;
            }
            case Constants.JaxRs.PARAM_FORM, Constants.Jakarta.PARAM_FORM -> {
                String paramName = AnnotationUtils.getParamName(AnnotationUtils.getValue(annotationValues, List.of("value", "a")), paramInfo, localVars, methodArg);
                AnnotationUtils.processBodyParameter(request, paramName, paramInfo.getDefaultValue(), argType, rootNode);
                return ArgProcessingState.PARAMETER_CREATED;
            }
            case Constants.JaxRs.PARAM_COOKIE, Constants.Jakarta.PARAM_COOKIE -> {
                String paramName = AnnotationUtils.getParamName(AnnotationUtils.getValue(annotationValues, List.of("value", "a")), paramInfo, localVars, methodArg);
                AnnotationUtils.processCookieParameter(request, paramName, paramInfo.getDefaultValue());
                return ArgProcessingState.PARAMETER_CREATED;
            }
            case Constants.JaxRs.PARAM_MATRIX, Constants.Jakarta.PARAM_MATRIX -> {
                return ArgProcessingState.PROCESSED_NO_PARAMETER;
            }
            case Constants.JaxRs.PARAM_BEAN, Constants.Jakarta.PARAM_BEAN -> {
                AnnotationUtils.processArbitraryBodyParameter(request, null, argType, rootNode);
                return ArgProcessingState.PARAMETER_CREATED;
            }
            case Constants.JaxRs.DEFAULT_VALUE, Constants.Jakarta.DEFAULT_VALUE -> {
                EncodedValue defaultValue = AnnotationUtils.getValue(annotationValues, List.of("value", "a"));
                if (defaultValue != null) {
                    paramInfo.setDefaultValue(Helpers.stringWrapper(defaultValue));
                }
                return ArgProcessingState.PROCESSED_NO_PARAMETER;
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
            case Constants.JaxRs.GET, Constants.Jakarta.GET, Constants.JaxRs.POST, Constants.Jakarta.POST, Constants.JaxRs.PUT, Constants.Jakarta.PUT, Constants.JaxRs.DELETE, Constants.Jakarta.DELETE, Constants.JaxRs.HEAD, Constants.Jakarta.HEAD, Constants.JaxRs.OPTIONS, Constants.Jakarta.OPTIONS -> {
                request.setMethod(Helpers.classSigToRawShortName(annotationClass));
                request.addAdditionalInformation("JAX-RS/Jakarta Method");
                return true;
            }
            case Constants.JaxRs.METHOD, Constants.Jakarta.METHOD -> {
                request.addAdditionalInformation("Jax/Jakarta Method");
                EncodedValue value = AnnotationUtils.getValue(annotationValues, List.of("value", "a"));
                if (value != null) {
                    request.setMethod(Helpers.stringWrapper(value));
                }
                request.addAdditionalInformation("JAX-RS/Jakarta Method");
                return true;
            }
            case Constants.JaxRs.PATH, Constants.Jakarta.PATH -> {
                EncodedValue value = AnnotationUtils.getValue(annotationValues, List.of("value", "a"));
                if (value != null) {
                    request.setPath(Helpers.stringWrapper(value), false);
                }
                return true;
            }
            case Constants.JaxRs.PRODUCES, Constants.Jakarta.PRODUCES -> {
                return true;
            }
            case Constants.JaxRs.CONSUMES, Constants.Jakarta.CONSUMES -> {
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

    @Override
    public boolean processClassAnnotations(MultiHTTPRequest request,
            String annotationClass,
            Map<String, EncodedValue> annotationValues,
            String globalBasePath) {
        switch (annotationClass) {
            case Constants.JaxRs.PATH, Constants.Jakarta.PATH -> {
                EncodedValue value = AnnotationUtils.getValue(annotationValues, List.of("value", "a"));
                if (value != null) {
                    String classPath = Helpers.stringWrapper(value);
                    String fullPath = (classPath.startsWith("/") ? globalBasePath.substring(0, globalBasePath.length() - 1) : globalBasePath) + classPath;
                    request.setBasePaths(List.of(fullPath));
                }
                return false;
            }
            case Constants.JaxRs.WEBFILTER, Constants.Jakarta.WEBFILTER, Constants.JaxRs.WEBSERVLET, Constants.Jakarta.WEBSERVLET -> {
                request.addAdditionalInformation("Jax/Jakarta Webfilter/Webservlet");
                EncodedValue value = AnnotationUtils.getValue(annotationValues, List.of("value", "urlPatterns"));
                if (value != null) {
                    ArrayList<EncodedValue> pathList = (ArrayList) value.getValue();
                    List<String> paths = new ArrayList();
                    for (EncodedValue path : pathList) {
                        String classPath = Helpers.stringWrapper(path);
                        paths.add(classPath);
                    }
                    request.setPaths(paths);
                }
                return true;
            }
            case Constants.JaxRs.CONSUMES, Constants.Jakarta.CONSUMES -> {
                EncodedValue value = AnnotationUtils.getValue(annotationValues, List.of("value", "a"));
                if (value != null) {
                    ArrayList<EncodedValue> consumesList = (ArrayList) value.getValue();
                    if (!consumesList.isEmpty()) {
                        AnnotationUtils.processContentTypeFromList(request, consumesList);
                    }
                }
                return false;
            }
            default -> {
                return false;
            }
        }
    }
}
