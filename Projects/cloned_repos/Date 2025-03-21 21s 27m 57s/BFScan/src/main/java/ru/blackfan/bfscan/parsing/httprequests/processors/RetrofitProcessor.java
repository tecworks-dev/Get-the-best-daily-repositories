package ru.blackfan.bfscan.parsing.httprequests.processors;

import jadx.api.plugins.input.data.annotations.EncodedValue;
import jadx.api.plugins.input.data.ILocalVar;
import jadx.core.dex.instructions.args.ArgType;
import jadx.core.dex.nodes.RootNode;
import java.net.MalformedURLException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import ru.blackfan.bfscan.helpers.Helpers;
import ru.blackfan.bfscan.parsing.httprequests.Constants;
import ru.blackfan.bfscan.parsing.httprequests.MultiHTTPRequest;
import ru.blackfan.bfscan.parsing.httprequests.ParameterInfo;

public class RetrofitProcessor implements AnnotationProcessor {

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
            case Constants.Retrofit.PARAM_QUERY, Constants.Ktorfit.PARAM_QUERY -> {
                String paramName = AnnotationUtils.getParamName(AnnotationUtils.getValue(annotationValues, List.of("value", "a")), paramInfo, localVars, methodArg);
                AnnotationUtils.processQueryParameter(request, paramName, paramInfo.getDefaultValue(), argType, rootNode);
                return ArgProcessingState.PARAMETER_CREATED;
            }
            case Constants.Retrofit.PARAM_HEADER, Constants.Ktorfit.PARAM_HEADER -> {
                String paramName = AnnotationUtils.getParamName(AnnotationUtils.getValue(annotationValues, List.of("value", "a")), paramInfo, localVars, methodArg);
                AnnotationUtils.processHeader(request, paramName, paramInfo.getDefaultValue());
                return ArgProcessingState.PARAMETER_CREATED;
            }
            case Constants.Retrofit.PARAM_PATH, Constants.Ktorfit.PARAM_PATH -> {
                String paramName = AnnotationUtils.getParamName(AnnotationUtils.getValue(annotationValues, List.of("value", "a")), paramInfo, localVars, methodArg);
                if (paramName != null) {
                    request.addPathParameter(paramName);
                }
                return ArgProcessingState.PARAMETER_CREATED;
            }
            case Constants.Retrofit.PARAM_FIELD, Constants.Ktorfit.PARAM_FIELD -> {
                String paramName = AnnotationUtils.getParamName(AnnotationUtils.getValue(annotationValues, List.of("value", "a")), paramInfo, localVars, methodArg);
                AnnotationUtils.processBodyParameter(request, paramName, paramInfo.getDefaultValue(), argType, rootNode);
                return ArgProcessingState.PARAMETER_CREATED;
            }
            case Constants.Retrofit.PARAM_PART, Constants.Ktorfit.PARAM_PART -> {
                String paramName = AnnotationUtils.getParamName(AnnotationUtils.getValue(annotationValues, List.of("value")), paramInfo, localVars, methodArg);
                AnnotationUtils.processPart(request, paramName);
                return ArgProcessingState.PARAMETER_CREATED;
            }
            case Constants.Retrofit.PARAM_BODY, Constants.Ktorfit.PARAM_BODY -> {
                String paramName = AnnotationUtils.getParamName(null, paramInfo, localVars, methodArg);
                AnnotationUtils.processArbitraryBodyParameter(request, paramName, argType, rootNode);
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

            case Constants.Retrofit.HEAD, Constants.Retrofit.GET, Constants.Retrofit.POST, Constants.Retrofit.PUT, Constants.Retrofit.DELETE, Constants.Retrofit.PATCH, Constants.Retrofit.OPTIONS, Constants.Ktorfit.HEAD, Constants.Ktorfit.GET, Constants.Ktorfit.POST, Constants.Ktorfit.PUT, Constants.Ktorfit.DELETE, Constants.Ktorfit.PATCH, Constants.Ktorfit.OPTIONS -> {
                request.setMethod(Helpers.classSigToRawShortName(annotationClass));
                request.addAdditionalInformation("Retrofit/Ktorfit Method");
                EncodedValue value = AnnotationUtils.getValue(annotationValues, List.of("path", "value", "a"));
                if (value != null) {
                    if (Helpers.stringWrapper(value).matches("https?://.*")) {
                        try {
                            URI absUrl = new URL(Helpers.stringWrapper(value)).toURI();
                            request.setPath(absUrl.getPath(), true);
                            request.setHost(absUrl.getHost());
                        } catch (MalformedURLException | URISyntaxException e) {
                        }
                    } else {
                        request.setPath(Helpers.stringWrapper(value), true);
                    }
                }
                return true;
            }
            case Constants.Retrofit.HTTP, Constants.Ktorfit.HTTP -> {
                request.addAdditionalInformation("Retrofit/Ktorfit Method");
                if (annotationValues.get("method") != null) {
                    request.setMethod(Helpers.stringWrapper(annotationValues.get("method")));
                }
                if (annotationValues.get("path") != null) {
                    request.setPath(Helpers.stringWrapper(annotationValues.get("path")), true);
                }
                return true;
            }
            case Constants.Retrofit.MULTIPART, Constants.Ktorfit.MULTIPART -> {
                request.setEncType("multipart");
                return true;
            }
            case Constants.Retrofit.FORM, Constants.Ktorfit.FORM -> {
                request.setEncType("form");
                return true;
            }
            case Constants.Retrofit.HEADERS, Constants.Ktorfit.HEADERS -> {
                EncodedValue value = AnnotationUtils.getValue(annotationValues, List.of("value", "a"));
                if (value != null) {
                    AnnotationUtils.processHeadersFromList(request, (ArrayList<EncodedValue>) value.getValue());
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
        return false;
    }
}
