package ru.blackfan.bfscan.parsing.httprequests.processors;

import jadx.api.plugins.input.data.annotations.EncodedValue;
import jadx.api.plugins.input.data.annotations.JadxAnnotation;
import jadx.api.plugins.input.data.ILocalVar;
import jadx.core.dex.instructions.args.ArgType;
import jadx.core.dex.nodes.ClassNode;
import jadx.core.dex.nodes.RootNode;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import ru.blackfan.bfscan.helpers.Helpers;
import ru.blackfan.bfscan.parsing.httprequests.Constants;
import ru.blackfan.bfscan.parsing.httprequests.MultiHTTPRequest;
import ru.blackfan.bfscan.parsing.httprequests.ParameterInfo;

public class SwaggerProcessor implements AnnotationProcessor {

    @Override
    public ArgProcessingState processParameterAnnotations(MultiHTTPRequest request,
            ParameterInfo paramInfo,
            String annotationClass,
            Map<String, EncodedValue> annotationValues,
            List<ILocalVar> localVars,
            int methodArg,
            ArgType var,
            RootNode rn) throws Exception {
        switch (annotationClass) {
            case Constants.Swagger.API_PARAM -> {
                String paramName = AnnotationUtils.getParamName(AnnotationUtils.getValue(annotationValues, List.of("name")), paramInfo, localVars, methodArg);
                if (paramName != null) {
                    String paramValue = paramName;
                    EncodedValue value;
                    if ((value = AnnotationUtils.getValue(annotationValues, List.of("defaultValue", "example", "allowableValues"))) != null) {
                        paramValue = Helpers.stringWrapper(value);
                    }
                    request.putQueryParameter(paramName, paramValue);
                    return ArgProcessingState.PARAMETER_CREATED;
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
            RootNode rn) throws Exception {
        switch (annotationClass) {
            case Constants.Swagger.API -> {
                EncodedValue value;
                if ((value = AnnotationUtils.getValue(annotationValues, List.of("value", "basePath"))) != null) {
                    request.setBasePaths(List.of(Helpers.stringWrapper(value)));
                }
                if ((value = AnnotationUtils.getValue(annotationValues, List.of("description"))) != null) {
                    request.addAdditionalInformation("API Description: " + Helpers.stringWrapper(value));
                }
                if ((value = AnnotationUtils.getValue(annotationValues, List.of("consumes"))) != null) {
                    request.putHeader("Content-Type", Helpers.stringWrapper(value));
                }
                return true;
            }
            case Constants.Swagger.API_OPERATION -> {
                EncodedValue value;
                if ((value = AnnotationUtils.getValue(annotationValues, List.of("value"))) != null) {
                    request.addAdditionalInformation("Operation: " + Helpers.stringWrapper(value));
                }
                if ((value = AnnotationUtils.getValue(annotationValues, List.of("notes"))) != null) {
                    request.addAdditionalInformation("Notes: " + Helpers.stringWrapper(value));
                }
                if ((value = AnnotationUtils.getValue(annotationValues, List.of("httpMethod"))) != null) {
                    request.setMethod(Helpers.stringWrapper(value));
                }
                if ((value = AnnotationUtils.getValue(annotationValues, List.of("consumes"))) != null) {
                    request.putHeader("Content-Type", Helpers.stringWrapper(value));
                }
                request.addAdditionalInformation("Swagger ApiOperation");
                return true;
            }
            case Constants.Swagger.API_IMPLICIT_PARAM -> {
                EncodedValue value = AnnotationUtils.getValue(annotationValues, List.of("name"));
                if (value != null) {
                    String paramName = Helpers.stringWrapper(value);
                    String paramValue = paramName;
                    String paramType = "query";
                    String dataType = null;

                    if ((value = AnnotationUtils.getValue(annotationValues, List.of("defaultValue", "example"))) != null) {
                        paramValue = Helpers.stringWrapper(value);
                    }

                    if ((value = AnnotationUtils.getValue(annotationValues, List.of("paramType"))) != null) {
                        paramType = Helpers.stringWrapper(value);
                    }

                    if ((value = AnnotationUtils.getValue(annotationValues, List.of("dataType"))) != null) {
                        dataType = Helpers.stringWrapper(value);
                    }

                    switch (paramType) {
                        case "query" ->
                            request.putQueryParameter(paramName, paramValue);
                        case "header" ->
                            request.putHeader(paramName, paramValue);
                        case "path" ->
                            request.addPathParameter(paramName);
                        case "form" ->
                            request.putBodyParameter(paramName, paramValue);
                        case "body" -> {
                            ClassNode classNode = Helpers.loadClass(rn, dataType);
                            if(classNode != null) {
                                AnnotationUtils.processArbitraryBodyParameter(request, paramName, classNode.getType(), rn);
                            }
                        }
                    }
                }
                return true;
            }
            case Constants.Swagger.API_IMPLICIT_PARAMS -> {
                EncodedValue value = AnnotationUtils.getValue(annotationValues, List.of("value"));
                if (value != null) {
                    ArrayList<EncodedValue> params = (ArrayList) value.getValue();
                    for (EncodedValue param : params) {
                        if (param.getValue() != null) {
                            Map<String, EncodedValue> paramValues = ((JadxAnnotation) param.getValue()).getValues();
                            processMethodAnnotations(request, Constants.Swagger.API_IMPLICIT_PARAM, paramValues, rn);
                        }
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
            String globalBasePath
    ) {
        switch (annotationClass) {
            case Constants.Swagger.API -> {
                EncodedValue value;
                if ((value = AnnotationUtils.getValue(annotationValues, List.of("value", "basePath"))) != null) {
                    request.setBasePaths(List.of(Helpers.stringWrapper(value)));
                }
                if ((value = AnnotationUtils.getValue(annotationValues, List.of("description"))) != null) {
                    request.addAdditionalInformation("API Description: " + Helpers.stringWrapper(value));
                }
                if ((value = AnnotationUtils.getValue(annotationValues, List.of("consumes"))) != null) {
                    request.putHeader("Content-Type", Helpers.stringWrapper(value));
                }
                return false;
            }
            case Constants.Swagger.API_MODEL -> {
                EncodedValue value;
                if ((value = AnnotationUtils.getValue(annotationValues, List.of("value"))) != null) {
                    request.addAdditionalInformation("Model: " + Helpers.stringWrapper(value));
                }
                if ((value = AnnotationUtils.getValue(annotationValues, List.of("description"))) != null) {
                    request.addAdditionalInformation("Model Description: " + Helpers.stringWrapper(value));
                }
                return false;
            }
            default -> {
                return false;
            }
        }
    }
}
