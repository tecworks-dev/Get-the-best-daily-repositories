package ru.blackfan.bfscan.parsing.httprequests.processors;

import jadx.api.plugins.input.data.annotations.EncodedValue;
import jadx.api.plugins.input.data.annotations.JadxAnnotation;
import jadx.api.plugins.input.data.ILocalVar;
import jadx.core.dex.instructions.args.ArgType;
import jadx.core.dex.nodes.ClassNode;
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

public class OpenApiProcessor implements AnnotationProcessor {

    @Override
    public ArgProcessingState processParameterAnnotations(MultiHTTPRequest request,
            ParameterInfo paramInfo,
            String annotationClass,
            Map<String, EncodedValue> annotationValues,
            List<ILocalVar> localVars,
            int methodArg,
            ArgType var,
            RootNode rootNode) throws Exception {
        switch (annotationClass) {
            case Constants.OpenApi.PARAMETER, Constants.MicroProfileOpenApi.PARAMETER -> {
                processParameter(request, annotationValues, paramInfo, rootNode);
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
            RootNode rootNode) throws Exception {
        switch (annotationClass) {
            case Constants.OpenApi.PARAMETERS, Constants.MicroProfileOpenApi.PARAMETERS -> {
                EncodedValue value;
                if ((value = AnnotationUtils.getValue(annotationValues, List.of("value"))) != null) {
                    ArrayList<EncodedValue> parameters = (ArrayList) value.getValue();
                    for (EncodedValue parameter : parameters) {
                        if (parameter.getValue() != null) {
                            JadxAnnotation parameterAnnotation = (JadxAnnotation) parameter.getValue();
                            processParameter(request, parameterAnnotation.getValues(), null, rootNode);
                        }
                    }
                }
                return true;
            }
            case Constants.OpenApi.PARAMETER, Constants.MicroProfileOpenApi.PARAMETER -> {
                processParameter(request, annotationValues, null, rootNode);
                return true;
            }
            case Constants.OpenApi.REQUEST_BODY, Constants.MicroProfileOpenApi.REQUEST_BODY -> {
                processRequestBody(request, annotationValues, rootNode);
                return true;
            }
            case Constants.OpenApi.SERVER, Constants.MicroProfileOpenApi.SERVER -> {
                processServer(request, annotationValues, false, "");
                return true;
            }
            case Constants.OpenApi.SERVERS, Constants.MicroProfileOpenApi.SERVERS -> {
                processServers(request, annotationValues, false, "", "value");
                return true;
            }
            case Constants.OpenApi.OPERATION, Constants.MicroProfileOpenApi.OPERATION -> {
                processOperation(request, annotationValues, null, rootNode);
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
            case Constants.OpenApi.SERVER, Constants.MicroProfileOpenApi.SERVER -> {
                processServer(request, annotationValues, true, globalBasePath);
                return false;
            }
            case Constants.OpenApi.SERVERS, Constants.MicroProfileOpenApi.SERVERS -> {
                processServers(request, annotationValues, true, globalBasePath, "value");
                return false;
            }
            case Constants.OpenApi.OPENAPID_DEFINITION, Constants.MicroProfileOpenApi.OPENAPID_DEFINITION -> {
                processServers(request, annotationValues, true, globalBasePath, "servers");
                EncodedValue value = AnnotationUtils.getValue(annotationValues, List.of("info"));
                if (value != null) {
                    JadxAnnotation info = (JadxAnnotation) value.getValue();
                    processInfo(request, info.getValues());
                }
                return false;
            }
            default -> {
                return false;
            }
        }
    }

    private void processInfo(MultiHTTPRequest request,
            Map<String, EncodedValue> annotationValues) {

        EncodedValue value;
        if ((value = AnnotationUtils.getValue(annotationValues, List.of("title"))) != null) {
            request.addAdditionalInformation("Info Title: " + Helpers.stringWrapper(value));
        }
        if ((value = AnnotationUtils.getValue(annotationValues, List.of("description"))) != null) {
            request.addAdditionalInformation("Info Description: " + Helpers.stringWrapper(value));
        }
        if ((value = AnnotationUtils.getValue(annotationValues, List.of("summary"))) != null) {
            request.addAdditionalInformation("Info Summary: " + Helpers.stringWrapper(value));
        }
    }

    private void processOperation(MultiHTTPRequest request,
            Map<String, EncodedValue> annotationValues,
            ParameterInfo paramInfo,
            RootNode rootNode) {

        EncodedValue value;
        if ((value = AnnotationUtils.getValue(annotationValues, List.of("method"))) != null) {
            request.setMethod(Helpers.stringWrapper(value));
        }
        if ((value = AnnotationUtils.getValue(annotationValues, List.of("description"))) != null) {
            request.addAdditionalInformation("Operation Description: " + Helpers.stringWrapper(value));
        }
        if ((value = AnnotationUtils.getValue(annotationValues, List.of("summary"))) != null) {
            request.addAdditionalInformation("Operation Summary: " + Helpers.stringWrapper(value));
        }
        if ((value = AnnotationUtils.getValue(annotationValues, List.of("requestBody"))) != null) {
            JadxAnnotation requestBody = (JadxAnnotation) value.getValue();
            processRequestBody(request, requestBody.getValues(), rootNode);
        }
        if ((value = AnnotationUtils.getValue(annotationValues, List.of("parameters"))) != null) {
            ArrayList<EncodedValue> parameters = (ArrayList) value.getValue();

            for (EncodedValue parameter : parameters) {
                if (parameter.getValue() != null) {
                    JadxAnnotation parameterAnnotation = (JadxAnnotation) parameter.getValue();
                    processParameter(request, parameterAnnotation.getValues(), paramInfo, rootNode);
                }
            }
        }
        if ((value = AnnotationUtils.getValue(annotationValues, List.of("servers"))) != null) {
            ArrayList<EncodedValue> servers = (ArrayList) value.getValue();
            if (!servers.isEmpty()) {
                JadxAnnotation serverAnnotation = (JadxAnnotation) servers.get(0).getValue();
                processServer(request, serverAnnotation.getValues(), false, "");
            }
        }
        request.addAdditionalInformation("OpenApi Operation");
    }

    private void processParameter(MultiHTTPRequest request,
            Map<String, EncodedValue> annotationValues,
            ParameterInfo paramInfo,
            RootNode rootNode) {
        EncodedValue value = AnnotationUtils.getValue(annotationValues, List.of("name"));
        if (value != null) {
            String parameterName = Helpers.stringWrapper(value);
            if (paramInfo != null) {
                paramInfo.setName(parameterName);
            }
            value = AnnotationUtils.getValue(annotationValues, List.of("schema"));
            if (value != null) {
                JadxAnnotation schema = (JadxAnnotation) value.getValue();
                processSchema(request, schema.getValues(), paramInfo, rootNode);
            }
            value = AnnotationUtils.getValue(annotationValues, List.of("content"));
            if (value != null) {
                ArrayList<EncodedValue> contents = (ArrayList) value.getValue();
                if (!contents.isEmpty()) {
                    Map<String, EncodedValue> contentValues = ((JadxAnnotation) contents.get(0).getValue()).getValues();
                    processContent(request, contentValues, null, rootNode);
                }
            }
            value = AnnotationUtils.getValue(annotationValues, List.of("examples"));
            if (value != null) {
                ArrayList<EncodedValue> examples = (ArrayList) value.getValue();
                if (!examples.isEmpty()) {
                    JadxAnnotation exampleObject = (JadxAnnotation) examples.get(0).getValue();
                    processExampleObject(request, exampleObject.getValues(), paramInfo);
                }
            }
            value = AnnotationUtils.getValue(annotationValues, List.of("example"));
            if (value != null) {
                if (paramInfo != null) {
                    paramInfo.setDefaultValue(Helpers.stringWrapper(value));
                }
            }

        }
    }

    private void processRequestBody(MultiHTTPRequest request,
            Map<String, EncodedValue> annotationValues,
            RootNode rootNode) {
        EncodedValue value = AnnotationUtils.getValue(annotationValues, List.of("content"));
        if (value != null) {
            ArrayList<EncodedValue> contents = (ArrayList) value.getValue();
            for (EncodedValue content : contents) {
                if (content.getValue() != null) {
                    Map<String, EncodedValue> contentValues = ((JadxAnnotation) content.getValue()).getValues();
                    processContent(request, contentValues, null, rootNode);
                }
            }
        }
    }

    private void processExampleObject(MultiHTTPRequest request,
            Map<String, EncodedValue> annotationValues,
            ParameterInfo paramInfo) {
        EncodedValue value = AnnotationUtils.getValue(annotationValues, List.of("value"));
        if (value != null) {
            String parameterValue = Helpers.stringWrapper(value);
            if (paramInfo != null) {
                paramInfo.setDefaultValue(parameterValue);
            }
        }
    }

    private void processContent(MultiHTTPRequest request,
            Map<String, EncodedValue> annotationValues,
            ParameterInfo paramInfo,
            RootNode rootNode) {
        EncodedValue value = AnnotationUtils.getValue(annotationValues, List.of("mediaType"));
        if (value != null) {
            request.putHeader("Content-Type", Helpers.stringWrapper(value));
        }
        if (paramInfo != null) {
            value = AnnotationUtils.getValue(annotationValues, List.of("examples"));
            if (value != null) {
                ArrayList<EncodedValue> examples = (ArrayList) value.getValue();
                if (!examples.isEmpty()) {
                    JadxAnnotation exampleObject = (JadxAnnotation) examples.get(0).getValue();
                    processExampleObject(request, exampleObject.getValues(), paramInfo);
                }
            }
        }
        value = AnnotationUtils.getValue(annotationValues, List.of("schema"));
        if (value != null) {
            JadxAnnotation schema = (JadxAnnotation) value.getValue();
            processSchema(request, schema.getValues(), paramInfo, rootNode);
        }
    }

    private void processSchema(MultiHTTPRequest request,
            Map<String, EncodedValue> annotationValues,
            ParameterInfo paramInfo,
            RootNode rootNode) {
        EncodedValue value = AnnotationUtils.getValue(annotationValues, List.of("name"));
        String parameterName = null;
        if (value != null) {
            parameterName = Helpers.stringWrapper(value);
        }
        value = AnnotationUtils.getValue(annotationValues, List.of("implementation"));
        if (value != null) {
            String className = (String) value.getValue();
            ClassNode classNode = Helpers.loadClass(rootNode, className);
            if (classNode != null) {
                AnnotationUtils.processArbitraryBodyParameter(request, parameterName, classNode.getType(), rootNode);
            }
        }
        value = AnnotationUtils.getValue(annotationValues, List.of("exampleClasses"));

        if (value != null) {
            ArrayList<EncodedValue> exampleClasses = (ArrayList) value.getValue();
            if (!exampleClasses.isEmpty()) {
                String className = (String) exampleClasses.get(0).getValue();
                ClassNode classNode = Helpers.loadClass(rootNode, className);
                if (classNode != null) {
                    AnnotationUtils.processArbitraryBodyParameter(request, parameterName, classNode.getType(), rootNode);
                }
            }
        }
        value = AnnotationUtils.getValue(annotationValues, List.of("defaultValue", "example"));
        if (value != null && paramInfo != null) {
            paramInfo.setDefaultValue(Helpers.stringWrapper(value));
        } else {
            value = AnnotationUtils.getValue(annotationValues, List.of("allowableValues", "examples"));
            if (value != null && paramInfo != null) {
                ArrayList<EncodedValue> examples = (ArrayList) value.getValue();
                if (!examples.isEmpty()) {
                    paramInfo.setDefaultValue(Helpers.stringWrapper(examples.get(0)));
                }
            }
        }
    }

    private void processServers(MultiHTTPRequest request,
            Map<String, EncodedValue> annotationValues,
            boolean isClassAnnotation,
            String globalBasePath,
            String attrName) {
        EncodedValue value = AnnotationUtils.getValue(annotationValues, List.of(attrName));
        if (value != null) {
            ArrayList<EncodedValue> servers = (ArrayList) value.getValue();
            for (EncodedValue server : servers) {
                if (server.getValue() != null) {
                    Map<String, EncodedValue> serverValues = ((JadxAnnotation) server.getValue()).getValues();
                    processServer(request, serverValues, isClassAnnotation, globalBasePath);
                }
            }
        }
    }

    private void processServer(MultiHTTPRequest request,
            Map<String, EncodedValue> annotationValues,
            boolean isClassAnnotation,
            String globalBasePath) {

        EncodedValue value;
        if ((value = AnnotationUtils.getValue(annotationValues, List.of("url"))) != null) {
            String url = Helpers.stringWrapper(value);
            if (url.matches("https?://.*")) {
                try {
                    URI absUrl = new URL(url).toURI();
                    if (absUrl.getPort() != -1) {
                        request.setHost(absUrl.getHost() + ":" + absUrl.getPort());
                    } else {
                        request.setHost(absUrl.getHost());
                    }
                    String path = absUrl.getPath();
                    if ((path != null) && !path.isEmpty() && !path.equals("/")) {
                        if (isClassAnnotation) {
                            String classPath = path;
                            String fullPath = (classPath.startsWith("/") ? globalBasePath.substring(0, globalBasePath.length() - 1) : globalBasePath) + classPath;
                            request.setBasePaths(List.of(fullPath));
                        } else {
                            request.setPath(absUrl.getPath(), false);
                        }
                    }
                } catch (MalformedURLException | URISyntaxException e) {
                }
            } else if (Helpers.isValidUrlPath(url)) {
                if (isClassAnnotation) {
                    String classPath = url;
                    String fullPath = (classPath.startsWith("/") ? globalBasePath.substring(0, globalBasePath.length() - 1) : globalBasePath) + classPath;
                    request.setBasePaths(List.of(fullPath));
                } else {
                    request.setPath(url, false);
                }
            } else {
                request.addAdditionalInformation("Unrecognized Server url attribute: " + Helpers.stringWrapper(value));
            }
        }
        if ((value = AnnotationUtils.getValue(annotationValues, List.of("description"))) != null) {
            request.addAdditionalInformation("Server Description: " + Helpers.stringWrapper(value));
        }
    }
}
