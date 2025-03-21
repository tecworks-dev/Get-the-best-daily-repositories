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
import ru.blackfan.bfscan.parsing.httprequests.requestbody.RawRequestBody;

public class FeignProcessor implements AnnotationProcessor {

    @Override
    public ArgProcessingState processParameterAnnotations(MultiHTTPRequest request,
            ParameterInfo paramInfo,
            String annotationClass,
            Map<String, EncodedValue> annotationValues,
            List<ILocalVar> localVars,
            int methodArg,
            ArgType var,
            RootNode rn) {
        switch (annotationClass) {
            case Constants.Feign.QUERYMAP, Constants.Feign.HEADERMAP, Constants.Feign.PARAM -> {
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
            case Constants.Feign.BODY -> {
                EncodedValue value = AnnotationUtils.getValue(annotationValues, List.of("value", "a"));
                if (value != null) {
                    String body = Helpers.stringWrapper(value);
                    if (body.startsWith("<")) {
                        request.setEncType("xml");
                    } else if (body.toLowerCase().startsWith("%7b")) {
                        body = body.replaceAll("(?i)%7B", "{").replaceAll("(?i)%7D", "}");
                        request.setEncType("json");
                    }
                    request.setRequestBody(new RawRequestBody(body));
                }
                return true;
            }
            case Constants.Feign.REQUESTLINE -> {
                request.addAdditionalInformation("Feign Method");
                EncodedValue value = AnnotationUtils.getValue(annotationValues, List.of("value", "a"));
                if (value != null) {
                    String requestLine = Helpers.stringWrapper(value);
                    String[] parts = requestLine.split(" ", 2);
                    String method = parts[0];
                    request.setMethod(method);
                    String path = parts.length > 1 ? parts[1] : "";
                    if (!path.isEmpty()) {
                        request.setPath(path, true);
                    }
                }
                return true;
            }
            case Constants.Feign.HEADERS -> {
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
        switch (annotationClass) {
            case Constants.Feign.CLIENT -> {
                request.addAdditionalInformation("Feign Client");
                EncodedValue value;
                if ((value = AnnotationUtils.getValue(annotationValues, List.of("value", "name"))) != null) {
                    request.addAdditionalInformation("Client Name: " + Helpers.stringWrapper(value));
                }
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
                                String classPath = path;
                                String fullPath = (classPath.startsWith("/") ? globalBasePath.substring(0, globalBasePath.length() - 1) : globalBasePath) + classPath;
                                request.setBasePaths(List.of(fullPath));
                            }
                        } catch (MalformedURLException | URISyntaxException e) {
                        }
                    } else {
                        request.addAdditionalInformation("Unrecognized FeignClient url attribute: " + Helpers.stringWrapper(value));
                    }
                }
                if ((value = AnnotationUtils.getValue(annotationValues, List.of("path"))) != null) {
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
