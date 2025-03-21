package ru.blackfan.bfscan.parsing.httprequests.processors;

import jadx.api.plugins.input.data.annotations.EncodedValue;
import jadx.api.plugins.input.data.annotations.IAnnotation;
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

public class Struts2Processor implements AnnotationProcessor {

    @Override
    public ArgProcessingState processParameterAnnotations(MultiHTTPRequest request,
            ParameterInfo paramInfo,
            String annotationClass,
            Map<String, EncodedValue> annotationValues,
            List<ILocalVar> localVars,
            int methodArg,
            ArgType var,
            RootNode rn) {
        return ArgProcessingState.NOT_PROCESSED;
    }

    @Override
    public boolean processMethodAnnotations(MultiHTTPRequest request,
            String annotationClass,
            Map<String, EncodedValue> annotationValues,
            RootNode rn) {
        switch (annotationClass) {
            case Constants.Struts.ACTION -> {
                processAction(request, annotationValues);
                return true;
            }
            case Constants.Struts.ACTIONS -> {
                EncodedValue value = AnnotationUtils.getValue(annotationValues, List.of("value"));
                if (value != null) {
                    ArrayList<EncodedValue> pathList = (ArrayList) value.getValue();
                    List<String> paths = new ArrayList();
                    for (EncodedValue path : pathList) {
                        IAnnotation actionAnnotation = (IAnnotation) path.getValue();
                        if (actionAnnotation != null) {
                            EncodedValue actionPath = AnnotationUtils.getValue(actionAnnotation.getValues(), List.of("value"));
                            paths.add(Helpers.stringWrapper(actionPath));
                        }
                    }
                    request.addAdditionalInformation("Struts2 Action");
                    request.setPaths(paths);
                }
                return true;
            }
            default -> {
                return false;
            }
        }
    }

    private void processAction(MultiHTTPRequest request, Map<String, EncodedValue> annotationValues) {
        EncodedValue value = AnnotationUtils.getValue(annotationValues, List.of("value", "a"));
        if (value != null) {
            request.addAdditionalInformation("Struts2 Action");
            request.setPath(Helpers.stringWrapper(value), false);
        }
    }

    @Override
    public boolean processClassAnnotations(MultiHTTPRequest request,
            String annotationClass,
            Map<String, EncodedValue> annotationValues,
            String globalBasePath
    ) {
        switch (annotationClass) {
            case Constants.Struts.NAMESPACE -> {
                EncodedValue value = AnnotationUtils.getValue(annotationValues, List.of("value", "path"));
                if (value != null) {
                    String classPath = Helpers.stringWrapper(value);
                    String fullPath = (classPath.startsWith("/") ? globalBasePath.substring(0, globalBasePath.length() - 1) : globalBasePath) + classPath;
                    request.setBasePaths(List.of(fullPath));
                }
                return false;
            }
            case Constants.Struts.NAMESPACES -> {
                EncodedValue value = AnnotationUtils.getValue(annotationValues, List.of("value"));
                if (value != null) {
                    ArrayList<EncodedValue> pathList = (ArrayList) value.getValue();
                    List<String> paths = new ArrayList();
                    for (EncodedValue path : pathList) {
                        IAnnotation namespaceAnnotation = (IAnnotation) path.getValue();
                        if (namespaceAnnotation != null) {
                            EncodedValue namespacePath = AnnotationUtils.getValue(namespaceAnnotation.getValues(), List.of("value"));
                            paths.add(Helpers.stringWrapper(namespacePath));
                        }
                    }
                    request.setBasePaths(paths);
                }
                return false;
            }
            case Constants.Struts.ACTION -> {
                processAction(request, annotationValues);
                return true;
            }
            case Constants.Struts.ACTIONS -> {
                request.addAdditionalInformation("Struts2 Action");
                EncodedValue value = AnnotationUtils.getValue(annotationValues, List.of("value"));
                if (value != null) {
                    ArrayList<EncodedValue> pathList = (ArrayList) value.getValue();
                    List<String> paths = new ArrayList();
                    for (EncodedValue path : pathList) {
                        IAnnotation actionAnnotation = (IAnnotation) path.getValue();
                        if (actionAnnotation != null) {
                            EncodedValue actionPath = AnnotationUtils.getValue(actionAnnotation.getValues(), List.of("value"));
                            paths.add(Helpers.stringWrapper(actionPath));
                        }
                    }
                    request.setPaths(paths);
                }
                return true;
            }

            default -> {
                return false;
            }
        }
    }
}
