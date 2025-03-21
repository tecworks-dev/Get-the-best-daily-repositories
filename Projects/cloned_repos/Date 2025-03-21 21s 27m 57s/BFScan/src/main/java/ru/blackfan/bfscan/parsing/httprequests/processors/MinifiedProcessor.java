package ru.blackfan.bfscan.parsing.httprequests.processors;

import jadx.api.plugins.input.data.annotations.EncodedType;
import jadx.api.plugins.input.data.annotations.EncodedValue;
import jadx.api.plugins.input.data.ILocalVar;
import jadx.core.dex.instructions.args.ArgType;
import jadx.core.dex.nodes.RootNode;
import java.util.List;
import java.util.Map;
import ru.blackfan.bfscan.helpers.Helpers;
import ru.blackfan.bfscan.parsing.httprequests.MultiHTTPRequest;
import ru.blackfan.bfscan.parsing.httprequests.ParameterInfo;

public class MinifiedProcessor {

    public static ArgProcessingState processParameterAnnotations(MultiHTTPRequest request,
            ParameterInfo paramInfo,
            String annotationClass,
            Map<String, EncodedValue> annotationValues,
            List<ILocalVar> localVars,
            int methodArg,
            ArgType argType,
            RootNode rootNode) {

        EncodedValue name = AnnotationUtils.getValue(annotationValues, List.of("value", "a"));
        if (name != null && name.getType() == EncodedType.ENCODED_STRING) {
            String paramName = AnnotationUtils.getParamName(name, paramInfo, localVars, methodArg);
            if(Helpers.isValidRequestHeader(paramName)) {
                AnnotationUtils.processHeader(request, paramName, paramInfo.getDefaultValue());
                return ArgProcessingState.PARAMETER_CREATED;
            } else {
                AnnotationUtils.processQueryParameter(request, paramName, paramInfo.getDefaultValue(), argType, rootNode);
                return ArgProcessingState.PARAMETER_CREATED;
            }
        }
        return ArgProcessingState.NOT_PROCESSED;
    }

    public static boolean processMethodAnnotations(MultiHTTPRequest request,
            String annotationClass,
            Map<String, EncodedValue> annotationValues,
            RootNode rn) {

        EncodedValue value = AnnotationUtils.getValue(annotationValues, List.of("path", "value", "a"));
        if (value != null) {
            String path = Helpers.stringWrapper(value);
            if (Helpers.isValidUrlPath(path)
                    || (Helpers.isValidUrlPath("/" + path) && path.contains("/"))) {
                request.addAdditionalInformation("Minified annotations (experimental support)");
                request.setPath(Helpers.stringWrapper(value), true);
                return true;
            }
        }

        return false;
    }
}
